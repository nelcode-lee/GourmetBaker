"""
Chat interface API routes
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from uuid import UUID
import asyncpg
from app.core.db.database import get_db_pool
from app.core.rag.retriever import HybridRetriever
from app.core.rag.generator import RAGGenerator
from app.api.models.schemas import (
    QueryRequest,
    QueryResponse,
    QueryHistoryItem,
    FeedbackRequest,
    CitationResponse
)

# Try to import EmbeddingService, make it optional
try:
    from app.core.rag.embeddings import EmbeddingService
    embedding_service = EmbeddingService()
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    embedding_service = None
    EMBEDDINGS_AVAILABLE = False

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def send_query(
    request: QueryRequest,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Ask a question and get RAG-generated response
    """
    try:
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(request.query)
        
        # Retrieve relevant chunks (with learning enabled)
        retriever = HybridRetriever(pool, use_feedback=True, use_learning=True)
        retrieved_chunks = await retriever.retrieve(
            query_embedding=query_embedding,
            query_text=request.query,
            top_k=5,
            document_ids=[str(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
            core_area=request.core_area,
            factory=request.factory
        )
        
        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found. Please upload documents first."
            )
        
        # Generate response
        generator = RAGGenerator()
        result = await generator.generate_response(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            conversation_history=request.conversation_history
        )
        
        # Save query and response to database
        import json
        async with pool.acquire() as conn:
            # Convert query embedding to pgvector format
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Insert query and get created_at
            query_row = await conn.fetchrow("""
                INSERT INTO queries (query_text, embedding, user_id)
                VALUES ($1, $2::vector, NULL)
                RETURNING id, created_at
            """, request.query, query_embedding_str)
            query_id = query_row["id"]
            created_at = query_row["created_at"]
            
            # Insert response
            response_id = await conn.fetchval("""
                INSERT INTO responses (query_id, response_text, model, tokens_used, latency_ms)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, query_id, result["response_text"], generator.model, result["tokens_used"], result["latency_ms"])
            
            # Insert citations
            for idx, citation in enumerate(result["citations"]):
                await conn.execute("""
                    INSERT INTO query_citations (query_id, chunk_id, relevance_score, rank)
                    VALUES ($1, $2, $3, $4)
                """, query_id, UUID(citation["chunk_id"]), citation["relevance_score"], idx + 1)
        
        # Format citations for response
        citation_responses = [
            CitationResponse(
                chunk_id=UUID(c["chunk_id"]),
                content=c["content"],
                relevance_score=c["relevance_score"],
                rank=idx + 1,
                metadata=c.get("metadata", {})
            )
            for idx, c in enumerate(result["citations"])
        ]
        
        return QueryResponse(
            query_id=query_id,
            response_text=result["response_text"],
            citations=citation_responses,
            model=generator.model,
            tokens_used=result["tokens_used"],
            latency_ms=result["latency_ms"],
            created_at=created_at,
            groundedness_score=result.get("groundedness_score", 1.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/history", response_model=List[QueryHistoryItem])
async def get_query_history(
    limit: int = Query(50, ge=1, le=100),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get query history
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT q.id, q.query_text, r.response_text, q.created_at, r.feedback
            FROM queries q
            LEFT JOIN responses r ON q.id = r.query_id
            ORDER BY q.created_at DESC
            LIMIT $1
        """, limit)
        
        history = []
        for row in rows:
            history.append(QueryHistoryItem(
                id=row["id"],
                query_text=row["query_text"],
                response_text=row["response_text"] or "",
                created_at=row["created_at"],
                feedback=row["feedback"]
            ))
        
        return history


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Submit thumbs up/down feedback for a response
    """
    async with pool.acquire() as conn:
        # Get response_id from query_id
        response_id = await conn.fetchval("""
            SELECT id FROM responses WHERE query_id = $1
        """, request.query_id)
        
        if not response_id:
            raise HTTPException(status_code=404, detail="Response not found")
        
        # Update feedback
        await conn.execute("""
            UPDATE responses 
            SET feedback = $1
            WHERE id = $2
        """, request.feedback, response_id)
    
        # Trigger learning from this interaction
        from app.core.rag.learning import LearningService
        learning_service = LearningService(pool)
        await learning_service.learn_from_interaction(request.query_id, request.feedback)
    
    return {"message": "Feedback submitted", "query_id": str(request.query_id)}


@router.get("/citations/{query_id}")
async def get_citations(
    query_id: UUID,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get source citations for a query
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT qc.chunk_id, qc.relevance_score, qc.rank,
                   dc.content, dc.metadata
            FROM query_citations qc
            JOIN document_chunks dc ON qc.chunk_id = dc.id
            WHERE qc.query_id = $1
            ORDER BY qc.rank
        """, query_id)
        
        citations = []
        for row in rows:
            citations.append(CitationResponse(
                chunk_id=row["chunk_id"],
                content=row["content"][:500],  # Preview
                relevance_score=float(row["relevance_score"]),
                rank=row["rank"],
                metadata=row["metadata"] or {}
            ))
        
        return {"query_id": str(query_id), "citations": citations}


@router.get("/suggestions")
async def get_query_suggestions(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=10),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get query suggestions based on similar successful queries
    """
    from app.core.rag.learning import LearningService
    
    learning_service = LearningService(pool)
    suggestions = await learning_service.get_query_suggestions(query, limit)
    
    return {"suggestions": suggestions}

