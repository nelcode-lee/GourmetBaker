"""
Chat interface API routes
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from uuid import UUID
import asyncpg
import asyncio
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
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Received query: {request.query[:100]}...")
    
    async def process_query():
        """Inner function to process the query with timeout protection"""
        try:
            # Check cache first
            from app.core.config import get_settings
            from app.core.cache import get_cache_service
            settings = get_settings()
            cache_service = get_cache_service()
            
            # Normalize query for cache lookup (typos should get same cached result)
            from app.core.rag.query_normalization import QueryNormalizer
            normalized_query_for_cache = QueryNormalizer.normalize_query(request.query)
            
            if settings.CACHE_ENABLED:
                # Try cache with normalized query first (handles typos)
                cached_response = await cache_service.get(
                    query=normalized_query_for_cache,
                    document_ids=[str(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
                    core_area=request.core_area,
                    factory=request.factory
                )
                
                # If not found, try original query (for backward compatibility)
                if not cached_response:
                    cached_response = await cache_service.get(
                        query=request.query,
                        document_ids=[str(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
                        core_area=request.core_area,
                        factory=request.factory
                    )
                
                if cached_response:
                    # Get cached confidence score
                    cached_confidence = cached_response.get("overall_confidence")
                    logger.info(f"Returning cached response with confidence: {cached_confidence:.2f}%")
                    
                    # Convert cached data back to QueryResponse
                    from app.api.models.schemas import CitationResponse
                    citation_responses = [
                        CitationResponse(**c) for c in cached_response.get("citations", [])
                    ]
                    from datetime import datetime
                    created_at = None
                    if cached_response.get("created_at"):
                        try:
                            created_at = datetime.fromisoformat(cached_response["created_at"])
                        except:
                            created_at = datetime.now()
                    
                    # Ensure we use the exact cached confidence score
                    overall_confidence = float(cached_confidence) if cached_confidence is not None else 0.0
                    logger.info(f"Using cached confidence: {overall_confidence:.2f} (raw: {cached_confidence})")
                    
                    return QueryResponse(
                        query_id=UUID(cached_response.get("query_id")) if cached_response.get("query_id") else None,
                        response_text=cached_response.get("response_text"),
                        citations=citation_responses,
                        model=cached_response.get("model"),
                        tokens_used=cached_response.get("tokens_used", 0),
                        latency_ms=cached_response.get("latency_ms", 0),
                        created_at=created_at or datetime.now(),
                        groundedness_score=cached_response.get("groundedness_score"),
                        avg_relevance_score=cached_response.get("avg_relevance_score"),
                        overall_confidence=overall_confidence
                    )
            
            # Use normalized query from cache lookup (already computed above)
            normalized_query = normalized_query_for_cache
            if normalized_query != request.query:
                logger.info(f"Query normalized: '{request.query}' -> '{normalized_query}'")
            
            # Generate query embedding using normalized query (more robust)
            logger.info("Generating query embedding...")
            primary_embedding = embedding_service.generate_embedding(normalized_query)
            logger.info("Query embedding generated")
            
            # Retrieve relevant chunks (settings already loaded above)
            logger.info("Initializing retriever...")
            retriever = HybridRetriever(pool, use_feedback=True, use_learning=False)
            logger.info("Retrieving chunks...")
            
            # Primary retrieval - use normalized query for better matching
            retrieved_chunks = await retriever.retrieve(
                query_embedding=primary_embedding,
                query_text=normalized_query,  # Use normalized query for better keyword matching
                top_k=settings.TOP_K_RETRIEVAL,
                document_ids=[str(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
                core_area=request.core_area,
                factory=request.factory
            )
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            
            # Always try query variations if enabled - improves recall significantly
            if settings.USE_QUERY_VARIATIONS:
                # Check if we have high-quality chunks (relevance >= 0.4, more lenient)
                high_quality_count = sum(1 for _, score, _ in retrieved_chunks if score >= 0.4)
                
                # Use variations if we don't have enough high-quality chunks OR if best score is low
                best_score = max((chunk[1] for chunk in retrieved_chunks), default=0.0) if retrieved_chunks else 0.0
                should_use_variations = high_quality_count < 5 or best_score < 0.5
                
                if should_use_variations:
                    logger.info(f"Trying query variations (high quality: {high_quality_count}, best score: {best_score:.3f})...")
                    from app.core.rag.query_understanding import QueryUnderstanding
                    # Use normalized query for variations
                    query_variations = QueryUnderstanding.generate_query_variations(normalized_query)
                    # Use top 3 variations for better coverage
                    query_variations = query_variations[:3]
                    
                    # Generate embeddings and retrieve with variations
                    all_retrieved_chunks = {chunk[0]: chunk for chunk in retrieved_chunks}
                    
                    for variation in query_variations:
                        try:
                            variation_embedding = embedding_service.generate_embedding(variation)
                            variation_chunks = await retriever.retrieve(
                                query_embedding=variation_embedding,
                                query_text=variation,
                                top_k=settings.TOP_K_RETRIEVAL,  # Use full TOP_K for variations
                                document_ids=[str(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
                                core_area=request.core_area,
                                factory=request.factory
                            )
                            # Merge chunks, keeping highest score
                            for chunk_id, score, metadata in variation_chunks:
                                if chunk_id not in all_retrieved_chunks or all_retrieved_chunks[chunk_id][1] < score:
                                    all_retrieved_chunks[chunk_id] = (chunk_id, score, metadata)
                        except Exception as e:
                            logger.warning(f"Failed to retrieve with variation '{variation}': {e}")
                            continue
                    
                    # Convert back to list and sort
                    retrieved_chunks = list(all_retrieved_chunks.values())
                    retrieved_chunks.sort(key=lambda x: x[1], reverse=True)
                    # Allow more chunks after variations (up to 1.5x TOP_K)
                    retrieved_chunks = retrieved_chunks[:int(settings.TOP_K_RETRIEVAL * 1.5)]
                    logger.info(f"Retrieved {len(retrieved_chunks)} chunks after variations")
            
            # Filter chunks by minimum relevance score - very lenient approach
            # Use adaptive threshold that's very forgiving to catch simple questions
            if not retrieved_chunks:
                filtered_chunks = []
            else:
                # Get the best score available
                best_score = max(chunk[1] for chunk in retrieved_chunks)
                
                # Very adaptive threshold: use 50% of best score, but not below 0.15
                # This ensures we use chunks even if scores are lower than expected
                # For simple questions, we want to be very inclusive
                adaptive_threshold = max(best_score * 0.5, 0.15, settings.MIN_RELEVANCE_SCORE)
                
                filtered_chunks = [
                    chunk for chunk in retrieved_chunks 
                    if chunk[1] >= adaptive_threshold
                ]
                
                # If still nothing, use top 10 results regardless of score (increased from 8)
                if not filtered_chunks:
                    filtered_chunks = retrieved_chunks[:10]
                    logger.warning(f"Using top 10 chunks regardless of score - best score was {best_score:.3f}")
                else:
                    logger.info(f"Retrieved {len(retrieved_chunks)} chunks, filtered to {len(filtered_chunks)} using threshold {adaptive_threshold:.3f} (best score: {best_score:.3f})")
                    # Log top 3 chunks for debugging
                    for i, (chunk_id, score, metadata) in enumerate(filtered_chunks[:3]):
                        content_preview = metadata.get('content', '')[:100].replace('\n', ' ')
                        logger.info(f"  Top chunk {i+1}: score={score:.3f}, preview='{content_preview}...'")
            
            if not filtered_chunks:
                # Provide helpful error message with suggestions
                raise HTTPException(
                    status_code=404,
                    detail="I'm unable to find relevant information to answer your question. Please try:\n"
                           "1. Rephrasing your question using different words or terminology\n"
                           "2. Being more specific about what aspect you're interested in\n"
                           "3. Breaking down your question into smaller, more focused questions\n"
                           "4. Checking if the documents you've selected contain the information you need"
                )
            
            retrieved_chunks = filtered_chunks
            
            # Generate response - use original query for display, but retrieval used normalized
            logger.info("Generating response...")
            generator = RAGGenerator()
            result = await generator.generate_response(
                query=request.query,  # Use original query for response generation (preserves user's wording)
                retrieved_chunks=retrieved_chunks,
                conversation_history=request.conversation_history
            )
            logger.info("Response generated")
            
            # Save query and response to database
            logger.info("Saving query and response to database...")
            import json
            async with pool.acquire() as conn:
                # Convert query embedding to pgvector format (use primary embedding)
                query_embedding_str = '[' + ','.join(map(str, primary_embedding)) + ']'
                
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
                    # Handle chunk_id - might be string or UUID
                    chunk_id = citation.get("chunk_id")
                    if isinstance(chunk_id, str):
                        chunk_id = UUID(chunk_id)
                    elif not isinstance(chunk_id, UUID):
                        chunk_id = UUID(str(chunk_id))
                    
                    await conn.execute("""
                        INSERT INTO query_citations (query_id, chunk_id, relevance_score, rank)
                        VALUES ($1, $2, $3, $4)
                    """, query_id, chunk_id, citation.get("relevance_score", 0.0), idx + 1)
            
            # Format citations for response
            citation_responses = []
            citations = result.get("citations", [])
            
            # Create a lookup map of chunk_id to score from retrieved_chunks
            chunk_score_map = {chunk[0]: chunk[1] for chunk in retrieved_chunks}
            
            for idx, c in enumerate(citations):
                try:
                    # Handle chunk_id - might be string or UUID
                    chunk_id = c.get("chunk_id")
                    if chunk_id is None:
                        continue  # Skip if no chunk_id
                    if isinstance(chunk_id, str):
                        chunk_id = UUID(chunk_id)
                    elif not isinstance(chunk_id, UUID):
                        chunk_id = UUID(str(chunk_id))
                    
                    # Get relevance score - prefer from citation, fallback to retrieved_chunks, or use default
                    relevance_score = c.get("relevance_score")
                    if relevance_score is None or relevance_score == 0:
                        # Try to get from retrieved_chunks
                        relevance_score = chunk_score_map.get(str(chunk_id)) or chunk_score_map.get(chunk_id)
                        if relevance_score is None or relevance_score == 0:
                            # If still 0, use a reasonable default for cited chunks
                            relevance_score = 0.5  # Default to medium relevance for cited chunks
                            logger.warning(f"Citation {idx+1} has no relevance score, using default 0.5")
                    
                    citation_responses.append(
                        CitationResponse(
                            chunk_id=chunk_id,
                            content=str(c.get("content", ""))[:500],  # Limit content length
                            relevance_score=float(relevance_score),
                            rank=idx + 1,
                            document=c.get("document"),
                            chapter=c.get("chapter"),
                            page_number=c.get("page_number"),
                            citation_string=c.get("citation_string"),  # Formatted: "Document, Chapter, Page X"
                            metadata=c.get("metadata", {}) or {}
                        )
                    )
                except Exception as e:
                    # Skip invalid citations but log the error
                    logger.warning(f"Skipping invalid citation at index {idx}: {e}")
                    continue
            
            logger.info("Calculating confidence scores...")
            # Calculate average relevance score from citations
            avg_relevance = 0.0
            if citation_responses:
                try:
                    relevance_scores = [float(c.relevance_score) for c in citation_responses if c.relevance_score is not None]
                    if relevance_scores:
                        avg_relevance = sum(relevance_scores) / len(relevance_scores)
                        logger.info(f"Average relevance: {avg_relevance:.3f}, Citations: {len(citation_responses)}, Scores: {relevance_scores}")
                    else:
                        logger.warning(f"No valid relevance scores in {len(citation_responses)} citations")
                        # If no valid scores, use a default based on having citations
                        avg_relevance = 0.5  # Default to medium relevance if citations exist but scores are missing
                except (ZeroDivisionError, TypeError) as e:
                    logger.error(f"Error calculating avg relevance: {e}")
                    avg_relevance = 0.5 if citation_responses else 0.0  # Default to 0.5 if citations exist
            else:
                logger.warning("No citations found for relevance calculation")
            
            # Calculate overall confidence score - RAG principle: answer is either there or not
            # If answer has citations, it's grounded in documents = HIGH confidence
            # If no citations, answer may not be grounded = LOW confidence
            groundedness = result.get("groundedness_score")
            if groundedness is None:
                groundedness = 1.0  # Default to high if not calculated
            logger.info(f"Groundedness: {groundedness:.3f}, Avg relevance: {avg_relevance:.3f}, Citations: {len(citation_responses)}")
            
            has_citations = len(citation_responses) > 0
            
            # RAG Principle: If answer is cited, it's in the documents = high confidence
            if has_citations:
                # Answer has citations = it's grounded in documents
                # This is the primary signal - trust it
                citation_count = len(citation_responses)
                
                # Base confidence on citation count and groundedness
                # More citations = more confidence
                if citation_count >= 3:
                    # 3+ citations = very well supported
                    overall_confidence = 0.95  # Start high
                elif citation_count >= 2:
                    # 2 citations = well supported
                    overall_confidence = 0.90  # Start high
                elif citation_count >= 1:
                    # 1 citation = supported
                    overall_confidence = 0.85  # Start high
                
                # Adjust slightly based on groundedness (but don't penalize too much)
                # If groundedness is very high, boost a bit
                if groundedness >= 0.8:
                    overall_confidence = min(overall_confidence + 0.05, 1.0)  # Small boost
                elif groundedness < 0.6:
                    # Lower groundedness - slight reduction but still high
                    overall_confidence = max(overall_confidence - 0.05, 0.80)  # Small reduction, keep high
                
                # Ensure minimum based on citations
                if citation_count >= 2:
                    overall_confidence = max(overall_confidence, 0.88)  # High minimum for 2+ citations
                else:
                    overall_confidence = max(overall_confidence, 0.85)  # High minimum for 1 citation
            else:
                # No citations = answer may not be grounded
                # Use groundedness as primary signal
                if groundedness >= 0.7:
                    overall_confidence = 0.75  # Medium-high if well-grounded but no explicit citations
                elif groundedness >= 0.5:
                    overall_confidence = 0.60  # Medium if somewhat grounded
                else:
                    overall_confidence = 0.40  # Low if not well-grounded
            
            # Cap at 1.0
            overall_confidence = min(overall_confidence, 1.0)
            
            logger.info(f"Query completed successfully. Final confidence: {overall_confidence:.2f} (groundedness: {groundedness:.3f}, avg_relevance: {avg_relevance:.3f}, citations: {len(citation_responses)})")
            
            # Create response object
            response = QueryResponse(
                query_id=query_id,
                response_text=result["response_text"],
                citations=citation_responses,
                model=generator.model,
                tokens_used=result["tokens_used"],
                latency_ms=result["latency_ms"],
                created_at=created_at,
                groundedness_score=groundedness,
                avg_relevance_score=avg_relevance,
                overall_confidence=overall_confidence
            )
            
            # Cache the response
            if settings.CACHE_ENABLED:
                try:
                    cache_data = {
                        "query_id": str(query_id),
                        "response_text": result["response_text"],
                        "citations": [c.dict() for c in citation_responses],
                        "model": generator.model,
                        "tokens_used": result["tokens_used"],
                        "latency_ms": result["latency_ms"],
                        "created_at": created_at.isoformat() if created_at else None,
                        "groundedness_score": float(groundedness) if groundedness is not None else 0.0,
                        "avg_relevance_score": float(avg_relevance) if avg_relevance is not None else 0.0,
                        "overall_confidence": float(overall_confidence)  # Ensure it's stored as float
                    }
                    logger.info(f"Storing in cache with confidence: {overall_confidence:.2f}")
                    await cache_service.set(
                        query=normalized_query,  # Use normalized query for cache key (typos share same cache)
                        response_data=cache_data,
                        ttl_seconds=settings.CACHE_TTL_SECONDS,
                        document_ids=[str(doc_id) for doc_id in request.document_ids] if request.document_ids else None,
                        core_area=request.core_area,
                        factory=request.factory
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            return response
        except HTTPException:
            raise
        except Exception as e:
            # Log the full error for debugging
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Query processing failed: {str(e)}\n{error_trace}")
            raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    # Wrap the entire query processing in a timeout (2 minutes)
    try:
        return await asyncio.wait_for(process_query(), timeout=120.0)
    except asyncio.TimeoutError:
        logger.error(f"Query timed out after 120 seconds: {request.query[:100]}...")
        raise HTTPException(
            status_code=504,
            detail="Query processing timed out. The request took too long to process. Please try again with a simpler query."
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Unexpected error in query endpoint: {str(e)}\n{error_trace}")
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
    
    # Return thank you message based on feedback type
    if request.feedback == 1:
        message = "Thank you for your positive feedback! We're glad this response was helpful."
    elif request.feedback == -1:
        message = "Thank you for your feedback. We'll use this to improve our responses."
    else:
        message = "Thank you for your feedback!"
    
    return {"message": message, "query_id": str(request.query_id)}


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


@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics
    """
    from app.core.cache import get_cache_service
    cache_service = get_cache_service()
    stats = cache_service.get_stats()
    return stats


@router.delete("/cache")
async def clear_cache(
    query: Optional[str] = Query(None),
    all: bool = Query(False, description="Clear all cache entries")
):
    """
    Clear cache entries
    If query is provided, clears that specific query's cache.
    If all=true, clears all cache entries (use with caution).
    """
    from app.core.cache import get_cache_service
    cache_service = get_cache_service()
    
    if all:
        await cache_service.invalidate()  # Clear all
        return {"message": "All cache entries cleared"}
    elif query:
        await cache_service.invalidate(query=query)
        return {"message": f"Cache cleared for query: {query}"}
    else:
        return {"message": "Please provide a query parameter or set all=true"}

