"""
Analytics API routes
"""
from typing import List
from fastapi import APIRouter, Depends
import asyncpg
from app.core.db.database import get_db_pool
from app.api.models.schemas import (
    AnalyticsOverview,
    QueryPattern,
    DocumentUsageStats,
    ChunkTagStats,
    KeyTermStats,
    DocumentQueryStats,
    QualityMetrics
)

router = APIRouter()


@router.get("/overview", response_model=AnalyticsOverview)
async def get_analytics_overview(
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get dashboard metrics overview
    """
    async with pool.acquire() as conn:
        # Total documents
        total_documents = await conn.fetchval("SELECT COUNT(*) FROM documents")
        
        # Total queries
        total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries")
        
        # Total chunks
        total_chunks = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
        
        # Average response time
        avg_response_time = await conn.fetchval("""
            SELECT AVG(latency_ms)::FLOAT 
            FROM responses 
            WHERE latency_ms IS NOT NULL
        """) or 0.0
        
        # Positive feedback ratio
        total_feedback = await conn.fetchval("""
            SELECT COUNT(*) FROM responses WHERE feedback IS NOT NULL
        """) or 0
        
        positive_feedback = await conn.fetchval("""
            SELECT COUNT(*) FROM responses WHERE feedback = 1
        """) or 0
        
        positive_feedback_ratio = (
            positive_feedback / total_feedback if total_feedback > 0 else 0.0
        )
        
        # Documents by status
        status_rows = await conn.fetch("""
            SELECT status, COUNT(*) as count
            FROM documents
            GROUP BY status
        """)
        documents_by_status = {row["status"]: row["count"] for row in status_rows}
    
    return AnalyticsOverview(
        total_documents=total_documents or 0,
        total_queries=total_queries or 0,
        total_chunks=total_chunks or 0,
        avg_response_time_ms=avg_response_time,
        positive_feedback_ratio=positive_feedback_ratio,
        documents_by_status=documents_by_status
    )


@router.get("/queries", response_model=List[QueryPattern])
async def get_query_patterns(
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get query patterns and frequency
    """
    async with pool.acquire() as conn:
        # Get query frequency (exact matches)
        rows = await conn.fetch("""
            SELECT query_text, COUNT(*) as frequency,
                   AVG(r.feedback)::FLOAT as avg_feedback
            FROM queries q
            LEFT JOIN responses r ON q.id = r.query_id
            GROUP BY query_text
            ORDER BY frequency DESC
            LIMIT 20
        """)
        
        patterns = []
        for row in rows:
            patterns.append(QueryPattern(
                query_text=row["query_text"],
                frequency=row["frequency"],
                avg_feedback=row["avg_feedback"]
            ))
    
    return patterns


@router.get("/queries/by-documents", response_model=List[DocumentQueryStats])
async def get_queries_by_documents(
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get query statistics grouped by document
    Shows which documents are queried most frequently
    """
    import json
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                d.id as document_id,
                d.filename,
                d.metadata,
                COUNT(DISTINCT qc.query_id) as query_count,
                AVG(qc.relevance_score)::FLOAT as avg_relevance_score,
                COUNT(CASE WHEN r.feedback = 1 THEN 1 END)::INTEGER as positive_feedback_count,
                COUNT(CASE WHEN r.feedback = -1 THEN 1 END)::INTEGER as negative_feedback_count,
                MAX(q.created_at) as last_queried
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc.document_id
            LEFT JOIN query_citations qc ON dc.id = qc.chunk_id
            LEFT JOIN queries q ON qc.query_id = q.id
            LEFT JOIN responses r ON q.id = r.query_id
            WHERE d.status = 'ready'
            GROUP BY d.id, d.filename, d.metadata
            HAVING COUNT(DISTINCT qc.query_id) > 0
            ORDER BY query_count DESC, last_queried DESC
            LIMIT 50
        """)
        
        doc_stats = []
        for row in rows:
            # Parse document metadata
            metadata = row["metadata"]
            core_area = None
            factory = None
            
            if metadata:
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                if isinstance(metadata, dict):
                    core_area = metadata.get("core_area")
                    factory = metadata.get("factory")
            
            doc_stats.append(DocumentQueryStats(
                document_id=row["document_id"],
                filename=row["filename"],
                query_count=row["query_count"] or 0,
                avg_relevance_score=row["avg_relevance_score"] or 0.0,
                positive_feedback_count=row["positive_feedback_count"] or 0,
                negative_feedback_count=row["negative_feedback_count"] or 0,
                last_queried=row["last_queried"],
                core_area=core_area,
                factory=factory
            ))
    
    return doc_stats


@router.get("/queries/by-key-terms", response_model=List[KeyTermStats])
async def get_queries_by_key_terms(
    pool: asyncpg.Pool = Depends(get_db_pool),
    min_length: int = 3,
    limit: int = 30
):
    """
    Get query statistics grouped by key terms/keywords
    Extracts meaningful terms from queries and shows frequency
    """
    import re
    from collections import Counter, defaultdict
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
        'where', 'when', 'why', 'how', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
        'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'all', 'each', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
        'don', 'should', 'now'
    }
    
    async with pool.acquire() as conn:
        # Get all queries with their feedback
        rows = await conn.fetch("""
            SELECT q.query_text, r.feedback
            FROM queries q
            LEFT JOIN responses r ON q.id = r.query_id
            ORDER BY q.created_at DESC
        """)
    
    # Extract terms from all queries
    term_counter = Counter()
    term_queries = defaultdict(set)  # Track which queries contain each term
    term_feedback = defaultdict(list)  # Track feedback for queries with each term
    
    for row in rows:
        query_text = row["query_text"].lower()
        feedback = row["feedback"]
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-z0-9]+\b', query_text)
        
        for word in words:
            # Filter out stop words and short words
            if len(word) >= min_length and word not in stop_words:
                term_counter[word] += 1
                term_queries[word].add(query_text)  # Track unique queries
                if feedback is not None:
                    term_feedback[word].append(feedback)
    
    # Build results
    key_terms = []
    for term, frequency in term_counter.most_common(limit):
        unique_queries_count = len(term_queries[term])
        feedbacks = term_feedback[term]
        avg_feedback = sum(feedbacks) / len(feedbacks) if feedbacks else None
        
        key_terms.append(KeyTermStats(
            term=term,
            frequency=frequency,
            avg_feedback=avg_feedback,
            unique_queries=unique_queries_count
        ))
    
    return key_terms


@router.get("/documents", response_model=List[DocumentUsageStats])
async def get_document_usage_stats(
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get document usage statistics
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                d.id as document_id,
                d.filename,
                COUNT(DISTINCT qc.query_id) as query_count,
                AVG(qc.relevance_score)::FLOAT as avg_relevance_score,
                MAX(q.created_at) as last_queried
            FROM documents d
            LEFT JOIN document_chunks dc ON d.id = dc.document_id
            LEFT JOIN query_citations qc ON dc.id = qc.chunk_id
            LEFT JOIN queries q ON qc.query_id = q.id
            WHERE d.status = 'ready'
            GROUP BY d.id, d.filename
            ORDER BY query_count DESC, last_queried DESC
            LIMIT 50
        """)
        
        stats = []
        for row in rows:
            stats.append(DocumentUsageStats(
                document_id=row["document_id"],
                filename=row["filename"],
                query_count=row["query_count"] or 0,
                avg_relevance_score=row["avg_relevance_score"] or 0.0,
                last_queried=row["last_queried"]
            ))
    
    return stats


@router.get("/quality-metrics", response_model=QualityMetrics)
async def get_quality_metrics(
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get quality metrics for donut charts
    Includes feedback distribution, confidence scores, relevance scores, etc.
    """
    async with pool.acquire() as conn:
        # Feedback distribution
        feedback_rows = await conn.fetch("""
            SELECT 
                CASE 
                    WHEN feedback = 1 THEN 'positive'
                    WHEN feedback = -1 THEN 'negative'
                    ELSE 'neutral'
                END as feedback_type,
                COUNT(*) as count
            FROM responses
            WHERE feedback IS NOT NULL
            GROUP BY feedback_type
        """)
        feedback_distribution = {row["feedback_type"]: row["count"] for row in feedback_rows}
        # Ensure all keys exist
        for key in ['positive', 'negative', 'neutral']:
            if key not in feedback_distribution:
                feedback_distribution[key] = 0
        
        # Confidence score distribution - use a more efficient query
        # Calculate from query_citations grouped by query
        confidence_rows = await conn.fetch("""
            WITH query_avg_scores AS (
                SELECT 
                    qc.query_id,
                    AVG(qc.relevance_score) as avg_score
                FROM query_citations qc
                GROUP BY qc.query_id
            )
            SELECT 
                CASE 
                    WHEN avg_score >= 0.7 THEN 'high'
                    WHEN avg_score >= 0.4 THEN 'medium'
                    ELSE 'low'
                END as confidence_level,
                COUNT(*) as count
            FROM query_avg_scores
            GROUP BY confidence_level
        """)
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}
        for row in confidence_rows:
            level = row["confidence_level"]
            if level in confidence_distribution:
                confidence_distribution[level] += row["count"]
        
        # Relevance score distribution - optimized query
        relevance_rows = await conn.fetch("""
            SELECT 
                CASE 
                    WHEN relevance_score >= 0.7 THEN 'high'
                    WHEN relevance_score >= 0.4 THEN 'medium'
                    ELSE 'low'
                END as relevance_level,
                COUNT(*) as count
            FROM query_citations
            WHERE relevance_score IS NOT NULL
            GROUP BY 
                CASE 
                    WHEN relevance_score >= 0.7 THEN 'high'
                    WHEN relevance_score >= 0.4 THEN 'medium'
                    ELSE 'low'
                END
        """)
        relevance_distribution = {'high': 0, 'medium': 0, 'low': 0}
        for row in relevance_rows:
            level = row["relevance_level"]
            if level in relevance_distribution:
                relevance_distribution[level] += row["count"]
        
        # Document status distribution
        status_rows = await conn.fetch("""
            SELECT status, COUNT(*) as count
            FROM documents
            GROUP BY status
        """)
        document_status_distribution = {row["status"]: row["count"] for row in status_rows}
        
        # Average scores - use more efficient query
        avg_confidence = await conn.fetchval("""
            WITH query_avg_scores AS (
                SELECT AVG(qc.relevance_score) as avg_score
                FROM query_citations qc
                GROUP BY qc.query_id
            )
            SELECT AVG(
                CASE 
                    WHEN avg_score >= 0.7 THEN 0.85
                    WHEN avg_score >= 0.4 THEN 0.55
                    ELSE 0.25
                END
            )::FLOAT
            FROM query_avg_scores
        """) or 0.0
        
        avg_relevance = await conn.fetchval("""
            SELECT AVG(relevance_score)::FLOAT
            FROM query_citations
        """) or 0.0
        
        # For groundedness, we'd need to store it in responses table
        # For now, use a calculated value based on citation coverage
        avg_groundedness = await conn.fetchval("""
            SELECT AVG(
                CASE 
                    WHEN citation_count >= 3 THEN 0.9
                    WHEN citation_count >= 2 THEN 0.7
                    WHEN citation_count >= 1 THEN 0.5
                    ELSE 0.3
                END
            )::FLOAT
            FROM (
                SELECT qc.query_id, COUNT(*) as citation_count
                FROM query_citations qc
                GROUP BY qc.query_id
            ) subq
        """) or 0.0
        
        total_with_feedback = await conn.fetchval("""
            SELECT COUNT(*) FROM responses WHERE feedback IS NOT NULL
        """) or 0
        
        total_with_confidence = await conn.fetchval("""
            SELECT COUNT(DISTINCT query_id) FROM query_citations
        """) or 0
        
        # Calculate Training Quality Score (0-100)
        # Composite metric combining multiple indicators of agent performance
        # Components:
        # 1. Positive feedback ratio (40% weight) - how often users are satisfied
        # 2. Average confidence score (30% weight) - how confident the agent is
        # 3. Average relevance score (20% weight) - how relevant retrieved chunks are
        # 4. Query success rate (10% weight) - queries with positive feedback / total queries
        
        # 1. Positive feedback ratio
        total_queries = await conn.fetchval("SELECT COUNT(*) FROM queries") or 1
        positive_feedback_count = feedback_distribution.get('positive', 0)
        negative_feedback_count = feedback_distribution.get('negative', 0)
        total_feedback_count = positive_feedback_count + negative_feedback_count
        positive_feedback_ratio = (positive_feedback_count / total_feedback_count * 100) if total_feedback_count > 0 else 0.0
        
        # 2. Average confidence score (normalize to 0-100)
        confidence_score_normalized = avg_confidence * 100 if avg_confidence > 0 else 0.0
        
        # 3. Average relevance score (normalize to 0-100)
        relevance_score_normalized = avg_relevance * 100 if avg_relevance > 0 else 0.0
        
        # 4. Query success rate (queries with positive feedback / total queries)
        query_success_rate = (positive_feedback_count / total_queries * 100) if total_queries > 0 else 0.0
        
        # Calculate weighted composite score
        training_quality_score = (
            positive_feedback_ratio * 0.40 +      # 40% weight
            confidence_score_normalized * 0.30 +    # 30% weight
            relevance_score_normalized * 0.20 +     # 20% weight
            query_success_rate * 0.10              # 10% weight
        )
        
        # Cap at 100 and ensure minimum of 0
        training_quality_score = max(0.0, min(100.0, training_quality_score))
    
    return QualityMetrics(
        feedback_distribution=feedback_distribution,
        confidence_distribution=confidence_distribution,
        relevance_distribution=relevance_distribution,
        document_status_distribution=document_status_distribution,
        avg_confidence_score=avg_confidence,
        avg_relevance_score=avg_relevance,
        avg_groundedness_score=avg_groundedness,
        total_with_feedback=total_with_feedback,
        total_with_confidence=total_with_confidence,
        training_quality_score=training_quality_score
    )

