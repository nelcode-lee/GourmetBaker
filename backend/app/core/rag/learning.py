"""
Learning mechanisms for continuous improvement
"""
from typing import List, Tuple, Dict, Optional, Set
import asyncpg
from uuid import UUID
from collections import defaultdict, Counter
import re


class LearningService:
    """
    Service for learning from user interactions and improving retrieval
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        """
        Initialize learning service
        
        Args:
            db_pool: Database connection pool
        """
        self.db = db_pool
    
    async def expand_query(
        self,
        query_text: str,
        max_expansions: int = 3
    ) -> str:
        """
        Expand query with related terms learned from successful queries.
        Finds similar queries that received positive feedback and extracts
        additional relevant terms.
        
        Args:
            query_text: Original query text
            max_expansions: Maximum number of terms to add
            
        Returns:
            Expanded query text
        """
        # Extract keywords from current query
        query_words = set(re.findall(r'\b[a-z0-9]+\b', query_text.lower()))
        
        # Find similar successful queries (positive feedback)
        async with self.db.acquire() as conn:
            # Get queries with positive feedback that are similar
            rows = await conn.fetch("""
                SELECT q.query_text, COUNT(*) as positive_count
                FROM queries q
                JOIN responses r ON q.id = r.query_id
                WHERE r.feedback = 1
                  AND q.query_text != $1
                GROUP BY q.query_text
                ORDER BY positive_count DESC
                LIMIT 20
            """, query_text)
        
        # Extract terms from successful queries
        related_terms = Counter()
        for row in rows:
            successful_query = row["query_text"].lower()
            terms = set(re.findall(r'\b[a-z0-9]{3,}\b', successful_query))
            # Find terms that appear in successful queries but not in current query
            new_terms = terms - query_words
            for term in new_terms:
                related_terms[term] += row["positive_count"]
        
        # Add top related terms to query
        if related_terms:
            top_terms = [term for term, _ in related_terms.most_common(max_expansions)]
            expanded_query = query_text + " " + " ".join(top_terms)
            return expanded_query
        
        return query_text
    
    async def get_document_priority_boost(
        self,
        document_ids: List[UUID]
    ) -> Dict[str, float]:
        """
        Calculate priority boost for documents based on usage patterns.
        Frequently accessed documents with positive feedback get higher priority.
        
        Args:
            document_ids: List of document UUIDs
            
        Returns:
            Dictionary mapping document_id (str) to boost factor
        """
        if not document_ids:
            return {}
        
        async with self.db.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    d.id as document_id,
                    COUNT(DISTINCT qc.query_id) as query_count,
                    COUNT(CASE WHEN r.feedback = 1 THEN 1 END)::FLOAT as positive_feedback,
                    COUNT(CASE WHEN r.feedback = -1 THEN 1 END)::FLOAT as negative_feedback,
                    AVG(qc.relevance_score)::FLOAT as avg_relevance
                FROM documents d
                LEFT JOIN document_chunks dc ON d.id = dc.document_id
                LEFT JOIN query_citations qc ON dc.id = qc.chunk_id
                LEFT JOIN responses r ON qc.query_id = r.query_id
                WHERE d.id = ANY($1::uuid[])
                  AND d.status = 'ready'
                GROUP BY d.id
            """, document_ids)
        
        boosts = {}
        for row in rows:
            doc_id = str(row["document_id"])
            query_count = row["query_count"] or 0
            positive = row["positive_feedback"] or 0
            negative = row["negative_feedback"] or 0
            avg_relevance = row["avg_relevance"] or 0.5
            
            # Calculate boost based on:
            # - Query frequency (more queries = more important)
            # - Positive feedback ratio
            # - Average relevance score
            if query_count > 0:
                feedback_ratio = positive / max(query_count, 1)
                # Boost: 0-15% based on usage and feedback
                boost = min(
                    (query_count * 0.01) +  # 1% per query (capped)
                    (feedback_ratio * 0.05) +  # 5% for perfect feedback
                    (avg_relevance * 0.05),  # 5% for high relevance
                    0.15  # Cap at 15%
                )
                boosts[doc_id] = boost
        
        return boosts
    
    async def get_adaptive_weights(
        self
    ) -> Tuple[float, float]:
        """
        Learn optimal vector/keyword weights based on feedback patterns.
        Analyzes which retrieval method (vector vs keyword) performs better
        for queries with positive feedback.
        
        Returns:
            Tuple of (vector_weight, keyword_weight)
        """
        async with self.db.acquire() as conn:
            # Analyze successful queries to see which method worked better
            rows = await conn.fetch("""
                SELECT 
                    q.query_text,
                    AVG(CASE WHEN qc.relevance_score > 0.7 THEN 1.0 ELSE 0.0 END) as high_relevance_ratio,
                    COUNT(*) as citation_count
                FROM queries q
                JOIN responses r ON q.id = r.query_id
                JOIN query_citations qc ON q.id = qc.query_id
                WHERE r.feedback = 1
                  AND qc.relevance_score IS NOT NULL
                GROUP BY q.id, q.query_text
                HAVING COUNT(*) >= 3
                LIMIT 50
            """)
        
        if not rows:
            # Default weights if no data
            return (0.7, 0.3)
        
        # Simple heuristic: if queries with high relevance scores are common,
        # vector search is working well. If not, keyword search might be better.
        high_relevance_count = sum(1 for row in rows if (row["high_relevance_ratio"] or 0) > 0.5)
        total = len(rows)
        
        if total > 0:
            vector_ratio = high_relevance_count / total
            # Adjust weights: more vector weight if high relevance is common
            vector_weight = 0.6 + (vector_ratio * 0.2)  # 0.6 to 0.8
            keyword_weight = 1.0 - vector_weight
            return (vector_weight, keyword_weight)
        
        return (0.7, 0.3)
    
    async def get_query_suggestions(
        self,
        query_text: str,
        limit: int = 5
    ) -> List[Dict[str, any]]:
        """
        Suggest similar successful queries based on current query.
        
        Args:
            query_text: Current query text
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested queries with metadata
        """
        # Extract keywords from query
        query_words = set(re.findall(r'\b[a-z0-9]{3,}\b', query_text.lower()))
        
        async with self.db.acquire() as conn:
            # Find queries with positive feedback that share keywords
            rows = await conn.fetch("""
                SELECT 
                    q.query_text,
                    COUNT(DISTINCT r.id) as positive_count,
                    AVG(r.latency_ms)::FLOAT as avg_latency,
                    MAX(q.created_at) as last_used
                FROM queries q
                JOIN responses r ON q.id = r.query_id
                WHERE r.feedback = 1
                  AND q.query_text != $1
                  AND q.query_text ILIKE ANY(ARRAY['%' || unnest(string_to_array($1, ' ')) || '%'])
                GROUP BY q.query_text
                ORDER BY positive_count DESC, last_used DESC
                LIMIT $2
            """, query_text, limit * 2)
        
        suggestions = []
        for row in rows:
            suggested_query = row["query_text"]
            # Calculate similarity (shared keywords)
            suggested_words = set(re.findall(r'\b[a-z0-9]{3,}\b', suggested_query.lower()))
            shared_words = query_words & suggested_words
            similarity = len(shared_words) / max(len(query_words), 1)
            
            if similarity > 0.2:  # At least 20% keyword overlap
                suggestions.append({
                    "query": suggested_query,
                    "positive_feedback_count": row["positive_count"] or 0,
                    "avg_latency_ms": row["avg_latency"] or 0,
                    "similarity": similarity
                })
        
        # Sort by similarity and feedback, return top results
        suggestions.sort(key=lambda x: (x["similarity"], x["positive_feedback_count"]), reverse=True)
        return suggestions[:limit]
    
    async def learn_from_interaction(
        self,
        query_id: UUID,
        feedback: int
    ) -> None:
        """
        Learn from a single interaction and update internal models.
        This can be called after feedback is submitted.
        
        Args:
            query_id: Query ID
            feedback: Feedback value (1 for positive, -1 for negative)
        """
        # This is a placeholder for future learning mechanisms
        # Could include:
        # - Updating term associations
        # - Adjusting document priorities
        # - Refining query expansion rules
        pass

