"""
Hybrid RAG retriever with vector similarity and keyword search
"""
from typing import List, Tuple, Dict, Optional
import asyncpg
import numpy as np
from uuid import UUID
from app.core.rag.learning import LearningService


class HybridRetriever:
    """
    Hybrid retrieval combining:
    - Vector similarity search (pgvector)
    - Keyword search (full-text search)
    - Score combination and reranking
    - Learning from user feedback (relevance feedback)
    """
    
    def __init__(self, db_pool: asyncpg.Pool, use_feedback: bool = True, use_learning: bool = True):
        """
        Initialize retriever
        
        Args:
            db_pool: Database connection pool
            use_feedback: Whether to boost chunks based on past positive feedback
            use_learning: Whether to use advanced learning features (query expansion, document prioritization, adaptive weights)
        """
        self.db = db_pool
        self.vector_weight = 0.7
        self.keyword_weight = 0.3
        self.use_feedback = use_feedback
        self.use_learning = use_learning
        if use_learning:
            self.learning_service = LearningService(db_pool)
    
    async def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve relevant chunks using hybrid search with learning
        
        Process:
        1. Query expansion (learn from successful queries)
        2. Adaptive weight adjustment (learn optimal vector/keyword balance)
        3. Vector search (cosine similarity)
        4. Keyword search (full-text)
        5. Combine scores with learned weights
        6. Document prioritization (boost frequently accessed documents)
        7. Feedback-based boosting
        8. Return top_k with metadata
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text
            top_k: Number of results to return
            document_ids: Optional filter by document IDs
            
        Returns:
            List of (chunk_id, score, metadata) tuples
        """
        # Step 1: Query expansion (if learning enabled)
        expanded_query = query_text
        if self.use_learning and self.learning_service:
            try:
                expanded_query = await self.learning_service.expand_query(query_text)
            except Exception as e:
                import logging
                logging.warning(f"Query expansion failed, using original query: {e}")
                expanded_query = query_text
        
        # Step 2: Get adaptive weights (if learning enabled)
        if self.use_learning and self.learning_service:
            try:
                self.vector_weight, self.keyword_weight = await self.learning_service.get_adaptive_weights()
            except Exception as e:
                import logging
                logging.warning(f"Adaptive weights failed, using defaults: {e}")
                # Keep default weights
        
        # Get results from both searches
        vector_results = await self._vector_search(
            query_embedding, top_k * 2, document_ids, core_area, factory
        )
        keyword_results = await self._keyword_search(
            expanded_query, top_k * 2, document_ids, core_area, factory
        )
        
        # Combine scores
        combined_scores = {}
        
        # Add vector scores
        for chunk_id, score in vector_results:
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"vector": 0.0, "keyword": 0.0}
            combined_scores[chunk_id]["vector"] = score
        
        # Add keyword scores
        for chunk_id, score in keyword_results:
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {"vector": 0.0, "keyword": 0.0}
            combined_scores[chunk_id]["keyword"] = score
        
        # Normalize scores and combine
        final_results = []
        
        # Normalize vector scores if we have any
        if vector_results:
            max_vector = max(score for _, score in vector_results) or 1.0
            min_vector = min(score for _, score in vector_results) or 0.0
            vector_range = max_vector - min_vector if max_vector > min_vector else 1.0
        else:
            vector_range = 1.0
        
        # Normalize keyword scores if we have any
        if keyword_results:
            max_keyword = max(score for _, score in keyword_results) or 1.0
            min_keyword = min(score for _, score in keyword_results) or 0.0
            keyword_range = max_keyword - min_keyword if max_keyword > min_keyword else 1.0
        else:
            keyword_range = 1.0
        
        for chunk_id, scores in combined_scores.items():
            # Normalize scores to 0-1 range
            vector_score = scores["vector"]
            keyword_score = scores["keyword"]
            
            # Normalize vector score
            if vector_range > 0:
                vector_score = (vector_score - (min(score for _, score in vector_results) if vector_results else 0)) / vector_range
            vector_score = min(max(vector_score, 0), 1)
            
            # Normalize keyword score
            if keyword_range > 0:
                keyword_score = (keyword_score - (min(score for _, score in keyword_results) if keyword_results else 0)) / keyword_range
            keyword_score = min(max(keyword_score, 0), 1)
            
            # Combine with weights - boost keyword matches for specific queries
            # If keyword score is high, it's likely a good match
            if keyword_score > 0.5:
                # Boost keyword-heavy matches
                combined_score = (
                    self.vector_weight * vector_score * 0.7 +  # Slightly reduce vector weight
                    self.keyword_weight * keyword_score * 1.3  # Boost keyword weight
                )
            else:
                combined_score = (
                    self.vector_weight * vector_score +
                    self.keyword_weight * keyword_score
                )
            
            # Ensure minimum score if either search found something
            if vector_score > 0 or keyword_score > 0:
                combined_score = max(combined_score, 0.15)  # Minimum 0.15 if any match
            
            final_results.append((chunk_id, combined_score))
        
        # Apply document prioritization (if learning enabled)
        if self.use_learning and self.learning_service and document_ids:
            try:
                final_results = await self._apply_document_prioritization(final_results, document_ids)
            except Exception as e:
                import logging
                logging.warning(f"Document prioritization failed: {e}")
                # Continue without prioritization
        
        # Apply feedback-based boosting if enabled
        if self.use_feedback:
            try:
                final_results = await self._apply_feedback_boosting(final_results)
            except Exception as e:
                import logging
                logging.warning(f"Feedback boosting failed: {e}")
                # Continue without boosting
        
        # Sort by combined score and get top_k
        final_results.sort(key=lambda x: x[1], reverse=True)
        top_results = final_results[:top_k]
        
        # Fetch full chunk data with metadata
        results_with_metadata = []
        async with self.db.acquire() as conn:
            for chunk_id, score in top_results:
                row = await conn.fetchrow("""
                    SELECT dc.id, dc.content, dc.metadata, dc.chunk_index,
                           d.id as document_id, d.filename
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE dc.id = $1 AND d.status = 'ready'
                """, UUID(chunk_id))
                
                if row:
                    # Parse metadata if it's a JSON string
                    import json
                    row_metadata = row["metadata"]
                    if isinstance(row_metadata, str):
                        try:
                            row_metadata = json.loads(row_metadata)
                        except:
                            row_metadata = {}
                    elif row_metadata is None:
                        row_metadata = {}
                    
                    metadata = {
                        "content": row["content"],
                        "chunk_index": row["chunk_index"],
                        "document_id": str(row["document_id"]),
                        "filename": row["filename"],
                        **row_metadata
                    }
                    results_with_metadata.append((chunk_id, score, metadata))
        
        return results_with_metadata
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        document_ids: Optional[List[str]] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Vector similarity search using pgvector cosine similarity
        """
        # Convert embedding list to pgvector format
        query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        async with self.db.acquire() as conn:
            # Build WHERE conditions
            conditions = ["d.status = 'ready'"]
            params = [query_embedding_str]
            param_idx = 2
            
            if document_ids:
                doc_ids = [UUID(doc_id) for doc_id in document_ids]
                conditions.append(f"d.id = ANY(${param_idx}::uuid[])")
                params.append(doc_ids)
                param_idx += 1
            
            if core_area:
                conditions.append(f"d.metadata->>'core_area' = ${param_idx}")
                params.append(core_area)
                param_idx += 1
            
            if factory:
                conditions.append(f"d.metadata->>'factory' = ${param_idx}")
                params.append(factory)
                param_idx += 1
            
            where_clause = " AND ".join(conditions)
            params.append(top_k)
            
            query = f"""
                SELECT dc.id, 1 - (dc.embedding <=> $1::vector) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE {where_clause}
                ORDER BY dc.embedding <=> $1::vector
                LIMIT ${param_idx}
            """
            rows = await conn.fetch(query, *params)
            
            return [(str(row["id"]), float(row["similarity"])) for row in rows]
    
    async def _keyword_search(
        self,
        query_text: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
        core_area: Optional[str] = None,
        factory: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Keyword search using PostgreSQL full-text search
        Enhanced to handle multi-word phrases and improve scoring
        """
        import re
        
        # Extract keywords from query - keep important phrases together
        # Remove common stop words but keep important terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'are', 'were', 'was'}
        words = re.findall(r'\b[a-z0-9]+\b', query_text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also try to match common phrases (2-3 word combinations)
        phrases = []
        if len(words) >= 2:
            # Extract 2-word phrases
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if all(w not in stop_words for w in words[i:i+2]):
                    phrases.append(phrase)
        
        if not keywords and not phrases:
            return []
        
        async with self.db.acquire() as conn:
            # Build search conditions safely using parameters
            conditions = []
            params = []
            param_idx = 1
            
            # Base conditions
            base_conditions = ["d.status = 'ready'"]
            
            if document_ids:
                doc_ids = [UUID(doc_id) for doc_id in document_ids]
                base_conditions.append(f"d.id = ANY(${param_idx}::uuid[])")
                params.append(doc_ids)
                param_idx += 1
            
            if core_area:
                base_conditions.append(f"d.metadata->>'core_area' = ${param_idx}")
                params.append(core_area)
                param_idx += 1
            
            if factory:
                base_conditions.append(f"d.metadata->>'factory' = ${param_idx}")
                params.append(factory)
                param_idx += 1
            
            # Keyword search conditions - phrases get higher weight
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(f"dc.content ILIKE ${param_idx}")
                params.append(f"%{keyword}%")
                param_idx += 1
            
            # Phrase conditions (exact phrase matching gets higher score)
            phrase_conditions = []
            for phrase in phrases[:3]:  # Limit to top 3 phrases
                phrase_conditions.append(f"dc.content ILIKE ${param_idx}")
                params.append(f"%{phrase}%")
                param_idx += 1
            
            where_clause = " AND ".join(base_conditions)
            all_conditions = []
            if keyword_conditions:
                all_conditions.append("(" + " OR ".join(keyword_conditions) + ")")
            if phrase_conditions:
                all_conditions.append("(" + " OR ".join(phrase_conditions) + ")")
            
            if all_conditions:
                where_clause += " AND (" + " OR ".join(all_conditions) + ")"
            
            params.append(top_k)
            
            # Build scoring query - count matches with phrase weighting
            if phrases:
                # Build phrase match conditions
                phrase_match_expr = " + ".join([
                    f"CASE WHEN dc.content ILIKE ${param_idx - len(keywords) - len(phrases) + i} THEN 2.0 ELSE 0.0 END"
                    for i in range(len(phrases))
                ])
                keyword_match_expr = " + ".join([
                    f"CASE WHEN dc.content ILIKE ${param_idx - len(keywords) - len(phrases) + len(phrases) + i} THEN 1.0 ELSE 0.0 END"
                    for i in range(len(keywords))
                ])
                score_expr = f"({phrase_match_expr} + {keyword_match_expr})"
            else:
                # Just count keyword matches
                score_expr = " + ".join([
                    f"CASE WHEN dc.content ILIKE ${param_idx - len(keywords) + i} THEN 1.0 ELSE 0.0 END"
                    for i in range(len(keywords))
                ])
            
            query = f"""
                SELECT dc.id,
                       ({score_expr}) as score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE {where_clause}
                GROUP BY dc.id
                HAVING ({score_expr}) > 0
                ORDER BY score DESC
                LIMIT ${param_idx}
            """
            
            rows = await conn.fetch(query, *params)
            
            # Normalize scores (simple approach)
            results = []
            if rows:
                max_score = max(float(row["score"]) for row in rows) or 1.0
                for row in rows:
                    # Normalize and boost scores (ensure minimum 0.2 for any match)
                    normalized_score = min(float(row["score"]) / max_score, 1.0)
                    # Boost: if score > 0, ensure it's at least 0.2
                    if normalized_score > 0:
                        normalized_score = max(normalized_score, 0.2)
                    results.append((str(row["id"]), normalized_score))
            
            return results
    
    async def _apply_feedback_boosting(
        self,
        results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Boost chunks that have received positive feedback in the past.
        This implements relevance feedback learning.
        
        Args:
            results: List of (chunk_id, score) tuples
            
        Returns:
            List of (chunk_id, boosted_score) tuples
        """
        if not results:
            return results
        
        async with self.db.acquire() as conn:
            # Get feedback scores for chunks
            chunk_ids = [UUID(chunk_id) for chunk_id, _ in results]
            
            # Calculate feedback boost for each chunk
            # Positive feedback increases score, negative feedback decreases it
            feedback_rows = await conn.fetch("""
                SELECT 
                    qc.chunk_id,
                    COUNT(CASE WHEN r.feedback = 1 THEN 1 END)::FLOAT as positive_count,
                    COUNT(CASE WHEN r.feedback = -1 THEN 1 END)::FLOAT as negative_count,
                    COUNT(*)::FLOAT as total_feedback
                FROM query_citations qc
                JOIN responses r ON qc.query_id = r.query_id
                WHERE qc.chunk_id = ANY($1::uuid[])
                  AND r.feedback IS NOT NULL
                GROUP BY qc.chunk_id
            """, chunk_ids)
            
            # Create feedback boost map
            feedback_boost = {}
            for row in feedback_rows:
                chunk_id = str(row["chunk_id"])
                positive = row["positive_count"] or 0
                negative = row["negative_count"] or 0
                total = row["total_feedback"] or 1
                
                # Calculate boost: positive feedback increases, negative decreases
                # Boost factor: 0.1 (10%) per positive feedback, -0.05 (5%) per negative
                # Normalized by total feedback to avoid over-boosting
                if total > 0:
                    boost_factor = (positive * 0.1 - negative * 0.05) / max(total, 1)
                    feedback_boost[chunk_id] = boost_factor
            
            # Apply boosting to results
            boosted_results = []
            for chunk_id, score in results:
                boost = feedback_boost.get(chunk_id, 0.0)
                # Apply boost (capped at 20% increase)
                boosted_score = score * (1.0 + min(boost, 0.2))
                boosted_results.append((chunk_id, boosted_score))
            
            return boosted_results
    
    async def _apply_document_prioritization(
        self,
        results: List[Tuple[str, float]],
        document_ids: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Boost chunks from frequently accessed documents with positive feedback.
        
        Args:
            results: List of (chunk_id, score) tuples
            document_ids: List of document IDs to check
            
        Returns:
            List of (chunk_id, boosted_score) tuples
        """
        if not results or not document_ids:
            return results
        
        # Get document priority boosts
        doc_uuids = [UUID(doc_id) for doc_id in document_ids]
        doc_boosts = await self.learning_service.get_document_priority_boost(doc_uuids)
        
        if not doc_boosts:
            return results
        
        # Get document_id for each chunk
        async with self.db.acquire() as conn:
            chunk_ids = [UUID(chunk_id) for chunk_id, _ in results]
            chunk_docs = await conn.fetch("""
                SELECT dc.id, dc.document_id
                FROM document_chunks dc
                WHERE dc.id = ANY($1::uuid[])
            """, chunk_ids)
        
        # Create chunk to document mapping
        chunk_to_doc = {str(row["id"]): str(row["document_id"]) for row in chunk_docs}
        
        # Apply document boosts
        boosted_results = []
        for chunk_id, score in results:
            doc_id = chunk_to_doc.get(chunk_id)
            boost = doc_boosts.get(doc_id, 0.0) if doc_id else 0.0
            boosted_score = score * (1.0 + boost)
            boosted_results.append((chunk_id, boosted_score))
        
        return boosted_results

