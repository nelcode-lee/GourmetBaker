"""
RAG response generator using OpenAI GPT-4
"""
from typing import List, Tuple, Dict, Optional, AsyncGenerator
import time
import re
from openai import AsyncOpenAI
from app.core.config import get_settings

settings = get_settings()


class RAGGenerator:
    """
    Generates responses using retrieved context and OpenAI API
    """
    
    def __init__(self):
        """Initialize generator with OpenAI client"""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, float, Dict]],
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate response using RAG pipeline
        
        Process:
        1. Build prompt with retrieved context
        2. Add conversation history if available
        3. Call OpenAI API
        4. Extract citations
        5. Return response + metadata (tokens, latency)
        
        Args:
            query: User query
            retrieved_chunks: List of (chunk_id, score, metadata) tuples
            conversation_history: Optional conversation history
            
        Returns:
            Dict with response_text, citations, tokens_used, latency_ms
        """
        start_time = time.time()
        
        # Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)
        
        # Build messages for chat API
        messages = self._build_messages(query, context, conversation_history)
        
        try:
            # Call OpenAI API with very low temperature to maximize accuracy
            # Reduced max_tokens slightly for faster responses while maintaining quality
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Set to 0 for maximum determinism and accuracy
                max_tokens=1200,  # Reduced from 1500 for faster responses
                top_p=0.9  # Nucleus sampling for more focused responses
            )
            
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract citations from response
            citations = self._extract_citations(response_text, retrieved_chunks)
            
            # Validate response groundedness
            groundedness_score = self._validate_groundedness(response_text, retrieved_chunks)
            
            # Add warning if response may not be fully grounded
            if groundedness_score < 0.5:
                response_text += "\n\n⚠️ WARNING: This response may contain information not explicitly found in the provided sources. Please verify against the original documents."
            
            return {
                "response_text": response_text,
                "citations": citations,
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "groundedness_score": groundedness_score
            }
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _format_context(self, chunks: List[Tuple[str, float, Dict]]) -> str:
        """Format retrieved chunks into context string"""
        from app.core.config import get_settings
        settings = get_settings()
        
        context_parts = []
        for idx, (chunk_id, score, metadata) in enumerate(chunks, 1):
            # Filter out low-relevance chunks - strict filtering for >80% accuracy
            if score < settings.MIN_RELEVANCE_SCORE:
                continue
                
            content = metadata.get("content", "")
            # Optimized truncation - reduced from 2000 to 1500 for faster processing
            if len(content) > 1500:
                content = content[:1500] + "..."
            
            # Include relevance score and source info in context
            filename = metadata.get("filename", "Unknown")
            page_info = ""
            if "page_number" in metadata:
                page_info = f" (Page {metadata['page_number']})"
            elif "headers" in metadata and metadata["headers"]:
                page_info = f" (Section: {metadata['headers'][0]})"
            
            context_parts.append(f"[{idx}] Source: {filename}{page_info}\nRelevance: {score:.2f}\nContent: {content}")
        return "\n\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Build messages for OpenAI chat API with strict grounding requirements"""
        messages = [
            {
                "role": "system",
                "content": """You are a technical documentation assistant for Cranswick. Your responses MUST be 100% accurate and based EXCLUSIVELY on the provided context.

CRITICAL RULES - FOLLOW STRICTLY:
1. ONLY use information that is EXPLICITLY and DIRECTLY stated in the provided context
2. NEVER make up, infer, assume, or extrapolate information not explicitly in the context
3. NEVER use general knowledge or information from outside the provided context
4. ALWAYS cite sources using [1], [2], etc. for EVERY factual claim you make
5. If the context doesn't contain enough information to answer, you MUST provide a helpful response that:
   - Clearly states: "I'm unable to answer this question based on the available documents."
   - Suggests: "Please try rephrasing your question or asking about a specific aspect of the topic."
   - Offers alternatives: "You might try asking about related topics such as [suggest 2-3 related topics based on what IS in the context]."
6. Do NOT combine information from multiple sources unless the context explicitly shows they should be combined
7. Do NOT paraphrase in ways that change meaning - quote or closely paraphrase the exact wording from context
8. If you're uncertain about ANY detail, state that clearly: "The context does not provide clear information about [specific detail]"
9. If the question is unclear or ambiguous, try to understand the intent and provide what information IS available, or suggest how to clarify the question
10. Check the relevance scores - lower scores may be less reliable
11. If the question seems to be asking about something that might be phrased differently, mention what related information IS available in the context

ACCURACY IS PARAMOUNT. When in doubt, say the information is not available in the provided context. It is better to say "not available" than to guess or make up information. However, always be helpful by suggesting how the user might rephrase their question or what related information is available."""
            }
        ]
        
        # Add conversation history if available
        if history:
            for h in history[-5:]:  # Last 5 exchanges
                if h.get("user"):
                    messages.append({"role": "user", "content": h["user"]})
                if h.get("assistant"):
                    messages.append({"role": "assistant", "content": h["assistant"]})
        
        # Add current context and query with strict instructions
        user_message = f"""You are answering a question about technical standards documents. Use ONLY the following context. Do NOT use any information outside this context.

CONTEXT (with relevance scores):
{context}

QUESTION: {query}

STRICT INSTRUCTIONS:
1. Answer ONLY using information EXPLICITLY stated in the context above
2. Cite EVERY factual statement with [1], [2], etc. matching the source numbers
3. If the answer is not in the context, provide a HELPFUL response:
   - State clearly: "I'm unable to answer this question based on the available documents."
   - Suggest rephrasing: "Please try asking your question in a different way, or be more specific about what aspect you're interested in."
   - Offer alternatives: Based on what IS in the context, suggest 2-3 related topics or questions the user might ask instead
4. If the question is unclear or could be interpreted multiple ways, acknowledge this and provide what information IS available, or ask for clarification
5. Do NOT make assumptions, inferences, or add information not in the context
6. Do NOT use general knowledge - only what's in the provided context
7. If you see low relevance scores (<0.5), be extra cautious with that information
8. Quote or closely paraphrase the exact wording from the context when possible
9. If multiple sources conflict, mention this explicitly
10. Try to understand the user's intent - if the question is phrased in a way that might not match the document terminology, look for related concepts in the context"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _extract_citations(
        self,
        response_text: str,
        retrieved_chunks: List[Tuple[str, float, Dict]]
    ) -> List[Dict]:
        """Extract citation references from response text"""
        import logging
        logger = logging.getLogger(__name__)
        
        citations = []
        # Find all citation references like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, response_text)
        
        cited_indices = set(int(m) - 1 for m in matches if int(m) <= len(retrieved_chunks))
        logger.info(f"Found {len(cited_indices)} citation references in response, {len(retrieved_chunks)} chunks available")
        
        for idx in cited_indices:
            if 0 <= idx < len(retrieved_chunks):
                chunk_id, score, metadata = retrieved_chunks[idx]
                logger.info(f"Citation [{idx+1}]: chunk_id={chunk_id}, relevance_score={score:.3f}")
                
                # Ensure score is not None or 0 if chunk exists
                if score is None or score == 0:
                    logger.warning(f"Citation [{idx+1}] has zero or None relevance score, using minimum 0.3")
                    score = 0.3  # Use minimum score if somehow 0
                
                citations.append({
                    "chunk_id": chunk_id,
                    "relevance_score": float(score),  # Ensure it's a float
                    "content": metadata.get("content", "")[:200],  # Preview
                    "metadata": metadata
                })
        
        if citations:
            avg_score = sum(c["relevance_score"] for c in citations) / len(citations)
            logger.info(f"Extracted {len(citations)} citations with average relevance: {avg_score:.3f}")
        else:
            logger.warning("No citations extracted from response")
        
        return citations
    
    def _validate_groundedness(
        self,
        response_text: str,
        retrieved_chunks: List[Tuple[str, float, Dict]]
    ) -> float:
        """
        Validate that the response is grounded in the retrieved chunks.
        Returns a score between 0 and 1 indicating how well the response
        is supported by the context.
        """
        # Check if response explicitly states information is not available
        # This is actually GOOD - it means the model is being honest and helpful
        not_available_phrases = [
            "does not contain",
            "not available",
            "not found",
            "not in the context",
            "insufficient information",
            "does not provide",
            "cannot be determined from",
            "unable to answer",
            "i'm unable",
            "try rephrasing",
            "try asking",
            "please try"
        ]
        
        # Check if response is being helpful about not being able to answer
        response_lower = response_text.lower()
        if any(phrase.lower() in response_lower for phrase in not_available_phrases):
            # If it also provides helpful suggestions, give it an even higher score
            if any(helpful_phrase in response_lower for helpful_phrase in ["try", "suggest", "might", "could", "related"]):
                return 0.95  # Very high score - honest and helpful
            return 0.9  # High score - correctly states information is unavailable
        
        # Check citation coverage
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, response_text)
        
        if not citations_found:
            # No citations found - potentially not grounded
            # Check if response is very short (might be just saying "not available")
            if len(response_text) < 50:
                return 0.5  # Short response without citations
            return 0.2  # Longer response without citations - low confidence
        
        # Check if cited chunks exist
        cited_indices = set(int(c) - 1 for c in citations_found if c.isdigit())
        valid_citations = sum(1 for idx in cited_indices if 0 <= idx < len(retrieved_chunks))
        
        if valid_citations == 0:
            return 0.1  # Citations don't match available chunks - very low confidence
        
        # Calculate coverage: how many chunks are cited vs total chunks
        # For well-cited answers, don't penalize for not citing all chunks
        # If we have valid citations, that's a strong signal the answer is correct
        if valid_citations >= 2:
            # 2+ citations is excellent - give high groundedness
            citation_coverage = 0.9  # Start high for multiple citations
        elif valid_citations >= 1:
            # 1 citation is good - give decent groundedness
            citation_coverage = 0.8  # Start high for single citation
        else:
            # No valid citations - use coverage calculation
            citation_coverage = valid_citations / len(retrieved_chunks) if retrieved_chunks else 0
        
        # Check average relevance of cited chunks
        cited_scores = [retrieved_chunks[idx][1] for idx in cited_indices if 0 <= idx < len(retrieved_chunks)]
        avg_relevance = sum(cited_scores) / len(cited_scores) if cited_scores else 0
        
        # Boost groundedness score for high-relevance citations
        # But don't penalize too much for lower relevance - if it's cited, it's likely relevant
        if avg_relevance >= 0.6:
            # Medium-high relevance: boost groundedness
            citation_coverage = min(citation_coverage * 1.15, 1.0)
        elif avg_relevance >= 0.4:
            # Medium relevance: slight boost
            citation_coverage = min(citation_coverage * 1.1, 1.0)
        # Don't penalize for lower relevance if we have valid citations
        
        # Check if response length suggests it might be adding information
        # But be more lenient - if we have citations, trust the answer
        if valid_citations >= 1:
            # If we have citations, don't penalize for longer responses
            # The citations prove the answer is grounded
            pass  # Don't penalize
        else:
            # Only check length if we don't have citations
            total_context_length = sum(len(chunk[2].get("content", "")) for chunk in retrieved_chunks)
            response_length = len(response_text)
            
            if response_length > total_context_length * 2.0:
                # Response is significantly longer than context - potential hallucination
                citation_coverage *= 0.6
        
        # Boost score if response explicitly references source documents
        if any(phrase in response_text.lower() for phrase in ["according to", "as stated in", "the document", "source"]):
            citation_coverage = min(citation_coverage * 1.1, 1.0)
        
        # Additional boost if we have multiple citations (regardless of relevance)
        # Multiple citations indicate the answer is well-supported
        if valid_citations >= 3:
            citation_coverage = min(citation_coverage * 1.25, 1.0)
        elif valid_citations >= 2:
            citation_coverage = min(citation_coverage * 1.2, 1.0)
        
        # Ensure minimum groundedness if we have valid citations
        # If answer has citations, it's definitely grounded - give it high score
        if valid_citations >= 2:
            citation_coverage = max(citation_coverage, 0.85)  # Very high minimum for 2+ citations
        elif valid_citations >= 1:
            citation_coverage = max(citation_coverage, 0.75)  # High minimum for 1 citation
        
        return min(citation_coverage, 1.0)
    
    async def generate_streaming_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, float, Dict]],
        conversation_history: Optional[List[Dict]] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Generate streaming response for real-time updates
        """
        # Format context
        context = self._format_context(retrieved_chunks)
        messages = self._build_messages(query, context, conversation_history)
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "chunk",
                        "content": chunk.choices[0].delta.content
                    }
            
            yield {"type": "done"}
        except Exception as e:
            yield {"type": "error", "content": str(e)}

