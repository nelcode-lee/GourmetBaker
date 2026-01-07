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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Set to 0 for maximum determinism and accuracy
                max_tokens=1500,  # Increased to allow more detailed, accurate responses
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
            # Filter out low-relevance chunks
            if score < settings.MIN_RELEVANCE_SCORE:
                continue
                
            content = metadata.get("content", "")
            # Increased truncation limit for better context (was 1000, now 2000)
            if len(content) > 2000:
                content = content[:2000] + "..."
            
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
5. If the context doesn't contain enough information to answer, you MUST say: "The provided context does not contain sufficient information to answer this question. Please refer to the source documents."
6. Do NOT combine information from multiple sources unless the context explicitly shows they should be combined
7. Do NOT paraphrase in ways that change meaning - quote or closely paraphrase the exact wording from context
8. If you're uncertain about ANY detail, state that clearly: "The context does not provide clear information about [specific detail]"
9. If the question asks about something not in the context, explicitly state it's not available
10. Check the relevance scores - lower scores may be less reliable

ACCURACY IS PARAMOUNT. When in doubt, say the information is not available in the provided context. It is better to say "not available" than to guess or make up information."""
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
3. If the answer is not in the context, you MUST say: "The provided context does not contain sufficient information to answer this question."
4. Do NOT make assumptions, inferences, or add information not in the context
5. Do NOT use general knowledge - only what's in the provided context
6. If you see low relevance scores (<0.5), be extra cautious with that information
7. Quote or closely paraphrase the exact wording from the context when possible
8. If multiple sources conflict, mention this explicitly"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _extract_citations(
        self,
        response_text: str,
        retrieved_chunks: List[Tuple[str, float, Dict]]
    ) -> List[Dict]:
        """Extract citation references from response text"""
        citations = []
        # Find all citation references like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, response_text)
        
        cited_indices = set(int(m) - 1 for m in matches if int(m) <= len(retrieved_chunks))
        
        for idx in cited_indices:
            if 0 <= idx < len(retrieved_chunks):
                chunk_id, score, metadata = retrieved_chunks[idx]
                citations.append({
                    "chunk_id": chunk_id,
                    "relevance_score": score,
                    "content": metadata.get("content", "")[:200],  # Preview
                    "metadata": metadata
                })
        
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
        not_available_phrases = [
            "does not contain",
            "not available",
            "not found",
            "not in the context",
            "insufficient information",
            "does not provide",
            "cannot be determined from"
        ]
        
        if any(phrase.lower() in response_text.lower() for phrase in not_available_phrases):
            return 1.0  # High score - correctly states information is unavailable
        
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
        citation_coverage = valid_citations / len(retrieved_chunks) if retrieved_chunks else 0
        
        # Check average relevance of cited chunks
        cited_scores = [retrieved_chunks[idx][1] for idx in cited_indices if 0 <= idx < len(retrieved_chunks)]
        avg_relevance = sum(cited_scores) / len(cited_scores) if cited_scores else 0
        
        # Penalize if citing low-relevance chunks
        if avg_relevance < 0.4:
            citation_coverage *= 0.8
        
        # Check if response length suggests it might be adding information
        # If response is much longer than context, it might be hallucinating
        total_context_length = sum(len(chunk[2].get("content", "")) for chunk in retrieved_chunks)
        response_length = len(response_text)
        
        if response_length > total_context_length * 2.0:
            # Response is significantly longer than context - potential hallucination
            citation_coverage *= 0.6
        
        # Boost score if response explicitly references source documents
        if any(phrase in response_text.lower() for phrase in ["according to", "as stated in", "the document", "source"]):
            citation_coverage = min(citation_coverage * 1.1, 1.0)
        
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

