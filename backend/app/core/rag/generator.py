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
            # Call OpenAI API with lower temperature to reduce hallucination
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Lower temperature = more deterministic, less creative
                max_tokens=1000
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
        context_parts = []
        for idx, (chunk_id, score, metadata) in enumerate(chunks, 1):
            content = metadata.get("content", "")
            # Truncate very long chunks
            if len(content) > 1000:
                content = content[:1000] + "..."
            context_parts.append(f"[{idx}] {content}")
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
                "content": """You are a technical documentation assistant. Your responses MUST be 100% accurate and based ONLY on the provided context.

CRITICAL RULES:
1. ONLY use information explicitly stated in the provided context
2. NEVER make up, infer, or assume information not in the context
3. ALWAYS cite sources using [1], [2], etc. for every factual claim
4. If the context doesn't contain enough information to answer, say: "The provided context does not contain sufficient information to answer this question. Please refer to the source documents."
5. Do NOT combine information from multiple sources unless explicitly stated in the context
6. Do NOT add general knowledge or information not in the provided context
7. If you're uncertain, state that clearly

Your credibility depends on accuracy. When in doubt, say the information is not available in the provided context."""
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
        user_message = f"""Use ONLY the following context to answer the question. Do not use any information outside of this context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer ONLY using information from the context above
- Cite every factual statement with [1], [2], etc.
- If the answer is not in the context, explicitly state: "The provided context does not contain sufficient information to answer this question."
- Do not make assumptions or add information not in the context"""
        
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
            "insufficient information"
        ]
        
        if any(phrase.lower() in response_text.lower() for phrase in not_available_phrases):
            return 1.0  # High score - correctly states information is unavailable
        
        # Check citation coverage
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, response_text)
        
        if not citations_found:
            # No citations found - potentially not grounded
            return 0.3
        
        # Check if cited chunks exist
        cited_indices = set(int(c) - 1 for c in citations_found if c.isdigit())
        valid_citations = sum(1 for idx in cited_indices if 0 <= idx < len(retrieved_chunks))
        
        if valid_citations == 0:
            return 0.2  # Citations don't match available chunks
        
        # Calculate coverage: how many chunks are cited vs total chunks
        citation_coverage = valid_citations / len(retrieved_chunks) if retrieved_chunks else 0
        
        # Check if response length suggests it might be adding information
        # If response is much longer than context, it might be hallucinating
        total_context_length = sum(len(chunk[2].get("content", "")) for chunk in retrieved_chunks)
        response_length = len(response_text)
        
        if response_length > total_context_length * 1.5:
            # Response is significantly longer than context - potential hallucination
            citation_coverage *= 0.7
        
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

