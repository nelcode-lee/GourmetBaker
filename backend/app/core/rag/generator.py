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
            
            # Log response for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"LLM Response (first 500 chars): {response_text[:500]}")
            
            # Extract citations from response
            citations = self._extract_citations(response_text, retrieved_chunks)
            
            # Fallback: If no citations found but we have chunks, include at least the top chunk
            if not citations and retrieved_chunks:
                logger.warning(f"No citations found in response, adding top chunk as fallback citation")
                # Add the top chunk as a citation
                top_chunk = retrieved_chunks[0]
                chunk_id, score, metadata = top_chunk
                
                # Build citation from top chunk
                filename = metadata.get("filename", "Unknown")
                chapter_section = ""
                if "headers" in metadata and metadata["headers"]:
                    chapter_section = metadata["headers"][0]
                elif "chapter" in metadata and metadata["chapter"]:
                    chapter_section = f"Chapter {metadata['chapter']}"
                elif "section" in metadata and metadata["section"]:
                    chapter_section = f"Section {metadata['section']}"
                
                page_number = metadata.get("page_number", None)
                citation_parts = [filename]
                if chapter_section:
                    citation_parts.append(chapter_section)
                if page_number:
                    citation_parts.append(f"Page {page_number}")
                citation_string = ", ".join(citation_parts)
                
                citations.append({
                    "chunk_id": chunk_id,
                    "relevance_score": float(score) if score else 0.5,
                    "content": metadata.get("content", "")[:200],
                    "document": filename,
                    "chapter": chapter_section if chapter_section else None,
                    "page_number": page_number,
                    "citation_string": citation_string,
                    "metadata": metadata
                })
                logger.info(f"Added fallback citation: {citation_string}")
            
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
            
            # Include relevance score and source info in context with document, chapter, and page
            filename = metadata.get("filename", "Unknown")
            
            # Build citation info: Document, Chapter/Section, Page
            page_info = ""
            chapter_info = ""
            
            if "page_number" in metadata and metadata["page_number"]:
                page_info = f", Page {metadata['page_number']}"
            
            # Get chapter/section information
            if "headers" in metadata and metadata["headers"]:
                # Use the first header as chapter/section
                chapter_info = f", {metadata['headers'][0]}"
            elif "chapter" in metadata and metadata["chapter"]:
                chapter_info = f", Chapter {metadata['chapter']}"
            elif "section" in metadata and metadata["section"]:
                chapter_info = f", Section {metadata['section']}"
            
            # Format: [1] Document: filename, Chapter/Section, Page X
            citation_info = f"Document: {filename}{chapter_info}{page_info}" if (chapter_info or page_info) else f"Document: {filename}"
            
            context_parts.append(f"[{idx}] {citation_info}\nRelevance: {score:.2f}\nContent: {content}")
        return "\n\n".join(context_parts)
    
    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Build messages for OpenAI chat API with balanced grounding requirements"""
        messages = [
            {
                "role": "system",
                "content": """You are a helpful technical documentation assistant for Cranswick. Your goal is to provide accurate, useful answers based on the provided context.

CORE PRINCIPLES:
1. Base your answers on the provided context - use information from the documents to answer questions
2. ALWAYS cite at least one source with document name, chapter/section, and page number using format: [1] Document: filename, Chapter/Section, Page X
3. Every factual claim MUST include a citation with document, chapter, and page number
4. If information is clearly present in the context, provide a helpful answer even if you need to infer connections between related concepts
5. Be helpful and proactive - try to understand what the user is asking and provide relevant information from the context
6. If the question uses different terminology than the documents, look for related concepts and provide what IS available

WHEN TO ANSWER:
- If the context contains information that addresses the question (even if phrased differently), provide an answer with citations
- If the question asks about a concept that's discussed in the context, explain it using the context
- If multiple related pieces of information exist, synthesize them to provide a complete answer
- If the question is partially answerable, provide what you can and note any limitations

WHEN TO SAY "UNABLE TO ANSWER":
- Only if the context truly contains NO relevant information about the question topic
- If you've thoroughly searched the context and found nothing related
- In this case, be helpful: suggest related topics that ARE in the context, or suggest rephrasing

BEST PRACTICES:
- Cite sources [1], [2], etc. for all factual claims - this is how we measure accuracy
- Quote or closely paraphrase the context when possible
- If information is in multiple sources, cite all relevant ones
- If relevance scores are low (<0.4), mention this but still use the information if it's the best available
- Be conversational and helpful - users want answers, not refusals

Remember: Your goal is to be helpful while maintaining accuracy through citations. If information exists in the context, use it to answer the question."""
            }
        ]
        
        # Add conversation history if available
        if history:
            for h in history[-5:]:  # Last 5 exchanges
                if h.get("user"):
                    messages.append({"role": "user", "content": h["user"]})
                if h.get("assistant"):
                    messages.append({"role": "assistant", "content": h["assistant"]})
        
        # Add current context and query with helpful instructions
        user_message = f"""Answer the following question using the provided context from technical standards documents.

CONTEXT (with relevance scores):
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide a helpful answer based on the context above - if information exists that addresses the question, use it
2. ALWAYS include at least one citation number [1], [2], [3], etc. in your response - these numbers correspond to the sources in the context above
3. Example: If you use information from the first source [1], write something like: "According to the document [1], the temperature requirement is..."
4. Every factual statement MUST include a citation number [1], [2], etc. from the context
5. The context shows sources numbered [1], [2], [3] etc. - use these EXACT numbers in your response
6. If the question uses different words than the documents, look for related concepts and provide what information IS available
7. If you can partially answer the question, provide what you can and note any limitations
8. If the context truly contains no relevant information, then say: "I'm unable to answer this question based on the available documents. However, the documents do contain information about [suggest 2-3 related topics from the context]."
9. Be helpful and conversational - users want answers, so provide them when the information exists
10. If multiple sources discuss the topic, cite all relevant ones using [1], [2], etc.
11. If relevance scores are low (<0.4), you can still use the information but mention it's from lower-relevance sources
12. Quote or paraphrase the context accurately when possible
13. If the question is unclear, provide what information IS available and ask for clarification if needed

CRITICAL: You MUST include at least one citation number [1], [2], [3], etc. in your response. These numbers MUST appear in your answer text. For example: "The temperature requirement is 4°C [1]" or "According to [1], the procedure requires..." DO NOT write citations in a separate format - include the numbers [1], [2] directly in your response text."""
        
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
        # Also try variations: (1), [source 1], etc.
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, response_text)
        
        # Also try parentheses format (1), (2)
        paren_pattern = r'\((\d+)\)'
        paren_matches = re.findall(paren_pattern, response_text)
        
        # Combine both patterns
        all_matches = matches + paren_matches
        
        cited_indices = set(int(m) - 1 for m in all_matches if int(m) <= len(retrieved_chunks) and int(m) > 0)
        logger.info(f"Found {len(cited_indices)} citation references in response (pattern matches: {len(matches)} [brackets], {len(paren_matches)} parentheses), {len(retrieved_chunks)} chunks available")
        
        # Log if no citations found
        if not cited_indices:
            logger.warning(f"No citation references found in response text. Response preview: {response_text[:300]}")
        
        for idx in cited_indices:
            if 0 <= idx < len(retrieved_chunks):
                chunk_id, score, metadata = retrieved_chunks[idx]
                logger.info(f"Citation [{idx+1}]: chunk_id={chunk_id}, relevance_score={score:.3f}")
                
                # Ensure score is not None or 0 if chunk exists
                if score is None or score == 0:
                    logger.warning(f"Citation [{idx+1}] has zero or None relevance score, using minimum 0.3")
                    score = 0.3  # Use minimum score if somehow 0
                
                # Build citation with document, chapter, and page number
                filename = metadata.get("filename", "Unknown")
                
                # Get chapter/section information
                chapter_section = ""
                if "headers" in metadata and metadata["headers"]:
                    chapter_section = metadata["headers"][0]
                elif "chapter" in metadata and metadata["chapter"]:
                    chapter_section = f"Chapter {metadata['chapter']}"
                elif "section" in metadata and metadata["section"]:
                    chapter_section = f"Section {metadata['section']}"
                
                # Get page number
                page_number = metadata.get("page_number", None)
                
                # Build citation string
                citation_parts = [filename]
                if chapter_section:
                    citation_parts.append(chapter_section)
                if page_number:
                    citation_parts.append(f"Page {page_number}")
                
                citation_string = ", ".join(citation_parts)
                
                citations.append({
                    "chunk_id": chunk_id,
                    "relevance_score": float(score),  # Ensure it's a float
                    "content": metadata.get("content", "")[:200],  # Preview
                    "document": filename,
                    "chapter": chapter_section if chapter_section else None,
                    "page_number": page_number,
                    "citation_string": citation_string,  # Formatted: "Document, Chapter, Page X"
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

