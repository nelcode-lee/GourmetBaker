"""
Semantic chunking module for intelligent document splitting
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
import aiofiles
import re
from pathlib import Path

# Document parsing imports
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    import markdown
except ImportError:
    markdown = None


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    token_count: int
    chunk_index: int


class SemanticChunker:
    """
    Intelligent chunking that respects document structure
    and preserves context with overlap
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Use cl100k_base encoding (GPT-4 tokenizer)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    async def _parse_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse PDF file and extract text with page numbers"""
        if PdfReader is None:
            raise ImportError("pypdf is required for PDF parsing")
        
        pages = []
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        "text": text,
                        "page_number": page_num,
                        "type": "pdf"
                    })
        return pages
    
    async def _parse_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse DOCX file and extract text with structure"""
        if DocxDocument is None:
            raise ImportError("python-docx is required for DOCX parsing")
        
        sections = []
        doc = DocxDocument(file_path)
        
        current_section = {
            "text": "",
            "headers": [],
            "type": "docx"
        }
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Detect headings (simple heuristic: bold or style name)
            is_heading = para.style.name.startswith('Heading') or (
                para.runs and para.runs[0].bold
            )
            
            if is_heading:
                # Save previous section if it has content
                if current_section["text"]:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "text": text + "\n",
                    "headers": [text],
                    "type": "docx"
                }
            else:
                current_section["text"] += text + "\n"
        
        # Add last section
        if current_section["text"]:
            sections.append(current_section)
        
        return sections
    
    async def _parse_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse plain text file"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
        
        return [{
            "text": content,
            "type": "txt"
        }]
    
    async def _parse_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse Markdown file and extract text with structure"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
        
        sections = []
        lines = content.split('\n')
        current_section = {
            "text": "",
            "headers": [],
            "type": "md"
        }
        
        for line in lines:
            # Detect markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section["text"]:
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                current_section = {
                    "text": line + "\n",
                    "headers": [header_text],
                    "header_level": level,
                    "type": "md"
                }
            else:
                current_section["text"] += line + "\n"
        
        # Add last section
        if current_section["text"]:
            sections.append(current_section)
        
        return sections
    
    async def _extract_text(self, file_path: str, file_type: str) -> List[Dict[str, Any]]:
        """Extract text from document based on file type"""
        file_type_lower = file_type.lower().replace('.', '')
        
        if file_type_lower == 'pdf':
            return await self._parse_pdf(file_path)
        elif file_type_lower in ['docx', 'doc']:
            return await self._parse_docx(file_path)
        elif file_type_lower == 'md':
            return await self._parse_markdown(file_path)
        elif file_type_lower == 'txt':
            return await self._parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple approach)"""
        # Simple sentence splitting - can be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunks_with_overlap(
        self,
        text: str,
        base_metadata: Dict[str, Any],
        start_index: int = 0
    ) -> List[Chunk]:
        """
        Create chunks with overlap from text
        
        Args:
            text: Text to chunk
            base_metadata: Base metadata to include in all chunks
            start_index: Starting chunk index
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_index
        overlap_text = ""
        overlap_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata={**base_metadata, "chunk_type": "semantic"},
                    token_count=current_tokens,
                    chunk_index=chunk_index
                ))
                
                # Prepare overlap for next chunk
                # Get last N tokens worth of text for overlap
                overlap_sentences = []
                overlap_token_count = 0
                for s in reversed(sentences[:sentences.index(sentence)]):
                    s_tokens = self._count_tokens(s)
                    if overlap_token_count + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_token_count += s_tokens
                    else:
                        break
                
                overlap_text = " ".join(overlap_sentences)
                overlap_tokens = overlap_token_count
                
                # Start new chunk with overlap
                current_chunk = overlap_text + " " + sentence
                current_tokens = overlap_tokens + sentence_tokens
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={**base_metadata, "chunk_type": "semantic"},
                token_count=current_tokens,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    async def chunk_document(
        self, 
        file_path: str, 
        file_type: str
    ) -> List[Chunk]:
        """
        Parse document and create semantic chunks.
        
        Uses tiktoken for accurate token counting.
        Preserves document structure in metadata.
        
        Args:
            file_path: Path to document file
            file_type: File type (pdf, docx, txt, md)
            
        Returns:
            List of Chunk objects
        """
        # Extract text from document
        document_sections = await self._extract_text(file_path, file_type)
        
        all_chunks = []
        chunk_index = 0
        
        for section in document_sections:
            text = section["text"]
            if not text.strip():
                continue
            
            # Create base metadata from section
            base_metadata = {
                "file_type": section.get("type", file_type),
                "section_index": len(all_chunks)
            }
            
            # Add section-specific metadata
            if "page_number" in section:
                base_metadata["page_number"] = section["page_number"]
            if "headers" in section:
                base_metadata["headers"] = section["headers"]
            if "header_level" in section:
                base_metadata["header_level"] = section["header_level"]
            
            # Create chunks from this section
            section_chunks = self._create_chunks_with_overlap(
                text,
                base_metadata,
                start_index=chunk_index
            )
            
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return all_chunks

