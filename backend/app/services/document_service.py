"""
Document processing service
Handles document upload, parsing, chunking, and embedding generation
"""
import os
import uuid
from pathlib import Path
from typing import List, Optional
import asyncpg
from app.core.config import get_settings
from app.core.rag.chunker import SemanticChunker
from app.core.db.database import get_db_pool

settings = get_settings()

# Try to import EmbeddingService, make it optional
try:
    from app.core.rag.embeddings import EmbeddingService
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    EmbeddingService = None


class DocumentService:
    """Service for managing document processing pipeline"""
    
    def __init__(self):
        self.chunker = SemanticChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        if EMBEDDINGS_AVAILABLE:
            self.embedding_service = EmbeddingService()
        else:
            self.embedding_service = None
    
    async def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str
    ) -> str:
        """
        Save uploaded file to storage
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Path where file was saved
        """
        # Create uploads directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        file_ext = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = upload_dir / unique_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path)
    
    async def create_document_record(
        self,
        pool: asyncpg.Pool,
        filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Create document record in database
        
        Returns:
            Document ID (UUID as string)
        """
        async with pool.acquire() as conn:
            doc_id = await conn.fetchval("""
                INSERT INTO documents (filename, file_path, file_type, file_size, metadata, status)
                VALUES ($1, $2, $3, $4, $5, 'processing')
                RETURNING id
            """, filename, file_path, file_type, file_size, metadata)
            
            return str(doc_id)
    
    async def process_document(
        self,
        document_id: str,
        file_path: str,
        file_type: str
    ) -> dict:
        """
        Process document: chunk, embed, and store in database
        
        This is the main processing pipeline:
        1. Parse and chunk document
        2. Generate embeddings (batch)
        3. Store chunks and embeddings in DB
        4. Update document status
        
        Args:
            document_id: Document UUID
            file_path: Path to document file
            file_type: File type extension
            
        Returns:
            Processing result with chunk count
        """
        pool = await get_db_pool()
        
        try:
            # Step 1: Chunk document
            chunks = await self.chunker.chunk_document(file_path, file_type)
            
            if not chunks:
                # Update status to failed
                async with pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE documents 
                        SET status = 'failed' 
                        WHERE id = $1
                    """, uuid.UUID(document_id))
                return {"status": "failed", "error": "No chunks created"}
            
            # Step 2: Generate embeddings in batch (if available)
            if self.embedding_service:
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            else:
                # Placeholder embeddings (zeros) if sentence-transformers not available
                embeddings = [[0.0] * settings.EMBEDDING_DIMENSION for _ in chunks]
                print("⚠️  Warning: Embeddings not available. Install sentence-transformers and torch for full functionality.")
            
            # Step 3: Store chunks and embeddings
            import json
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for chunk, embedding in zip(chunks, embeddings):
                        # Convert embedding list to pgvector format
                        # pgvector expects the format: '[1.0, 2.0, 3.0]'::vector
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        
                        # Ensure metadata is JSON-serializable
                        metadata_json = json.dumps(chunk.metadata) if chunk.metadata else None
                        
                        await conn.execute("""
                            INSERT INTO document_chunks 
                            (document_id, chunk_index, content, embedding, metadata, token_count)
                            VALUES ($1, $2, $3, $4::vector, $5::jsonb, $6)
                        """, 
                            uuid.UUID(document_id),
                            chunk.chunk_index,
                            chunk.content,
                            embedding_str,
                            metadata_json,
                            chunk.token_count
                        )
                    
                    # Update document status to ready
                    await conn.execute("""
                        UPDATE documents 
                        SET status = 'ready', updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                    """, uuid.UUID(document_id))
            
            return {
                "status": "ready",
                "document_id": document_id,
                "chunk_count": len(chunks)
            }
            
        except Exception as e:
            # Update status to failed
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE documents 
                    SET status = 'failed' 
                    WHERE id = $1
                """, uuid.UUID(document_id))
            
            raise Exception(f"Document processing failed: {str(e)}")
    
    async def get_document(
        self,
        pool: asyncpg.Pool,
        document_id: str
    ) -> Optional[dict]:
        """Get document by ID"""
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, filename, file_path, file_type, file_size,
                       upload_date, status, version, replaced_by, metadata,
                       created_at, updated_at
                FROM documents
                WHERE id = $1
            """, uuid.UUID(document_id))
            
            if row:
                return dict(row)
            return None
    
    async def list_documents(
        self,
        pool: asyncpg.Pool,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> tuple[List[dict], int]:
        """List documents with pagination"""
        async with pool.acquire() as conn:
            # Build query
            where_clause = "WHERE 1=1"
            params = []
            param_idx = 1
            
            if status:
                where_clause += f" AND status = ${param_idx}"
                params.append(status)
                param_idx += 1
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM documents {where_clause}"
            total = await conn.fetchval(count_query, *params)
            
            # Get paginated results
            offset = (page - 1) * page_size
            query = f"""
                SELECT id, filename, file_path, file_type, file_size,
                       upload_date, status, version, replaced_by, metadata,
                       created_at, updated_at
                FROM documents
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([page_size, offset])
            
            rows = await conn.fetch(query, *params)
            documents = [dict(row) for row in rows]
            
            return documents, total
    
    async def delete_document(
        self,
        pool: asyncpg.Pool,
        document_id: str
    ) -> bool:
        """Delete document (cascade deletes chunks)"""
        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM documents WHERE id = $1
            """, uuid.UUID(document_id))
            
            return result == "DELETE 1"
    
    async def get_document_chunks(
        self,
        pool: asyncpg.Pool,
        document_id: str
    ) -> List[dict]:
        """Get all chunks for a document"""
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, document_id, chunk_index, content, metadata, token_count, created_at
                FROM document_chunks
                WHERE document_id = $1
                ORDER BY chunk_index
            """, uuid.UUID(document_id))
            
            return [dict(row) for row in rows]

