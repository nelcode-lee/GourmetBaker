"""
Database connection and pool management using asyncpg
"""
import asyncpg
from typing import Optional
from app.core.config import get_settings

settings = get_settings()

_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """Get or create database connection pool"""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
    return _pool


async def close_db_pool():
    """Close database connection pool"""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def init_db():
    """Initialize database: create tables and extensions"""
    pool = await get_db_pool()
    
    async with pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename VARCHAR(255) NOT NULL,
                file_path TEXT NOT NULL,
                file_type VARCHAR(50) NOT NULL,
                file_size INTEGER NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(50) DEFAULT 'processing',
                version INTEGER DEFAULT 1,
                replaced_by UUID REFERENCES documents(id),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create document_chunks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384),
                metadata JSONB,
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, chunk_index)
            );
        """)
        
        # Create queries table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_text TEXT NOT NULL,
                user_id VARCHAR(255),
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create responses table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
                response_text TEXT NOT NULL,
                model VARCHAR(100),
                tokens_used INTEGER,
                latency_ms INTEGER,
                feedback INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create query_citations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS query_citations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
                chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
                relevance_score FLOAT,
                rank INTEGER
            );
        """)
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_queries_created ON queries(created_at);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_citations_query ON query_citations(query_id);")

