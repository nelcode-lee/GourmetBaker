# RAG Document Intelligence System - Cursor AI Build Instructions

## Project Overview
Build a production-ready RAG (Retrieval Augmented Generation) system for document intelligence with chat interface, document management, and analytics.

## Tech Stack
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI (Python 3.11+) with async support
- **Database**: Neon PostgreSQL with pgvector extension
- **Vector Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-4 (via API)
- **Task Queue**: Celery + Redis
- **Storage**: Local filesystem or S3-compatible (MinIO)

## Architecture Principles
1. **Async-first**: All I/O operations should be async
2. **Type safety**: Full TypeScript on frontend, Pydantic on backend
3. **Modular RAG**: Separate chunking, embedding, retrieval, and generation
4. **Observability**: Log every step of the RAG pipeline
5. **Scalability**: Design for horizontal scaling from day one

## Database Schema

### Tables to Create

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INTEGER NOT NULL,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'processing', -- processing, ready, failed
    version INTEGER DEFAULT 1,
    replaced_by UUID REFERENCES documents(id),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunks table with vectors
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384), -- dimension for all-MiniLM-L6-v2
    metadata JSONB, -- page_number, section, etc.
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, chunk_index)
);

-- Queries table
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    user_id VARCHAR(255), -- optional user tracking
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Responses table
CREATE TABLE responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
    response_text TEXT NOT NULL,
    model VARCHAR(100),
    tokens_used INTEGER,
    latency_ms INTEGER,
    feedback INTEGER, -- 1 (thumbs up), -1 (thumbs down), null
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Query-chunk citations (which chunks were used)
CREATE TABLE query_citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES queries(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES document_chunks(id) ON DELETE CASCADE,
    relevance_score FLOAT,
    rank INTEGER
);

-- Indexes for performance
CREATE INDEX idx_chunks_document ON document_chunks(document_id);
CREATE INDEX idx_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_queries_created ON queries(created_at);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_citations_query ON query_citations(query_id);
```

## Backend Implementation Guide

### 1. Core Configuration (`app/core/config.py`)

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "RAG Document Intelligence"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # RAG Parameters
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Storage
    UPLOAD_DIR: str = "./uploads"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

### 2. Semantic Chunking (`app/core/rag/chunker.py`)

**Requirements**: Implement intelligent chunking that:
- Respects document structure (paragraphs, sections)
- Uses sliding window with overlap for context preservation
- Extracts metadata (page numbers, headers)
- Handles multiple file types (PDF, DOCX, TXT, MD)

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: Dict
    token_count: int
    chunk_index: int

class SemanticChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    async def chunk_document(self, file_path: str, file_type: str) -> List[Chunk]:
        """
        Parse document and create semantic chunks.
        Use tiktoken for accurate token counting.
        Preserve document structure in metadata.
        """
        pass
```

### 3. RAG Retriever (`app/core/rag/retriever.py`)

**Requirements**: Implement hybrid retrieval:
- Vector similarity search using pgvector
- BM25 keyword search for exact matches
- Reranking using cross-encoder
- MMR (Maximal Marginal Relevance) for diversity

```python
from typing import List, Tuple
import asyncpg

class HybridRetriever:
    def __init__(self, db_pool: asyncpg.Pool):
        self.db = db_pool
    
    async def retrieve(
        self, 
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        document_ids: List[str] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        1. Vector search (cosine similarity)
        2. BM25 keyword search
        3. Combine scores with weights (0.7 vector, 0.3 keyword)
        4. Rerank top 20 results
        5. Return top_k with metadata
        """
        pass
```

### 4. FastAPI Routes

**Implement these endpoints**:

```python
# Document Management
POST   /api/documents/upload          # Upload single/multiple documents
GET    /api/documents                 # List all documents with filters
GET    /api/documents/{id}            # Get document details
DELETE /api/documents/{id}            # Delete document (soft delete)
PUT    /api/documents/{id}/replace    # Upload new version
GET    /api/documents/{id}/chunks     # View document chunks

# Chat Interface
POST   /api/chat/query                # Ask a question
GET    /api/chat/history              # Get query history
POST   /api/chat/feedback             # Submit thumbs up/down
GET    /api/chat/citations/{query_id} # Get source citations

# Analytics
GET    /api/analytics/overview        # Dashboard metrics
GET    /api/analytics/queries         # Query patterns
GET    /api/analytics/documents       # Document usage stats
```

### 5. Document Processing Pipeline

**Background Task Flow**:
1. Upload document → Save to storage → Create DB entry (status: processing)
2. Celery task triggered:
   - Parse document
   - Semantic chunking
   - Generate embeddings (batch process)
   - Store chunks + embeddings in DB
   - Update document status to 'ready'
3. If document is a replacement:
   - Mark old version's chunks as archived
   - Preserve query history linking to old version

### 6. RAG Generation (`app/core/rag/generator.py`)

```python
class RAGGenerator:
    async def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Tuple[str, float, Dict]],
        conversation_history: List[Dict] = None
    ) -> Dict:
        """
        1. Build prompt with retrieved context
        2. Add conversation history if available
        3. Call OpenAI API
        4. Extract citations
        5. Return response + metadata (tokens, latency)
        """
        
        prompt = f"""You are a helpful assistant. Answer the question based on the context below.

Context:
{self._format_context(retrieved_chunks)}

Question: {query}

Answer with citations [1], [2], etc. If the context doesn't contain the answer, say so."""
        
        # Implement streaming support for real-time responses
        pass
```

## Frontend Implementation Guide

### 1. Project Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install @tanstack/react-query axios zustand
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npx shadcn-ui@latest init
```

### 2. Core Components to Build

#### `src/components/chat/ChatInterface.tsx`
- Message list with streaming support
- Input with file attachment
- Citation bubbles (click to see source)
- Thumbs up/down feedback
- Loading states with skeleton screens

#### `src/components/documents/DocumentManager.tsx`
- Drag-and-drop upload with progress
- Document grid/list view
- Search and filter
- Document version history
- Delete with confirmation modal

#### `src/components/analytics/AnalyticsDashboard.tsx`
- Key metrics cards (total docs, queries, avg response time)
- Query volume chart (Recharts)
- Most queried topics (word cloud or bar chart)
- Document usage heatmap

### 3. State Management (`src/lib/store.ts`)

```typescript
import { create } from 'zustand';

interface ChatStore {
  messages: Message[];
  isLoading: boolean;
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isLoading: false,
  addMessage: (message) =>
    set((state) => ({ messages: [...state.messages, message] })),
  setLoading: (loading) => set({ isLoading: loading }),
}));
```

### 4. API Service (`src/services/api.ts`)

```typescript
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
});

export const chatAPI = {
  sendQuery: (query: string, documentIds?: string[]) =>
    api.post('/chat/query', { query, document_ids: documentIds }),
  
  getHistory: (limit: number = 50) =>
    api.get('/chat/history', { params: { limit } }),
  
  submitFeedback: (queryId: string, feedback: number) =>
    api.post('/chat/feedback', { query_id: queryId, feedback }),
};

export const documentAPI = {
  upload: (files: FileList, onProgress?: (progress: number) => void) => {
    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file));
    
    return api.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total));
        }
      },
    });
  },
  
  list: (filters?: DocumentFilters) =>
    api.get('/documents', { params: filters }),
  
  delete: (id: string) => api.delete(`/documents/${id}`),
};
```

## Key Implementation Details

### Document Replacement Strategy
When a document is updated:
1. Upload new version
2. Process new version (chunks + embeddings)
3. Mark old version with `replaced_by` = new document ID
4. Keep old chunks for query history integrity
5. New queries only search against latest version
6. Analytics can show version comparison

### Semantic Chunking Algorithm
```
1. Parse document preserving structure
2. Split on semantic boundaries:
   - Section headers
   - Paragraph breaks
   - Sentence boundaries (if needed)
3. Ensure chunks are 400-600 tokens (not strict character count)
4. Add overlap: last 50 tokens of chunk N = first 50 tokens of chunk N+1
5. Metadata: page, section, heading hierarchy, position
```

### Query Analytics to Track
- Query frequency and patterns
- Response quality (feedback ratio)
- Average response time
- Most cited documents/chunks
- Query clustering (similar questions)
- Dead-end queries (no good results)

### Performance Optimizations
1. **Batch embedding generation**: Process multiple chunks at once
2. **Connection pooling**: Use asyncpg pool for database
3. **Caching**: Redis for frequent queries
4. **Lazy loading**: Paginate document lists and chat history
5. **Index optimization**: Proper indexes on foreign keys and search fields

## Docker Compose Setup

```yaml
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000/api
    depends_on:
      - backend
  
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - redis
    volumes:
      - ./uploads:/app/uploads
  
  celery-worker:
    build: ./backend
    command: celery -A app.worker worker --loglevel=info
    env_file:
      - .env
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## Testing Strategy

### Backend Tests
- Unit tests for chunking logic
- Integration tests for RAG pipeline
- API endpoint tests with pytest
- Load testing with Locust

### Frontend Tests
- Component tests with Vitest + Testing Library
- E2E tests with Playwright
- Accessibility tests

## Deployment Checklist

- [ ] Environment variables secured
- [ ] Database migrations run
- [ ] pgvector extension enabled
- [ ] File upload limits configured
- [ ] CORS properly configured
- [ ] Rate limiting on API endpoints
- [ ] Monitoring (Sentry for errors)
- [ ] Logging (structured JSON logs)
- [ ] Backup strategy for documents + database

## Getting Started

1. **Database Setup**:
   ```bash
   # Create Neon database with pgvector
   # Run migrations from alembic
   ```

2. **Backend**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

3. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Celery Worker**:
   ```bash
   celery -A app.worker worker --loglevel=info
   ```

## Next Steps After MVP

1. **Multi-tenant support**: Separate documents by organization
2. **Advanced chunking**: Try RecursiveCharacterTextSplitter from LangChain
3. **Multiple LLM support**: Add Anthropic, Gemini as alternatives
4. **Document preprocessing**: OCR for scanned PDFs, table extraction
5. **Query suggestions**: Show related questions
6. **Export functionality**: Export chat history, analytics reports
7. **API key management**: Let users bring their own OpenAI keys
8. **Collaboration**: Share documents and chats with team members

## Notes for Cursor

- Use `@workspace` to reference this entire prompt
- Use `@docs` to pull in specific library documentation as needed
- For complex RAG logic, ask Cursor to explain trade-offs before implementing
- Request code in small, testable chunks
- Ask Cursor to add comprehensive docstrings and type hints
- Use Cursor's "Cmd+K" to refactor code sections iteratively

---

**Start by asking Cursor to create the project structure and database schema, then proceed component by component.**