# Cranswick Technical Standards Agent

A production-ready RAG (Retrieval Augmented Generation) system for technical standards document intelligence with chat interface, document management, and analytics.

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI (Python 3.11+) with async support
- **Database**: Neon PostgreSQL with pgvector extension
- **Vector Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-4 (via API)
- **Task Queue**: Celery + Redis
- **Storage**: Local filesystem or S3-compatible (MinIO)

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── models/          # Pydantic schemas
│   │   │   └── routes/          # FastAPI route handlers
│   │   ├── core/
│   │   │   ├── config.py        # Application settings
│   │   │   ├── db/              # Database utilities
│   │   │   └── rag/             # RAG core modules
│   │   ├── services/            # Business logic services
│   │   ├── worker/              # Celery tasks
│   │   └── main.py              # FastAPI app entry point
│   ├── database_schema.sql      # Database schema
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── (to be created)
├── docker-compose.yml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18.0.0+ (Vite 5.x compatible)
- PostgreSQL with pgvector extension (Neon recommended)
- Redis

### Database Setup

1. Create a Neon PostgreSQL database with pgvector extension enabled
2. Run the schema:

```bash
psql $DATABASE_URL -f backend/database_schema.sql
```

Or use the Python initialization:

```python
from app.core.db.database import init_db
await init_db()
```

### Backend Setup

1. Create virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` file from `.env.example` and configure:

```bash
cp .env.example .env
# Edit .env with your settings
```

4. Run the server:

```bash
uvicorn app.main:app --reload
```

5. Start Celery worker (in separate terminal):

```bash
celery -A app.worker.tasks celery_app worker --loglevel=info
```

### Frontend Setup

1. Install Node.js 18.0.0+ (see [NODE_SETUP.md](NODE_SETUP.md) for detailed instructions):
   ```bash
   # Using nvm (recommended)
   nvm install 18
   nvm use 18
   
   # Or in frontend directory (uses .nvmrc)
   cd frontend
   nvm use
   ```

2. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

### Docker Setup

```bash
docker-compose up --build
```

## API Endpoints

### Document Management
- `POST /api/documents/upload` - Upload documents
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Delete document
- `PUT /api/documents/{id}/replace` - Replace document
- `GET /api/documents/{id}/chunks` - View document chunks

### Chat Interface
- `POST /api/chat/query` - Ask a question
- `GET /api/chat/history` - Get query history
- `POST /api/chat/feedback` - Submit feedback
- `GET /api/chat/citations/{query_id}` - Get citations

### Analytics
- `GET /api/analytics/overview` - Dashboard metrics
- `GET /api/analytics/queries` - Query patterns
- `GET /api/analytics/documents` - Document usage stats

## Development Status

✅ Project structure created
✅ Database schema defined
✅ Core configuration setup
✅ API route stubs created
✅ RAG module structure created

🚧 Implementation in progress:
- Document processing pipeline
- Semantic chunking
- Hybrid retrieval
- RAG generation
- Frontend components

## License

MIT

