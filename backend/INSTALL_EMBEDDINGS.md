# Installing Embeddings (Optional but Recommended)

## Current Status

The FastAPI server can start without embeddings, but **embeddings are required for the full RAG functionality** (document search and chat queries).

## Why Embeddings Are Needed

- **Document Processing**: Generate vector embeddings for document chunks
- **Query Processing**: Generate embeddings for user queries
- **Semantic Search**: Find relevant documents using vector similarity

## Installation Options

### Option 1: Wait for Python 3.13 Support (Recommended)

Torch (required for sentence-transformers) doesn't have official wheels for Python 3.13 yet. 

**Solution**: Use Python 3.11 or 3.12 instead:

```bash
# Create new venv with Python 3.11 or 3.12
cd backend
rm -rf venv
python3.11 -m venv venv  # or python3.12
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Install from Source (Advanced)

If you want to stick with Python 3.13, you can try building torch from source (not recommended for production).

### Option 3: Use Alternative Embedding Service

We could modify the code to use OpenAI's embedding API instead of sentence-transformers (requires API key).

## Current Workaround

The system will work for:
- ✅ Document upload and storage
- ✅ Document listing and management
- ✅ Basic document chunking

But will show errors for:
- ❌ Document processing (embeddings generation)
- ❌ Chat queries (requires embeddings)

## Quick Check

To see if embeddings are available:
```bash
cd backend
source venv/bin/activate
python3 -c "from app.core.rag.embeddings import SENTENCE_TRANSFORMERS_AVAILABLE; print('Available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Not Available')"
```

## Recommendation

For now, the server can start and you can test the API endpoints. For full functionality, switch to Python 3.11 or 3.12 and install the full requirements.

