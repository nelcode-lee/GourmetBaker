# Python Version Setup

## Current Setup

✅ **Python 3.12.12** - Installed and configured
✅ **Virtual Environment** - Created with Python 3.12
✅ **All Dependencies** - Installed including torch and sentence-transformers

## Using the Virtual Environment

Always activate the virtual environment before running commands:

```bash
cd backend
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Verify Python Version

```bash
cd backend
source venv/bin/activate
python --version
# Should show: Python 3.12.12
```

## Start Server

```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

## All Features Available

With Python 3.12, you now have:
- ✅ Full RAG functionality
- ✅ Document embeddings
- ✅ Query embeddings
- ✅ Vector similarity search
- ✅ Chat interface with semantic search

## Note

If you switch terminals or close the terminal, remember to:
1. `cd backend`
2. `source venv/bin/activate`
3. Then run your commands

