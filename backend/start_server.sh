#!/bin/bash
# Helper script to start the FastAPI server

cd "$(dirname "$0")"

# Kill any existing server on port 8000
echo "Checking for existing server on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "✅ Killed existing process" || echo "No existing process"

# Wait a moment
sleep 2

# Activate virtual environment and start server
echo "Starting FastAPI server..."
source venv/bin/activate
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
