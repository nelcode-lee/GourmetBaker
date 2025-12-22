"""
Celery tasks for background document processing
"""
import asyncio
from celery import Celery
from app.core.config import get_settings
from app.services.document_service import DocumentService
from app.core.db.database import get_db_pool

settings = get_settings()

# Initialize Celery app
celery_app = Celery(
    "rag_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def run_async(coro):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(name="process_document")
def process_document_task(document_id: str):
    """
    Background task to process uploaded document:
    1. Parse document
    2. Semantic chunking
    3. Generate embeddings (batch process)
    4. Store chunks + embeddings in DB
    5. Update document status to 'ready'
    
    Args:
        document_id: UUID of document to process
    """
    document_service = DocumentService()
    
    async def process():
        # Get document info
        pool = await get_db_pool()
        doc = await document_service.get_document(pool, document_id)
        
        if not doc:
            return {"status": "failed", "error": "Document not found"}
        
        # Process document
        result = await document_service.process_document(
            document_id=document_id,
            file_path=doc["file_path"],
            file_type=doc["file_type"]
        )
        
        return result
    
    try:
        result = run_async(process())
        return result
    except Exception as e:
        return {"status": "failed", "error": str(e)}

