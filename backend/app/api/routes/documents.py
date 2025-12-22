"""
Document management API routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional
from uuid import UUID
from pathlib import Path
import asyncpg
from app.core.db.database import get_db_pool
from app.services.document_service import DocumentService
from app.api.models.schemas import (
    DocumentResponse,
    DocumentListResponse,
    ChunkResponse
)

router = APIRouter()

# Lazy initialization of DocumentService to handle optional embeddings
_document_service = None

def get_document_service():
    """Get DocumentService instance (lazy initialization)"""
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


async def process_document_background(document_id: str):
    """Background task to process document (works without Celery)"""
    import traceback
    document_service = get_document_service()
    pool = await get_db_pool()
    
    try:
        # Get document info
        doc = await document_service.get_document(pool, document_id)
        
        if not doc:
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE documents 
                    SET status = 'failed', metadata = jsonb_build_object('error', 'Document not found')
                    WHERE id = $1
                """, UUID(document_id))
            print(f"❌ Document {document_id} not found in database")
            return
        
        print(f"🔄 Processing document {document_id}: {doc['filename']}")
        
        # Process document
        result = await document_service.process_document(
            document_id=document_id,
            file_path=doc["file_path"],
            file_type=doc["file_type"]
        )
        
        print(f"✅ Document {document_id} processed successfully: {result.get('chunk_count', 0)} chunks")
        
    except Exception as e:
        # Update status to failed on error with error details
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ Error processing document {document_id}: {error_msg}")
        print(f"Traceback:\n{error_trace}")
        
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE documents 
                SET status = 'failed', 
                    metadata = jsonb_build_object('error', $2, 'traceback', $3)
                WHERE id = $1
            """, UUID(document_id), error_msg, error_trace)


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Upload single or multiple documents
    Returns document IDs for tracking processing status
    """
    import traceback
    document_service = get_document_service()
    uploaded_docs = []
    
    try:
        for file in files:
            try:
                if not file.filename:
                    raise ValueError("Filename is required")
                
                print(f"📤 Uploading file: {file.filename}")
                
                # Read file content
                content = await file.read()
                file_size = len(content)
                
                if file_size == 0:
                    raise ValueError("File is empty")
                
                # Determine file type
                file_type = Path(file.filename).suffix.lower().lstrip('.')
                if not file_type:
                    file_type = "txt"
                
                # Validate file type
                allowed_types = ['pdf', 'docx', 'txt', 'md', 'doc']
                if file_type not in allowed_types:
                    raise ValueError(f"File type '{file_type}' not supported. Allowed: {allowed_types}")
                
                # Save file
                file_path = await document_service.save_uploaded_file(content, file.filename)
                print(f"💾 File saved to: {file_path}")
                
                # Create document record
                doc_id = await document_service.create_document_record(
                    pool=pool,
                    filename=file.filename,
                    file_path=file_path,
                    file_type=file_type,
                    file_size=file_size
                )
                print(f"📝 Document record created: {doc_id}")
                
                # Trigger background processing (works without Celery)
                background_tasks.add_task(
                    process_document_background,
                    doc_id
                )
                
                uploaded_docs.append({
                    "id": doc_id,
                    "filename": file.filename,
                    "status": "processing"
                })
                
            except Exception as e:
                error_msg = f"Error uploading {file.filename}: {str(e)}"
                print(f"❌ {error_msg}")
                print(traceback.format_exc())
                uploaded_docs.append({
                    "id": None,
                    "filename": file.filename if file.filename else "unknown",
                    "status": "failed",
                    "error": error_msg
                })
        
        return {
            "message": f"Uploaded {len(uploaded_docs)} document(s)",
            "documents": uploaded_docs
        }
    
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    List all documents with optional filters
    """
    document_service = get_document_service()
    documents, total = await document_service.list_documents(
        pool=pool,
        status=status,
        page=page,
        page_size=page_size
    )
    
    # Convert to response models
    doc_responses = [
        DocumentResponse(**doc) for doc in documents
    ]
    
    return DocumentListResponse(
        documents=doc_responses,
        total=total,
        page=page,
        page_size=page_size
    )


@router.patch("/{document_id}/tags")
async def update_document_tags(
    document_id: UUID,
    core_area: Optional[str] = None,
    factory: Optional[str] = None,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Update document tags (core_area and/or factory)
    """
    import json
    async with pool.acquire() as conn:
        # Get current metadata
        current_meta = await conn.fetchval("""
            SELECT metadata FROM documents WHERE id = $1
        """, document_id)
        
        if current_meta is None:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Parse existing metadata
        if isinstance(current_meta, str):
            metadata = json.loads(current_meta) if current_meta else {}
        else:
            metadata = current_meta if current_meta else {}
        
        # Update tags
        if core_area is not None:
            metadata["core_area"] = core_area
        if factory is not None:
            metadata["factory"] = factory
        
        # Save updated metadata
        await conn.execute("""
            UPDATE documents 
            SET metadata = $1::jsonb, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
        """, json.dumps(metadata), document_id)
    
    return {"message": "Document tags updated", "document_id": str(document_id)}


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Get document details by ID
    """
    document_service = get_document_service()
    doc = await document_service.get_document(pool, str(document_id))
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(**doc)


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Delete document (cascade deletes chunks)
    """
    document_service = get_document_service()
    deleted = await document_service.delete_document(pool, str(document_id))
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted", "id": str(document_id)}


@router.put("/{document_id}/replace")
async def replace_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    Upload new version of existing document
    """
    document_service = get_document_service()
    # Check if old document exists
    old_doc = await document_service.get_document(pool, str(document_id))
    if not old_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Read new file
    content = await file.read()
    file_size = len(content)
    file_type = Path(file.filename).suffix.lower().lstrip('.') or "txt"
    
    # Save new file
    file_path = await document_service.save_uploaded_file(content, file.filename)
    
    # Get old version number
    old_version = old_doc.get("version", 1)
    
    # Create new document record
    new_doc_id = await document_service.create_document_record(
        pool=pool,
        filename=file.filename,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        metadata={"replaced_from": str(document_id)}
    )
    
    # Update old document to point to new version
    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE documents 
            SET replaced_by = $1, updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
        """, UUID(new_doc_id), document_id)
        
        # Update version number
        await conn.execute("""
            UPDATE documents 
            SET version = $1
            WHERE id = $2
        """, old_version + 1, UUID(new_doc_id))
    
    # Trigger background processing
    background_tasks.add_task(
        process_document_background,
        new_doc_id
    )
    
    return {
        "message": "Document replacement initiated",
        "old_id": str(document_id),
        "new_id": new_doc_id,
        "version": old_version + 1
    }


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: UUID,
    pool: asyncpg.Pool = Depends(get_db_pool)
):
    """
    View all chunks for a document
    """
    document_service = get_document_service()
    chunks = await document_service.get_document_chunks(pool, str(document_id))
    
    return [ChunkResponse(**chunk) for chunk in chunks]

