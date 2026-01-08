"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


# Document Models
class DocumentBase(BaseModel):
    filename: str
    file_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    file_path: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    id: UUID
    file_path: str
    upload_date: datetime
    status: str
    version: int
    replaced_by: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


# Chunk Models
class ChunkResponse(BaseModel):
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    metadata: Optional[Dict[str, Any]] = None
    token_count: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Chat Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[UUID]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    core_area: Optional[str] = None  # BRCGS, Tesco, Marks and Spencer, Asda, Hello Fresh, Morrisons
    factory: Optional[str] = None  # Gourmet Pastry, Gourmet Kitchen, Gourmet Sausage, Convenience foods, Fresh Poultry, Prepared Poultry, Gourmet Bacon, Continental Foods


class CitationResponse(BaseModel):
    chunk_id: UUID
    content: str
    relevance_score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    query_id: UUID
    response_text: str
    citations: List[CitationResponse]
    model: str
    tokens_used: int
    latency_ms: int
    created_at: datetime
    groundedness_score: Optional[float] = None  # 0-1 score indicating how well response is grounded
    avg_relevance_score: Optional[float] = None  # Average relevance score of all citations (0-1)
    overall_confidence: Optional[float] = None  # Overall confidence score (0-1, weighted combination)


class QueryHistoryItem(BaseModel):
    id: UUID
    query_text: str
    response_text: str
    created_at: datetime
    feedback: Optional[int] = None


class FeedbackRequest(BaseModel):
    query_id: UUID
    feedback: int = Field(..., ge=-1, le=1)  # -1 (thumbs down), 1 (thumbs up)


# Analytics Models
class AnalyticsOverview(BaseModel):
    total_documents: int
    total_queries: int
    total_chunks: int
    avg_response_time_ms: float
    positive_feedback_ratio: float
    documents_by_status: Dict[str, int]


class QueryPattern(BaseModel):
    query_text: str
    frequency: int
    avg_feedback: Optional[float] = None


class DocumentUsageStats(BaseModel):
    document_id: UUID
    filename: str
    query_count: int
    avg_relevance_score: float
    last_queried: Optional[datetime] = None


class ChunkTagStats(BaseModel):
    tag: str  # Could be header, page number, section, etc.
    tag_type: str  # 'header', 'page', 'section', etc.
    query_count: int
    avg_relevance_score: float
    document_id: Optional[UUID] = None
    filename: Optional[str] = None


class DocumentQueryStats(BaseModel):
    document_id: UUID
    filename: str
    query_count: int
    avg_relevance_score: float
    positive_feedback_count: int
    negative_feedback_count: int
    last_queried: Optional[datetime] = None
    core_area: Optional[str] = None
    factory: Optional[str] = None


class KeyTermStats(BaseModel):
    term: str
    frequency: int
    avg_feedback: Optional[float] = None
    unique_queries: int  # Number of distinct queries containing this term


class QualityMetrics(BaseModel):
    feedback_distribution: Dict[str, int]  # positive, negative, neutral
    confidence_distribution: Dict[str, int]  # high, medium, low
    relevance_distribution: Dict[str, int]  # high, medium, low
    document_status_distribution: Dict[str, int]  # ready, processing, failed
    avg_confidence_score: float
    avg_relevance_score: float
    avg_groundedness_score: float
    total_with_feedback: int
    total_with_confidence: int
    training_quality_score: float  # Composite score 0-100 indicating how well trained the agent is

