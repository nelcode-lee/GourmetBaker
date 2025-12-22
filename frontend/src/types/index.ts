export interface Document {
  id: string
  filename: string
  file_path: string
  file_type: string
  file_size: number
  upload_date: string
  status: 'processing' | 'ready' | 'failed'
  version: number
  replaced_by?: string
  metadata?: {
    core_area?: string
    factory?: string
    [key: string]: any
  }
  created_at: string
  updated_at: string
}

export interface Chunk {
  id: string
  document_id: string
  chunk_index: number
  content: string
  metadata?: Record<string, any>
  token_count?: number
  created_at: string
}

export interface QueryResponse {
  query_id: string
  response_text: string
  citations: Citation[]
  model: string
  tokens_used: number
  latency_ms: number
  created_at: string
  groundedness_score?: number  // 0-1 score indicating how well response is grounded in sources
}

export interface Citation {
  chunk_id: string
  content: string
  relevance_score: number
  rank: number
  metadata?: Record<string, any>
}

export interface QueryHistoryItem {
  id: string
  query_text: string
  response_text: string
  created_at: string
  feedback?: number
}

export interface AnalyticsOverview {
  total_documents: number
  total_queries: number
  total_chunks: number
  avg_response_time_ms: number
  positive_feedback_ratio: number
  documents_by_status: Record<string, number>
}

export interface QueryPattern {
  query_text: string
  frequency: number
  avg_feedback?: number
}

export interface DocumentUsageStats {
  document_id: string
  filename: string
  query_count: number
  avg_relevance_score: number
  last_queried?: string
}

export interface ChunkTagStats {
  tag: string
  tag_type: 'header' | 'page' | 'section' | 'type'
  query_count: number
  avg_relevance_score: number
  document_id?: string
  filename?: string
}

export interface DocumentQueryStats {
  document_id: string
  filename: string
  query_count: number
  avg_relevance_score: number
  positive_feedback_count: number
  negative_feedback_count: number
  last_queried?: string
  core_area?: string
  factory?: string
}

export interface KeyTermStats {
  term: string
  frequency: number
  avg_feedback?: number
  unique_queries: number
}

