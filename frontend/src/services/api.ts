import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api',
})

export const chatAPI = {
  sendQuery: (query: string, options?: { documentIds?: string[]; coreArea?: string; factory?: string }) =>
    api.post('/chat/query', { 
      query, 
      document_ids: options?.documentIds,
      core_area: options?.coreArea,
      factory: options?.factory
    }, {
      timeout: 120000 // 2 minute timeout
    }),
  
  getHistory: (limit: number = 50) =>
    api.get('/chat/history', { params: { limit } }),
  
  submitFeedback: (queryId: string, feedback: number) =>
    api.post('/chat/feedback', { query_id: queryId, feedback }),
  
  getCitations: (queryId: string) =>
    api.get(`/chat/citations/${queryId}`),
  
  getSuggestions: (query: string, limit?: number) =>
    api.get('/chat/suggestions', { params: { query, limit } }),
}

export const documentAPI = {
  upload: (files: FileList, onProgress?: (progress: number) => void) => {
    const formData = new FormData()
    Array.from(files).forEach(file => formData.append('files', file))
    
    return api.post('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total))
        }
      },
    })
  },
  
  list: (filters?: { status?: string; page?: number; page_size?: number }) =>
    api.get('/documents', { params: filters }),
  
  get: (id: string) => api.get(`/documents/${id}`),
  
  delete: (id: string) => api.delete(`/documents/${id}`),
  
  replace: (id: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.put(`/documents/${id}/replace`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
  
  getChunks: (id: string) => api.get(`/documents/${id}/chunks`),
  
  updateTags: (id: string, tags: { core_area?: string; factory?: string }) =>
    api.patch(`/documents/${id}/tags`, null, { params: tags }),
}

export const analyticsAPI = {
  getOverview: () => api.get('/analytics/overview'),
  getQueryPatterns: () => api.get('/analytics/queries'),
  getDocumentUsage: () => api.get('/analytics/documents'),
  getQueriesByChunkTags: () => api.get('/analytics/queries/by-chunk-tags'),
  getQueriesByDocuments: () => api.get('/analytics/queries/by-documents'),
  getQueriesByKeyTerms: () => api.get('/analytics/queries/by-key-terms'),
  getQualityMetrics: () => api.get('/analytics/quality-metrics'),
}

