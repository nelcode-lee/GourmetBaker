import { useQuery } from '@tanstack/react-query'
import { analyticsAPI } from '../../services/api'
import { DocumentQueryStats, KeyTermStats } from '../../types'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

export default function AnalyticsDashboard() {
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ['analytics-overview'],
    queryFn: async () => {
      const response = await analyticsAPI.getOverview()
      return response.data
    },
  })

  const { data: queryPatterns, isLoading: patternsLoading } = useQuery({
    queryKey: ['analytics-queries'],
    queryFn: async () => {
      const response = await analyticsAPI.getQueryPatterns()
      return response.data
    },
  })

  const { data: documentUsage, isLoading: usageLoading } = useQuery({
    queryKey: ['analytics-documents'],
    queryFn: async () => {
      const response = await analyticsAPI.getDocumentUsage()
      return response.data
    },
  })

  const { data: documentQueryStats, isLoading: docQueriesLoading } = useQuery({
    queryKey: ['analytics-document-queries'],
    queryFn: async () => {
      const response = await analyticsAPI.getQueriesByDocuments()
      return response.data
    },
  })

  const { data: keyTermStats, isLoading: keyTermsLoading } = useQuery({
    queryKey: ['analytics-key-terms'],
    queryFn: async () => {
      const response = await analyticsAPI.getQueriesByKeyTerms()
      return response.data
    },
  })

  if (overviewLoading || patternsLoading || usageLoading || docQueriesLoading || keyTermsLoading) {
    return <div className="text-center text-gray-500">Loading analytics...</div>
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500">Total Documents</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {overview?.total_documents || 0}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500">Total Queries</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {overview?.total_queries || 0}
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500">Avg Response Time</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {overview?.avg_response_time_ms?.toFixed(0) || 0}ms
          </p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500">Positive Feedback</h3>
          <p className="text-3xl font-bold text-gray-900 mt-2">
            {((overview?.positive_feedback_ratio || 0) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-2">Queries by Document</h2>
        <p className="text-sm text-gray-500 mb-4">
          Shows which documents are queried most frequently and their performance metrics
        </p>
        {documentQueryStats && documentQueryStats.length > 0 ? (
          <div>
            <ResponsiveContainer width="100%" height={500}>
              <BarChart 
                data={documentQueryStats.slice(0, 15)} 
                margin={{ bottom: 120, left: 20, right: 20, top: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="filename" 
                  angle={-45} 
                  textAnchor="end" 
                  height={180}
                  tick={{ fontSize: 9 }}
                  interval={0}
                  tickFormatter={(value) => {
                    // Truncate long filenames to prevent overlap
                    if (value.length > 30) {
                      return value.substring(0, 27) + '...'
                    }
                    return value
                  }}
                />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                <Tooltip 
                  formatter={(value: any, name: string) => {
                    if (name === 'query_count') return [value, 'Queries']
                    if (name === 'avg_relevance_score') return [`${(value * 100).toFixed(1)}%`, 'Avg Relevance']
                    if (name === 'positive_feedback_count') return [value, '👍 Positive']
                    if (name === 'negative_feedback_count') return [value, '👎 Negative']
                    return [value, name]
                  }}
                  labelFormatter={(label, payload) => {
                    if (payload && payload[0]) {
                      const data = payload[0].payload as DocumentQueryStats
                      return data.filename
                    }
                    return label
                  }}
                />
                <Legend />
                <Bar 
                  yAxisId="left"
                  dataKey="query_count" 
                  fill="#3b82f6" 
                  name="Query Count"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  yAxisId="right"
                  dataKey="avg_relevance_score" 
                  fill="#10b981" 
                  name="Avg Relevance"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  yAxisId="left"
                  dataKey="positive_feedback_count" 
                  fill="#22c55e" 
                  name="👍 Positive"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  yAxisId="left"
                  dataKey="negative_feedback_count" 
                  fill="#ef4444" 
                  name="👎 Negative"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
            
            {/* Document summary cards */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Most Queried Document</h3>
                <p className="text-lg font-bold text-gray-900 truncate" title={documentQueryStats[0]?.filename}>
                  {documentQueryStats[0]?.filename || 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {documentQueryStats[0]?.query_count || 0} queries
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Total Documents Queried</h3>
                <p className="text-2xl font-bold text-gray-900">{documentQueryStats.length}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Documents with query activity
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Total Queries</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {documentQueryStats.reduce((sum: number, d: DocumentQueryStats) => sum + d.query_count, 0)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Across all documents
                </p>
              </div>
            </div>
            
            {/* Document list with details */}
            <div className="mt-6 space-y-2">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Document Details</h3>
              {documentQueryStats.slice(0, 10).map((doc: DocumentQueryStats) => (
                <div key={doc.document_id} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 truncate">{doc.filename}</p>
                    <div className="flex gap-2 mt-1">
                      {doc.core_area && (
                        <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-800 rounded">
                          {doc.core_area}
                        </span>
                      )}
                      {doc.factory && (
                        <span className="text-xs px-2 py-0.5 bg-green-100 text-green-800 rounded">
                          {doc.factory}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right text-sm text-gray-600">
                    <p>{doc.query_count} queries</p>
                    <p className="text-xs">
                      {(doc.avg_relevance_score * 100).toFixed(1)}% relevance
                    </p>
                    <p className="text-xs mt-1">
                      👍 {doc.positive_feedback_count} • 👎 {doc.negative_feedback_count}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No document query data yet. Make some queries to see which documents are accessed most frequently.
          </p>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-2">Queries by Key Terms</h2>
        <p className="text-sm text-gray-500 mb-4">
          Shows which keywords and terms appear most frequently in user queries
        </p>
        {keyTermStats && keyTermStats.length > 0 ? (
          <div>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={keyTermStats.slice(0, 20)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="term" 
                  angle={-45} 
                  textAnchor="end" 
                  height={120}
                  tick={{ fontSize: 12 }}
                />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip 
                  formatter={(value: any, name: string) => {
                    if (name === 'frequency') return [value, 'Total Occurrences']
                    if (name === 'unique_queries') return [value, 'Unique Queries']
                    if (name === 'avg_feedback') return value !== null ? [`${(value * 100).toFixed(1)}%`, 'Avg Feedback'] : ['N/A', 'Avg Feedback']
                    return [value, name]
                  }}
                />
                <Legend />
                <Bar 
                  yAxisId="left"
                  dataKey="frequency" 
                  fill="#3b82f6" 
                  name="Total Occurrences"
                  radius={[4, 4, 0, 0]}
                />
                <Bar 
                  yAxisId="left"
                  dataKey="unique_queries" 
                  fill="#10b981" 
                  name="Unique Queries"
                  radius={[4, 4, 0, 0]}
                />
                {keyTermStats.some((k: KeyTermStats) => k.avg_feedback !== null && k.avg_feedback !== undefined) && (
                  <Bar 
                    yAxisId="right"
                    dataKey="avg_feedback" 
                    fill="#f59e0b" 
                    name="Avg Feedback"
                    radius={[4, 4, 0, 0]}
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
            
            {/* Top terms summary */}
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Most Frequent Term</h3>
                <p className="text-2xl font-bold text-gray-900 capitalize">
                  {keyTermStats[0]?.term || 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {keyTermStats[0]?.frequency || 0} occurrences
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Total Unique Terms</h3>
                <p className="text-2xl font-bold text-gray-900">{keyTermStats.length}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Across all queries
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Total Term Occurrences</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {keyTermStats.reduce((sum: number, k: KeyTermStats) => sum + k.frequency, 0)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Combined frequency
                </p>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No key term data yet. Make some queries to see which terms are searched most frequently.
          </p>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Query Patterns (by Text)</h2>
        {queryPatterns && queryPatterns.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={queryPatterns.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="query_text" angle={-45} textAnchor="end" height={100} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="frequency" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500 text-center">No query data yet</p>
        )}
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Document Usage</h2>
        {documentUsage && documentUsage.length > 0 ? (
          <div className="space-y-2">
            {documentUsage.slice(0, 10).map((doc: any) => (
              <div key={doc.document_id} className="flex justify-between items-center p-3 border-b">
                <span className="font-medium">{doc.filename}</span>
                <span className="text-sm text-gray-500">
                  {doc.query_count} queries • {(doc.avg_relevance_score * 100).toFixed(1)}% relevance
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center">No document usage data yet</p>
        )}
      </div>
    </div>
  )
}

