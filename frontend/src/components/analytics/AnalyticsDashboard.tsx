import { useQuery } from '@tanstack/react-query'
import { analyticsAPI } from '../../services/api'
import { KeyTermStats, QualityMetrics } from '../../types'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell } from 'recharts'

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

  const { data: keyTermStats, isLoading: keyTermsLoading } = useQuery({
    queryKey: ['analytics-key-terms'],
    queryFn: async () => {
      const response = await analyticsAPI.getQueriesByKeyTerms()
      return response.data
    },
  })

  const { data: qualityMetrics, isLoading: qualityLoading, error: qualityError } = useQuery({
    queryKey: ['analytics-quality-metrics'],
    queryFn: async () => {
      const response = await analyticsAPI.getQualityMetrics()
      return response.data as QualityMetrics
    },
    retry: 2,
    staleTime: 30000, // Cache for 30 seconds
  })

  // Show partial results - don't block on all queries
  const isLoading = overviewLoading || patternsLoading || usageLoading || keyTermsLoading || qualityLoading

  // Prepare data for donut charts
  const COLORS = {
    positive: '#22c55e',
    negative: '#ef4444',
    neutral: '#94a3b8',
    high: '#10b981',
    medium: '#f59e0b',
    low: '#ef4444',
    ready: '#10b981',
    processing: '#3b82f6',
    failed: '#ef4444'
  }

  const feedbackData = qualityMetrics ? [
    { name: 'Positive', value: qualityMetrics.feedback_distribution.positive || 0, color: COLORS.positive },
    { name: 'Negative', value: qualityMetrics.feedback_distribution.negative || 0, color: COLORS.negative },
    { name: 'Neutral', value: qualityMetrics.feedback_distribution.neutral || 0, color: COLORS.neutral }
  ].filter(item => item.value > 0) : []

  const confidenceData = qualityMetrics ? [
    { name: 'High (≥70%)', value: qualityMetrics.confidence_distribution.high || 0, color: COLORS.high },
    { name: 'Medium (40-70%)', value: qualityMetrics.confidence_distribution.medium || 0, color: COLORS.medium },
    { name: 'Low (<40%)', value: qualityMetrics.confidence_distribution.low || 0, color: COLORS.low }
  ].filter(item => item.value > 0) : []

  const relevanceData = qualityMetrics ? [
    { name: 'High (≥70%)', value: qualityMetrics.relevance_distribution.high || 0, color: COLORS.high },
    { name: 'Medium (40-70%)', value: qualityMetrics.relevance_distribution.medium || 0, color: COLORS.medium },
    { name: 'Low (<40%)', value: qualityMetrics.relevance_distribution.low || 0, color: COLORS.low }
  ].filter(item => item.value > 0) : []

  const documentStatusData = qualityMetrics ? Object.entries(qualityMetrics.document_status_distribution).map(([status, count]) => ({
    name: status.charAt(0).toUpperCase() + status.slice(1),
    value: count,
    color: COLORS[status as keyof typeof COLORS] || '#6b7280'
  })).filter(item => item.value > 0) : []

  const renderLabel = (entry: any, data: any[]) => {
    const total = data.reduce((sum: number, item: any) => sum + item.value, 0)
    if (total === 0) return ''
    const percent = ((entry.value / total) * 100).toFixed(1)
    return `${percent}%`
  }

  // Calculate training quality color and label
  const getTrainingQualityInfo = (score: number) => {
    if (score >= 80) return { color: 'text-green-600', bgColor: 'bg-green-50', borderColor: 'border-green-200', label: 'Excellent', emoji: '🌟' }
    if (score >= 60) return { color: 'text-blue-600', bgColor: 'bg-blue-50', borderColor: 'border-blue-200', label: 'Good', emoji: '✅' }
    if (score >= 40) return { color: 'text-yellow-600', bgColor: 'bg-yellow-50', borderColor: 'border-yellow-200', label: 'Fair', emoji: '⚠️' }
    return { color: 'text-red-600', bgColor: 'bg-red-50', borderColor: 'border-red-200', label: 'Needs Improvement', emoji: '📈' }
  }

  const trainingQuality = qualityMetrics ? getTrainingQualityInfo(qualityMetrics.training_quality_score) : null

  return (
    <div className="space-y-6">
      {/* Training Quality Score - Prominent Display */}
      {qualityMetrics && (
        <div className={`bg-white rounded-lg shadow-lg p-6 border-2 ${trainingQuality?.borderColor || 'border-gray-200'}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-gray-800">Agent Training Quality</h2>
              <p className="text-sm text-gray-500 mt-1">Composite score based on feedback, confidence, relevance, and success rate</p>
            </div>
            <div className={`text-4xl ${trainingQuality?.color || 'text-gray-400'}`}>
              {trainingQuality?.emoji || '📊'}
            </div>
          </div>
          
          <div className="flex items-end gap-6">
            <div className="flex-1">
              <div className="flex items-baseline gap-2 mb-2">
                <span className={`text-5xl font-bold ${trainingQuality?.color || 'text-gray-400'}`}>
                  {qualityMetrics.training_quality_score.toFixed(1)}%
                </span>
                <span className={`text-xl font-semibold ${trainingQuality?.color || 'text-gray-400'}`}>
                  {trainingQuality?.label || 'N/A'}
                </span>
              </div>
              
              {/* Progress bar */}
              <div className="w-full bg-gray-200 rounded-full h-4 mb-4">
                <div
                  className={`h-4 rounded-full transition-all duration-500 ${
                    qualityMetrics.training_quality_score >= 80 ? 'bg-green-500' :
                    qualityMetrics.training_quality_score >= 60 ? 'bg-blue-500' :
                    qualityMetrics.training_quality_score >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${Math.min(100, qualityMetrics.training_quality_score)}%` }}
                />
              </div>
              
              {/* Score breakdown */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 text-sm">
                <div>
                  <div className="text-gray-500">Positive Feedback</div>
                  <div className="font-semibold text-gray-800">
                    {qualityMetrics.feedback_distribution.positive > 0 || qualityMetrics.feedback_distribution.negative > 0
                      ? ((qualityMetrics.feedback_distribution.positive / 
                          (qualityMetrics.feedback_distribution.positive + qualityMetrics.feedback_distribution.negative)) * 100).toFixed(1)
                      : '0'}%
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Avg Confidence</div>
                  <div className="font-semibold text-gray-800">
                    {(qualityMetrics.avg_confidence_score * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Avg Relevance</div>
                  <div className="font-semibold text-gray-800">
                    {(qualityMetrics.avg_relevance_score * 100).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-gray-500">Success Rate</div>
                  <div className="font-semibold text-gray-800">
                    {overview && overview.total_queries > 0
                      ? ((qualityMetrics.feedback_distribution.positive / overview.total_queries) * 100).toFixed(1)
                      : '0'}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quality Metrics Section with Donut Charts */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Quality Metrics</h2>
        {qualityLoading && (
          <div className="text-center text-gray-500 py-8">Loading quality metrics...</div>
        )}
        {qualityError && (
          <div className="text-center text-red-500 py-8">
            Error loading quality metrics. Please try refreshing the page.
          </div>
        )}
        {!qualityLoading && !qualityError && qualityMetrics && (
        <>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Feedback Distribution */}
          <div className="flex flex-col items-center">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Feedback Distribution</h3>
            {feedbackData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={feedbackData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                      label={(entry: any) => {
                        const total = feedbackData.reduce((sum: number, item: any) => sum + item.value, 0)
                        if (total === 0) return ''
                        const percent = ((entry.value / total) * 100).toFixed(1)
                        return `${percent}%`
                      }}
                    >
                      {feedbackData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => [value, 'Responses']} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-500 mt-2">
                  Total: {qualityMetrics?.total_with_feedback || 0} responses
                </p>
              </>
            ) : (
              <div className="flex items-center justify-center h-[200px] text-gray-400 text-sm">
                No feedback data
              </div>
            )}
          </div>

          {/* Confidence Distribution */}
          <div className="flex flex-col items-center">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Confidence Levels</h3>
            {confidenceData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={confidenceData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                      label={(entry: any) => {
                        const total = confidenceData.reduce((sum: number, item: any) => sum + item.value, 0)
                        if (total === 0) return ''
                        const percent = ((entry.value / total) * 100).toFixed(1)
                        return `${percent}%`
                      }}
                    >
                      {confidenceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => [value, 'Queries']} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-500 mt-2">
                  Avg: {(qualityMetrics?.avg_confidence_score || 0) * 100}%
                </p>
              </>
            ) : (
              <div className="flex items-center justify-center h-[200px] text-gray-400 text-sm">
                No confidence data
              </div>
            )}
          </div>

          {/* Relevance Distribution */}
          <div className="flex flex-col items-center">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Relevance Scores</h3>
            {relevanceData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={relevanceData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                      label={(entry: any) => {
                        const total = relevanceData.reduce((sum: number, item: any) => sum + item.value, 0)
                        if (total === 0) return ''
                        const percent = ((entry.value / total) * 100).toFixed(1)
                        return `${percent}%`
                      }}
                    >
                      {relevanceData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => [value, 'Citations']} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-500 mt-2">
                  Avg: {(qualityMetrics?.avg_relevance_score || 0) * 100}%
                </p>
              </>
            ) : (
              <div className="flex items-center justify-center h-[200px] text-gray-400 text-sm">
                No relevance data
              </div>
            )}
          </div>

          {/* Document Status Distribution */}
          <div className="flex flex-col items-center">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Document Status</h3>
            {documentStatusData.length > 0 ? (
              <>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={documentStatusData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={80}
                      paddingAngle={2}
                      dataKey="value"
                      label={(entry: any) => {
                        const total = documentStatusData.reduce((sum: number, item: any) => sum + item.value, 0)
                        if (total === 0) return ''
                        const percent = ((entry.value / total) * 100).toFixed(1)
                        return `${percent}%`
                      }}
                    >
                      {documentStatusData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value: number) => [value, 'Documents']} />
                  </PieChart>
                </ResponsiveContainer>
                <p className="text-xs text-gray-500 mt-2">
                  Total: {overview?.total_documents || 0} documents
                </p>
              </>
            ) : (
              <div className="flex items-center justify-center h-[200px] text-gray-400 text-sm">
                No document data
              </div>
            )}
          </div>
        </div>

        {/* Quality Score Summary */}
        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 pt-6 border-t border-gray-200">
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-blue-700 mb-1">Average Confidence</h3>
            <p className="text-2xl font-bold text-blue-900">
              {qualityMetrics ? (qualityMetrics.avg_confidence_score * 100).toFixed(1) : '0.0'}%
            </p>
            <p className="text-xs text-blue-600 mt-1">
              Based on {qualityMetrics?.total_with_confidence || 0} queries
            </p>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-green-700 mb-1">Average Relevance</h3>
            <p className="text-2xl font-bold text-green-900">
              {qualityMetrics ? (qualityMetrics.avg_relevance_score * 100).toFixed(1) : '0.0'}%
            </p>
            <p className="text-xs text-green-600 mt-1">
              Across all citations
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <h3 className="text-sm font-medium text-purple-700 mb-1">Average Groundedness</h3>
            <p className="text-2xl font-bold text-purple-900">
              {qualityMetrics ? (qualityMetrics.avg_groundedness_score * 100).toFixed(1) : '0.0'}%
            </p>
            <p className="text-xs text-purple-600 mt-1">
              Response quality score
            </p>
          </div>
        </div>
        </>
        )}
      </div>

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

