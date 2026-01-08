import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { chatAPI } from '../../services/api'
import { QueryResponse } from '../../types'

const CORE_AREAS = [
  'BRCGS',
  'Tesco',
  'Marks and Spencer',
  'Asda',
  'Hello Fresh',
  'Morrisons'
]

const FACTORIES = [
  'Gourmet Pastry',
  'Gourmet Kitchen',
  'Gourmet Sausage',
  'Convenience foods',
  'Fresh Poultry',
  'Prepared Poultry',
  'Gourmet Bacon',
  'Continental Foods'
]

export default function ChatInterface() {
  const [query, setQuery] = useState('')
  const [selectedCoreArea, setSelectedCoreArea] = useState<string>('')
  const [selectedFactory, setSelectedFactory] = useState<string>('')
  const [messages, setMessages] = useState<Array<{ 
    type: 'user' | 'assistant'; 
    content: string; 
    queryId?: string;
    citations?: Array<{ chunk_id: string; content: string; relevance_score: number; rank: number }>;
    groundednessScore?: number;
    avgRelevanceScore?: number;
    overallConfidence?: number;
  }>>([])
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null)
  const [feedbackMessages, setFeedbackMessages] = useState<Record<string, string>>({})

  const { data: history, refetch: refetchHistory } = useQuery({
    queryKey: ['chat-history'],
    queryFn: async () => {
      const response = await chatAPI.getHistory(5) // Get top 5 queries
      return response.data
    },
  })

  const mutation = useMutation({
    mutationFn: async (queryText: string) => {
      const response = await chatAPI.sendQuery(queryText, {
        coreArea: selectedCoreArea || undefined,
        factory: selectedFactory || undefined
      })
      return response.data as QueryResponse
    },
    onSuccess: (data) => {
      setMessages(prev => [
        ...prev,
        { 
          type: 'assistant', 
          content: data.response_text, 
          queryId: data.query_id,
          citations: data.citations || [],
          groundednessScore: data.groundedness_score,
          avgRelevanceScore: data.avg_relevance_score,
          overallConfidence: data.overall_confidence
        }
      ])
      // Refresh history to show the new query in recent queries
      refetchHistory()
    },
    onError: (error: any) => {
      console.error('Query error:', error)
      setMessages(prev => [
        ...prev,
        { 
          type: 'assistant', 
          content: `Error: ${error.response?.data?.detail || error.message || 'Failed to process query. Please try again.'}` 
        }
      ])
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setMessages(prev => [...prev, { type: 'user', content: query }])
    mutation.mutate(query)
    setQuery('')
  }

  return (
    <div className="flex flex-col h-[calc(100vh-12rem)]">
      {/* Input box at top for better visibility */}
      <form onSubmit={handleSubmit} className="border-b border-gray-200 bg-white p-4 sticky top-0 z-10 shadow-sm mb-4">
        <div className="max-w-4xl mx-auto space-y-3">
          {/* Filter dropdowns */}
          <div className="flex gap-3">
            <select
              value={selectedCoreArea}
              onChange={(e) => setSelectedCoreArea(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm bg-white"
            >
              <option value="">All Core Areas</option>
              {CORE_AREAS.map(area => (
                <option key={area} value={area}>{area}</option>
              ))}
            </select>
            <select
              value={selectedFactory}
              onChange={(e) => setSelectedFactory(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm bg-white"
            >
              <option value="">All Factories</option>
              {FACTORIES.map(factory => (
                <option key={factory} value={factory}>{factory}</option>
              ))}
            </select>
          </div>
          
          {/* Query input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
              disabled={mutation.isPending}
            />
            <button
              type="submit"
              disabled={mutation.isPending || !query.trim()}
              className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-base shadow-md"
            >
              {mutation.isPending ? 'Processing...' : 'Send'}
            </button>
            {mutation.isPending && (
              <button
                type="button"
                onClick={() => {
                  mutation.reset()
                  setMessages(prev => {
                    // Remove the last user message if it's still waiting for a response
                    const lastMessage = prev[prev.length - 1]
                    if (lastMessage && lastMessage.type === 'user') {
                      return prev.slice(0, -1)
                    }
                    return prev
                  })
                }}
                className="px-4 py-2 text-sm text-red-600 hover:text-red-700 underline"
              >
                Cancel
              </button>
            )}
          </div>
          
          {/* Active filters display */}
          {(selectedCoreArea || selectedFactory) && (
            <div className="flex gap-2 items-center text-sm text-gray-600">
              <span className="font-medium">Filtering by:</span>
              {selectedCoreArea && (
                <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded">
                  {selectedCoreArea}
                </span>
              )}
              {selectedFactory && (
                <span className="px-2 py-1 bg-green-100 text-green-800 rounded">
                  {selectedFactory}
                </span>
              )}
              <button
                onClick={() => {
                  setSelectedCoreArea('')
                  setSelectedFactory('')
                }}
                className="text-gray-500 hover:text-gray-700 underline"
              >
                Clear filters
              </button>
            </div>
          )}
          
          {/* Recent Queries - Top 5 */}
          {history && history.length > 0 && (
            <div className="pt-2 border-t border-gray-200">
              <p className="text-xs font-medium text-gray-500 mb-2">Recent Queries:</p>
              <div className="flex flex-wrap gap-2">
                {history.slice(0, 5).map((item) => (
                  <button
                    key={item.id}
                    onClick={() => {
                      const queryText = item.query_text
                      setQuery(queryText)
                      // Add user message and submit
                      setMessages(prev => [...prev, { type: 'user', content: queryText }])
                      mutation.mutate(queryText)
                    }}
                    className="px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg border border-gray-300 hover:border-gray-400 transition-colors truncate max-w-xs"
                    title={item.query_text}
                    disabled={mutation.isPending}
                  >
                    {item.query_text.length > 40 
                      ? `${item.query_text.substring(0, 40)}...` 
                      : item.query_text}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </form>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !mutation.isPending && (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-lg">Ask a question about your documents</p>
            <p className="text-sm mt-2">Upload documents first to get started</p>
          </div>
        )}
        {mutation.isPending && (
          <div className="flex justify-center items-center py-8">
            <div className="flex flex-col items-center gap-2">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <p className="text-sm text-gray-500">Processing your query...</p>
            </div>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-2xl rounded-lg p-4 ${
                msg.type === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white border border-gray-200'
              }`}
            >
              {/* Confidence Score Badge - Always visible for assistant messages */}
              {msg.type === 'assistant' && msg.overallConfidence !== undefined && (
                <div className="mb-3 pb-3 border-b border-gray-200">
                  <div className="flex items-center gap-3 flex-wrap">
                    {/* Overall Confidence Score */}
                    {(() => {
                      const confidenceLevel = msg.overallConfidence >= 0.65 
                        ? { label: 'High', icon: '✓', color: 'bg-green-100 text-green-800 border-green-300' }
                        : msg.overallConfidence >= 0.4
                        ? { label: 'Medium', icon: '⚠', color: 'bg-yellow-100 text-yellow-800 border-yellow-300' }
                        : { label: 'Low', icon: '✗', color: 'bg-red-100 text-red-800 border-red-300' }
                      
                      return (
                        <div className={`px-3 py-1.5 rounded-lg font-semibold text-sm flex items-center gap-2 ${confidenceLevel.color}`}>
                          <span className="text-base">{confidenceLevel.icon}</span>
                          <span>Confidence: {confidenceLevel.label}</span>
                        </div>
                      )
                    })()}
                    
                    {/* Average Relevance Score */}
                    {msg.avgRelevanceScore !== undefined && (
                      <div className="px-3 py-1.5 rounded-lg font-medium text-xs bg-blue-50 text-blue-700 border border-blue-200">
                        Avg Relevance: {(msg.avgRelevanceScore * 100).toFixed(0)}%
                      </div>
                    )}
                    
                    {/* Groundedness Score */}
                    {msg.groundednessScore !== undefined && (
                      <div className="px-3 py-1.5 rounded-lg font-medium text-xs bg-gray-50 text-gray-700 border border-gray-200">
                        Groundedness: {(msg.groundednessScore * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                  
                  {/* Warning for low confidence */}
                  {msg.overallConfidence < 0.4 && (
                    <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                      <p className="text-yellow-800 font-medium">
                        ⚠️ Low confidence: This response may not be fully supported by the source documents.
                      </p>
                    </div>
                  )}
                </div>
              )}
              
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {/* Citations */}
              {msg.type === 'assistant' && msg.citations && msg.citations.length > 0 && (
                <div className="mt-4 space-y-2 border-t border-gray-200 pt-3">
                  <p className="text-sm font-semibold text-gray-700 mb-2">📚 Sources ({msg.citations.length}):</p>
                  {msg.citations.map((citation, idx) => {
                    const relevancePercent = citation.relevance_score * 100
                    const relevanceColor = relevancePercent >= 70 
                      ? 'bg-green-100 text-green-800 border-green-300' 
                      : relevancePercent >= 50
                      ? 'bg-yellow-100 text-yellow-800 border-yellow-300'
                      : 'bg-red-100 text-red-800 border-red-300'
                    
                    return (
                      <div key={citation.chunk_id} className="text-sm border border-gray-200 rounded-lg p-2 hover:bg-gray-50 transition-colors">
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <button
                            onClick={() => setExpandedCitation(
                              expandedCitation === citation.chunk_id ? null : citation.chunk_id
                            )}
                            className="text-blue-600 hover:text-blue-800 font-medium text-left flex-1"
                          >
                            <span className="font-semibold">[{citation.rank}]</span> {citation.content.substring(0, 80)}...
                            <span className="text-xs text-gray-500 ml-2">
                              {expandedCitation === citation.chunk_id ? '(hide)' : '(show more)'}
                            </span>
                          </button>
                          <div className={`px-2 py-1 rounded font-semibold text-xs border ${relevanceColor} whitespace-nowrap`}>
                            {relevancePercent.toFixed(0)}% match
                          </div>
                        </div>
                        {expandedCitation === citation.chunk_id && (
                          <div className="mt-2 p-3 bg-gray-50 rounded text-sm text-gray-700 border-l-4 border-blue-500">
                            <p className="font-semibold mb-2 text-gray-900">Source {citation.rank} - Full Content:</p>
                            <p className="whitespace-pre-wrap mb-2">{citation.content}</p>
                            <div className="flex items-center gap-2 text-xs">
                              <span className="font-medium text-gray-600">Relevance Score:</span>
                              <span className={`px-2 py-1 rounded font-semibold border ${relevanceColor}`}>
                                {(citation.relevance_score * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
              
              {/* Feedback buttons */}
              {msg.type === 'assistant' && msg.queryId && (
                <div className="mt-4 border-t border-gray-200 pt-3">
                  {feedbackMessages[msg.queryId] ? (
                    <div className="flex items-center gap-2 text-sm text-green-600 font-medium">
                      <span>✓</span>
                      <span>{feedbackMessages[msg.queryId]}</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-gray-500 font-medium">Was this helpful?</span>
                      <button
                        onClick={async () => {
                          try {
                            const response = await chatAPI.submitFeedback(msg.queryId!, 1)
                            setFeedbackMessages(prev => ({
                              ...prev,
                              [msg.queryId!]: response.data.message || "Thank you for your feedback!"
                            }))
                          } catch (error) {
                            console.error('Failed to submit feedback:', error)
                          }
                        }}
                        className="px-3 py-1.5 text-lg hover:bg-green-50 rounded-lg transition-colors border border-gray-200 hover:border-green-300"
                        title="Thumbs up - This response was helpful"
                      >
                        👍
                      </button>
                      <button
                        onClick={async () => {
                          try {
                            const response = await chatAPI.submitFeedback(msg.queryId!, -1)
                            setFeedbackMessages(prev => ({
                              ...prev,
                              [msg.queryId!]: response.data.message || "Thank you for your feedback!"
                            }))
                          } catch (error) {
                            console.error('Failed to submit feedback:', error)
                          }
                        }}
                        className="px-3 py-1.5 text-lg hover:bg-red-50 rounded-lg transition-colors border border-gray-200 hover:border-red-300"
                        title="Thumbs down - This response was not helpful"
                      >
                        👎
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {mutation.isPending && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <p className="text-gray-500">Thinking...</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

