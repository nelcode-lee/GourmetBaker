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
  }>>([])
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null)

  const { data: history } = useQuery({
    queryKey: ['chat-history'],
    queryFn: async () => {
      const response = await chatAPI.getHistory()
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
          groundednessScore: data.groundedness_score
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
              Send
            </button>
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
        </div>
      </form>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-lg">Ask a question about your documents</p>
            <p className="text-sm mt-2">Upload documents first to get started</p>
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
              <p className="whitespace-pre-wrap">{msg.content}</p>
              
              {/* Groundedness warning */}
              {msg.type === 'assistant' && msg.groundednessScore !== undefined && msg.groundednessScore < 0.5 && (
                <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded text-xs">
                  <p className="text-yellow-800 font-medium">
                    ⚠️ Low confidence: This response may not be fully supported by the source documents.
                  </p>
                </div>
              )}
              
              {/* Citations */}
              {msg.type === 'assistant' && msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 space-y-2">
                  <p className="text-xs font-medium text-gray-500">Sources:</p>
                  {msg.citations.map((citation, idx) => (
                    <div key={citation.chunk_id} className="text-xs">
                      <button
                        onClick={() => setExpandedCitation(
                          expandedCitation === citation.chunk_id ? null : citation.chunk_id
                        )}
                        className="text-blue-600 hover:text-blue-800 underline"
                      >
                        [{citation.rank}] {citation.content.substring(0, 60)}...
                        {expandedCitation === citation.chunk_id ? ' (hide)' : ' (show)'}
                      </button>
                      {expandedCitation === citation.chunk_id && (
                        <div className="mt-2 p-2 bg-gray-50 rounded text-xs text-gray-700 border-l-2 border-blue-500">
                          <p className="font-medium mb-1">Source {citation.rank}:</p>
                          <p className="whitespace-pre-wrap">{citation.content}</p>
                          <p className="mt-1 text-gray-500">
                            Relevance: {(citation.relevance_score * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
              
              {/* Feedback buttons */}
              {msg.type === 'assistant' && msg.queryId && (
                <div className="mt-4 flex items-center gap-3 border-t border-gray-200 pt-3">
                  <span className="text-xs text-gray-500 font-medium">Was this helpful?</span>
                  <button
                    onClick={() => chatAPI.submitFeedback(msg.queryId!, 1)}
                    className="px-3 py-1.5 text-lg hover:bg-green-50 rounded-lg transition-colors border border-gray-200 hover:border-green-300"
                    title="Thumbs up - This response was helpful"
                  >
                    👍
                  </button>
                  <button
                    onClick={() => chatAPI.submitFeedback(msg.queryId!, -1)}
                    className="px-3 py-1.5 text-lg hover:bg-red-50 rounded-lg transition-colors border border-gray-200 hover:border-red-300"
                    title="Thumbs down - This response was not helpful"
                  >
                    👎
                  </button>
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

