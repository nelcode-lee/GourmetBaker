import { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ChatInterface from './components/chat/ChatInterface'
import DocumentManager from './components/documents/DocumentManager'
import AnalyticsDashboard from './components/analytics/AnalyticsDashboard'

const queryClient = new QueryClient()

function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'documents' | 'analytics'>('chat')

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white shadow-sm border-b">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              <h1 className="text-2xl font-bold text-gray-900">
                Cranswick Technical Standards Agent
              </h1>
              <nav className="flex space-x-4">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`px-4 py-2 rounded-md ${
                    activeTab === 'chat'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Chat
                </button>
                <button
                  onClick={() => setActiveTab('documents')}
                  className={`px-4 py-2 rounded-md ${
                    activeTab === 'documents'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Documents
                </button>
                <button
                  onClick={() => setActiveTab('analytics')}
                  className={`px-4 py-2 rounded-md ${
                    activeTab === 'analytics'
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  Analytics
                </button>
              </nav>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex justify-center mb-8">
            <img 
              src="/cranswicklogo.png" 
              alt="Cranswick Logo" 
              className="h-24 w-auto object-contain"
            />
          </div>
          {activeTab === 'chat' && <ChatInterface />}
          {activeTab === 'documents' && <DocumentManager />}
          {activeTab === 'analytics' && <AnalyticsDashboard />}
        </main>
      </div>
    </QueryClientProvider>
  )
}

export default App

