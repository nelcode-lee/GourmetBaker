import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { documentAPI } from '../../services/api'
import { Document } from '../../types'

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

export default function DocumentManager() {
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadingFiles, setUploadingFiles] = useState<Set<string>>(new Set())
  const [editingTags, setEditingTags] = useState<string | null>(null)
  const [tagCoreArea, setTagCoreArea] = useState<string>('')
  const [tagFactory, setTagFactory] = useState<string>('')
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: async () => {
      const response = await documentAPI.list()
      return response.data
    },
    refetchInterval: (query) => {
      // Auto-refresh every 2 seconds if any documents are processing
      const documents = query.state.data?.documents || []
      const hasProcessing = documents.some((doc: Document) => doc.status === 'processing')
      return hasProcessing ? 2000 : false
    },
  })

  const uploadMutation = useMutation({
    mutationFn: async (files: FileList) => {
      const fileNames = Array.from(files).map(f => f.name)
      setUploadingFiles(new Set(fileNames))
      setUploadProgress(0)
      
      try {
        const result = await documentAPI.upload(files, setUploadProgress)
        return result
      } finally {
        setUploadProgress(0)
        // Keep files in uploading state until processing completes
        setTimeout(() => {
          setUploadingFiles(new Set())
        }, 1000)
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      // Keep polling for status updates
    },
    onError: () => {
      setUploadProgress(0)
      setUploadingFiles(new Set())
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => {
      return documentAPI.delete(id)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const updateTagsMutation = useMutation({
    mutationFn: async ({ id, coreArea, factory }: { id: string; coreArea?: string; factory?: string }) => {
      return documentAPI.updateTags(id, { core_area: coreArea, factory })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      setEditingTags(null)
      setTagCoreArea('')
      setTagFactory('')
    },
  })

  const handleEditTags = (doc: Document) => {
    setEditingTags(doc.id)
    setTagCoreArea(doc.metadata?.core_area || '')
    setTagFactory(doc.metadata?.factory || '')
  }

  const handleSaveTags = (docId: string) => {
    updateTagsMutation.mutate({
      id: docId,
      coreArea: tagCoreArea || undefined,
      factory: tagFactory || undefined
    })
  }

  const handleCancelEdit = () => {
    setEditingTags(null)
    setTagCoreArea('')
    setTagFactory('')
  }

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadMutation.mutate(e.target.files)
    }
  }

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this document?')) {
      deleteMutation.mutate(id)
    }
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Upload Documents</h2>
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            multiple
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
            accept=".pdf,.docx,.txt,.md"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Choose Files
          </label>
          <p className="mt-2 text-sm text-gray-500">
            Supports PDF, DOCX, TXT, and Markdown files
          </p>
          {uploadProgress > 0 && uploadProgress < 100 && (
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-700 mb-2">Uploading... {uploadProgress}%</p>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}
          {uploadProgress === 100 && (
            <div className="mt-4">
              <p className="text-sm font-medium text-green-600">✓ Upload complete! Processing document...</p>
            </div>
          )}
          {uploadMutation.isPending && uploadProgress === 0 && (
            <div className="mt-4">
              <p className="text-sm font-medium text-blue-600">Preparing upload...</p>
            </div>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold">Documents</h2>
        </div>
        {isLoading ? (
          <div className="p-6 text-center text-gray-500">Loading...</div>
        ) : data?.documents?.length === 0 ? (
          <div className="p-6 text-center text-gray-500">No documents uploaded yet</div>
        ) : (
          <div className="divide-y divide-gray-200">
            {data?.documents?.map((doc: Document) => (
              <div key={doc.id} className="p-6">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="font-medium text-gray-900">{doc.filename}</h3>
                    <p className="text-sm text-gray-500">
                      {doc.file_type.toUpperCase()} • {(doc.file_size / 1024).toFixed(2)} KB
                    </p>
                    <div className="mt-2 flex items-center gap-2 flex-wrap">
                      <span
                        className={`inline-block px-2 py-1 text-xs rounded font-medium ${
                          doc.status === 'ready'
                            ? 'bg-green-100 text-green-800'
                            : doc.status === 'processing'
                            ? 'bg-yellow-100 text-yellow-800 animate-pulse'
                            : 'bg-red-100 text-red-800'
                        }`}
                      >
                        {doc.status === 'ready' && '✓ Ready'}
                        {doc.status === 'processing' && '⏳ Processing...'}
                        {doc.status === 'failed' && '✗ Failed'}
                      </span>
                      {doc.status === 'processing' && (
                        <div className="flex items-center gap-1 text-xs text-gray-500">
                          <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                          <span>Processing chunks...</span>
                        </div>
                      )}
                      {doc.metadata?.core_area && (
                        <span className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                          {doc.metadata.core_area}
                        </span>
                      )}
                      {doc.metadata?.factory && (
                        <span className="px-2 py-1 text-xs bg-green-100 text-green-800 rounded">
                          {doc.metadata.factory}
                        </span>
                      )}
                    </div>
                    
                    {/* Tag editing UI */}
                    {editingTags === doc.id ? (
                      <div className="mt-3 p-3 bg-gray-50 rounded-lg space-y-2">
                        <div className="flex gap-2">
                          <select
                            value={tagCoreArea}
                            onChange={(e) => setTagCoreArea(e.target.value)}
                            className="px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                          >
                            <option value="">Select Core Area</option>
                            {CORE_AREAS.map(area => (
                              <option key={area} value={area}>{area}</option>
                            ))}
                          </select>
                          <select
                            value={tagFactory}
                            onChange={(e) => setTagFactory(e.target.value)}
                            className="px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                          >
                            <option value="">Select Factory</option>
                            {FACTORIES.map(factory => (
                              <option key={factory} value={factory}>{factory}</option>
                            ))}
                          </select>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleSaveTags(doc.id)}
                            className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                            disabled={updateTagsMutation.isPending}
                          >
                            Save
                          </button>
                          <button
                            onClick={handleCancelEdit}
                            className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={() => handleEditTags(doc)}
                        className="mt-2 text-sm text-blue-600 hover:text-blue-800 underline"
                        disabled={doc.status !== 'ready'}
                      >
                        {doc.metadata?.core_area || doc.metadata?.factory ? 'Edit Tags' : 'Add Tags'}
                      </button>
                    )}
                  </div>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg ml-4"
                    disabled={deleteMutation.isPending}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

