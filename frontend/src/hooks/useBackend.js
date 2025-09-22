import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiClient } from '@/services/apiClient.js'

const normaliseCorpusDocument = (raw = {}) => {
  if (!raw || typeof raw !== 'object') return raw

  const fileName = raw.file_name ?? raw.fileName ?? raw.id
  const readability = raw.readability_fkgl ?? raw.readability ?? null
  const alignmentRisk = raw.alignment_risk ?? raw.alignmentRisk ?? 'pendiente'
  const updatedAt = raw.updated_at ?? raw.updatedAt ?? null

  return {
    id: raw.id ?? fileName,
    fileName,
    source: raw.source ?? 'Desconocido',
    language: raw.language ?? 'en',
    readability,
    translated: raw.translated ?? false,
    metricsReady: raw.metrics_ready ?? raw.metricsReady ?? false,
    tokens: raw.tokens ?? null,
    domain: raw.domain ?? null,
    alignmentRisk,
    updatedAt,
    originalContent: raw.original ?? raw.originalContent ?? null,
    translatedContent: raw.translation ?? raw.translatedContent ?? null,
    metrics: raw.metrics ?? null,
    comments: Array.isArray(raw.comments) ? raw.comments : [],
  }
}

const normaliseCorpusDetail = (raw = {}) => {
  if (!raw || typeof raw !== 'object') {
    return {
      id: raw?.id,
      source: 'Desconocido',
      language: 'en',
      original: null,
      translation: null,
      metrics: {},
      comments: [],
      tokens: null,
      updated_at: null,
      updatedAt: null,
    }
  }

  const updatedAt = raw.updated_at ?? raw.updatedAt ?? null

  return {
    id: raw.id,
    source: raw.source ?? 'Desconocido',
    language: raw.language ?? 'en',
    original: raw.original ?? raw.originalContent ?? null,
    translation: raw.translation ?? raw.translatedContent ?? null,
    metrics: raw.metrics ?? {},
    comments: Array.isArray(raw.comments) ? raw.comments : [],
    tokens: raw.tokens ?? null,
    updated_at: updatedAt,
    updatedAt,
  }
}

export const useDatasetState = () =>
  useQuery({
    queryKey: ['datasets', 'state'],
    queryFn: async () => {
      const { data } = await apiClient.get('/datasets/state')
      return data
    },
  })

export const useCorpusDocuments = (filters) =>
  useQuery({
    queryKey: ['corpus', 'documents', filters],
    queryFn: async () => {
      const { data } = await apiClient.get('/corpus/documents', { params: filters })
      return {
        ...data,
        items: Array.isArray(data?.items) ? data.items.map(normaliseCorpusDocument) : [],
      }
    },
  })

export const useCorpusDocument = (documentId, enabled) =>
  useQuery({
    queryKey: ['corpus', 'document', documentId],
    enabled: Boolean(documentId) && enabled,
    queryFn: async () => {
      const { data } = await apiClient.get(`/corpus/${documentId}`)
      return normaliseCorpusDetail(data)
    },
  })

export const useTranslationMetrics = (documentId, enabled = Boolean(documentId)) =>
  useQuery({
    queryKey: ['translation', 'metrics', documentId],
    enabled,
    queryFn: async () => {
      const { data } = await apiClient.get(`/translation/metrics/${documentId}`)
      return data
    },
  })

export const useRecomputeTranslation = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (payload) => {
      const { data } = await apiClient.post('/translation/recompute', payload)
      return data
    },
    onSuccess: (_, variables) => {
      if (variables?.document_id) {
        queryClient.invalidateQueries({ queryKey: ['translation', 'metrics', variables.document_id] })
      }
    },
  })
}

export const useSummaryJob = (jobId) =>
  useQuery({
    queryKey: ['summaries', 'jobs', jobId],
    queryFn: async () => {
      const { data } = await apiClient.get(`/summaries/jobs/${jobId}`)
      return data
    },
  })

export const useCreateSummaryJob = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: async (payload) => {
      const { data } = await apiClient.post('/summaries/jobs', payload)
      return data
    },
    onSuccess: (data) => {
      if (data?.job_id) {
        queryClient.invalidateQueries({ queryKey: ['summaries', 'jobs', data.job_id] })
      }
    },
  })
}

export const useEvaluationOverview = () =>
  useQuery({
    queryKey: ['evaluation', 'overview'],
    queryFn: async () => {
      const { data } = await apiClient.get('/evaluation/overview')
      return data
    },
  })

export const useAlignmentFindings = (summaryId) =>
  useQuery({
    queryKey: ['evaluation', 'alignment', summaryId],
    enabled: Boolean(summaryId),
    queryFn: async () => {
      const { data } = await apiClient.get(`/evaluation/alignment/${summaryId}`)
      return data
    },
  })

export const useRagMetrics = () =>
  useQuery({
    queryKey: ['evaluation', 'rag-metrics'],
    queryFn: async () => {
      const { data } = await apiClient.get('/evaluation/rag/metrics')
      return data
    },
  })

export const useJobsStatus = () =>
  useQuery({
    queryKey: ['jobs', 'status'],
    queryFn: async () => {
      const { data } = await apiClient.get('/jobs/status')
      return data
    },
  })

export const useJobsCosts = () =>
  useQuery({
    queryKey: ['jobs', 'costs'],
    queryFn: async () => {
      const { data } = await apiClient.get('/jobs/costs')
      return data
    },
  })
