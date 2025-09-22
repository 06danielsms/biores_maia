import { http, HttpResponse } from 'msw'
import {
  alignmentFindingsBySummaryId,
  corpusDocuments,
  datasetState,
  evaluationOverview,
  health,
  jobsCosts,
  jobsStatus,
  ragMetrics,
  summaryJobsById,
  translationMetricsByDocumentId,
} from './data.js'

const asJson = (data, init) => HttpResponse.json(data, init)

export const handlers = [
  http.get('*/health/', () => asJson(health)),
  http.get('*/datasets/state', () => asJson(datasetState)),
  http.get('*/jobs/status', () => asJson(jobsStatus)),
  http.get('*/jobs/costs', () => asJson(jobsCosts)),
  http.get('*/evaluation/overview', () => asJson(evaluationOverview)),
  http.get('*/evaluation/rag/metrics', () => asJson(ragMetrics)),
  http.get('*/evaluation/alignment/:summaryId', ({ params }) => {
    const data = alignmentFindingsBySummaryId[params.summaryId]
    if (!data) {
      return asJson({ detail: 'Summary not found' }, { status: 404 })
    }
    return asJson(data)
  }),
  http.get('*/corpus/documents', ({ request }) => {
    const url = new URL(request.url)
    const source = url.searchParams.get('source')
    const status = url.searchParams.get('status')
    const metrics = url.searchParams.get('metrics')

    const items = corpusDocuments.filter((doc) => {
      const matchesSource = !source || doc.source === source
      const matchesStatus = !status || doc.status === status
      const matchesMetrics = !metrics || doc.metricsStatus === metrics
      return matchesSource && matchesStatus && matchesMetrics
    })

    return asJson({ items, total: items.length })
  }),
  http.get('*/corpus/:documentId', ({ params }) => {
    const document = corpusDocuments.find((doc) => doc.id === params.documentId)
    if (!document) {
      return asJson({ detail: 'Documento no encontrado' }, { status: 404 })
    }

    return asJson({
      id: document.id,
      source: document.source,
      domain: document.domain,
      tokens: document.tokens,
      updated_at: document.updatedAt,
      metrics: document.metrics,
      comments: document.comments,
      original: document.originalContent,
      translation: document.translatedContent,
    })
  }),
  http.get('*/translation/metrics/:documentId', ({ params }) => {
    const data =
      translationMetricsByDocumentId[params.documentId] ??
      translationMetricsByDocumentId['CT-2024-0001']
    return asJson(data)
  }),
  http.post('*/translation/recompute', async ({ request }) => {
    const payload = await request.json()
    return asJson({
      job_id: `translation-recompute-${Date.now()}`,
      status: 'queued',
      received: payload,
    })
  }),
  http.get('*/summaries/jobs/:jobId', ({ params }) => {
    const data = summaryJobsById[params.jobId]
    if (!data) {
      return asJson({ detail: 'Job no encontrado' }, { status: 404 })
    }
    return asJson(data)
  }),
  http.post('*/summaries/jobs', async ({ request }) => {
    const payload = await request.json()
    const jobId = `summary-job-${Date.now()}`
    summaryJobsById[jobId] = {
      job_id: jobId,
      status: 'queued',
      started_at: new Date().toISOString(),
      model: payload?.model ?? 'ollama-phi3-mini',
      progress: 0,
      metrics: null,
    }
    return asJson({ job_id: jobId, status: 'queued' }, { status: 201 })
  }),
]
