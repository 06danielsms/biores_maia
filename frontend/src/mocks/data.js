export const health = {
  status: 'ok',
  version: '2024.11.0',
  timestamp: '2024-11-22T09:15:00Z',
}

export const datasetState = {
  datasets: [
    { name: 'ClinicalTrials.gov', version: '2024-11-21', status: 'synced' },
    { name: 'Cochrane Evidence Pack', version: '2024-11-20', status: 'resync_required' },
    { name: 'Pfizer Lote Q4', version: '2024-11-18', status: 'queued' },
  ],
}

export const corpusDocuments = [
  {
    id: 'CT-2024-0001',
    fileName: 'CT-2024-0001.json',
    source: 'ClinicalTrials.gov',
    language: 'en',
    readability: 46.2,
    translated: true,
    status: 'translated',
    metricsReady: true,
    metricsStatus: 'published',
    tokens: 1874,
    domain: 'Cardiología',
    updatedAt: '2024-11-21 22:14 UTC',
    alignmentRisk: 'bajo',
    originalContent:
      'Phase III randomized study evaluating the efficacy of beta blockers in reducing readmission for heart failure patients.',
    translatedContent:
      'Estudio fase III aleatorizado que evalúa la eficacia de los betabloqueadores para reducir las re-hospitalizaciones en pacientes con insuficiencia cardiaca.',
    metrics: {
      bleu: 54.9,
      chrf2: 0.721,
      fkgl: 6.3,
      bertscore: 0.89,
    },
    comments: [
      {
        author: 'María Ortega',
        role: 'Especialista clínica',
        content: 'Validar el término “re-hospitalizaciones” con guía cardiológica.',
        timestamp: 'Hace 4 horas',
      },
      {
        author: 'Luis Gómez',
        role: 'Analista de datos',
        content: 'BLEU recalculado tras despliegue Helsinki, valores OK.',
        timestamp: 'Hace 1 día',
      },
    ],
  },
  {
    id: 'COCH-2023-0812',
    fileName: 'COCH-2023-0812.json',
    source: 'Cochrane',
    language: 'en',
    readability: 32.5,
    translated: true,
    status: 'in_progress',
    metricsReady: false,
    metricsStatus: 'processing',
    tokens: 2140,
    domain: 'Oncología',
    updatedAt: '2024-11-19 17:02 UTC',
    alignmentRisk: 'medio',
    originalContent:
      'Systematic review summarizing adjuvant therapies for gastric cancer with focus on immunotherapy outcomes.',
    translatedContent:
      'Revisión sistemática que resume terapias adyuvantes para cáncer gástrico con foco en resultados de inmunoterapia.',
    metrics: {
      bleu: 38.1,
      chrf2: 0.654,
      fkgl: 8.1,
      bertscore: 0.82,
    },
    comments: [
      {
        author: 'Claudia Rivas',
        role: 'Especialista médica',
        content: 'Solicitar revisión manual: términos de inmunoterapia requieren glosario extendido.',
        timestamp: 'Hace 8 horas',
      },
    ],
  },
  {
    id: 'PFZ-2024-0032',
    fileName: 'PFZ-2024-0032.json',
    source: 'Pfizer',
    language: 'es',
    readability: 58.9,
    translated: false,
    status: 'pending',
    metricsReady: false,
    metricsStatus: 'processing',
    tokens: 980,
    domain: 'Vacunas',
    updatedAt: '2024-11-10 10:22 UTC',
    alignmentRisk: 'pendiente',
    originalContent:
      'Resumen técnico sobre avance de vacuna conjugada pediátrica. Falta homologar criterios de inclusión.',
    translatedContent:
      'Pendiente de traducción automática. Priorizar lote 2024-Q4 antes de marcar como listo.',
    metrics: {
      bleu: null,
      chrf2: null,
      fkgl: 5.9,
      bertscore: null,
    },
    comments: [
      {
        author: 'Valeria Ruiz',
        role: 'PM',
        content: 'Asignado al sprint de traducción 2024-Q4. Confirmar licencia antes de publicar.',
        timestamp: 'Hace 2 días',
      },
    ],
  },
]

export const translationMetricsByDocumentId = {
  'CT-2024-0001': {
    document_id: 'CT-2024-0001',
    metrics: {
      bleu: 54.9,
      chrf2: 0.721,
      fkgl: 6.3,
      length_ratio: 0.98,
      medical_terms: 38,
    },
  },
  'COCH-2023-0812': {
    document_id: 'COCH-2023-0812',
    metrics: {
      bleu: 38.1,
      chrf2: 0.654,
      fkgl: 8.1,
      length_ratio: 1.04,
      medical_terms: 44,
    },
  },
}

export const summaryJobsById = {
  'summary-job-4221': {
    job_id: 'summary-job-4221',
    status: 'streaming',
    started_at: '2024-11-22 09:47 UTC',
    model: 'ollama-phi3-mini',
    progress: 62,
    metrics: {
      readability_fkgl: 6.1,
      coherence: 0.84,
      alignscore: 0.73,
    },
  },
}

export const evaluationOverview = {
  metrics: {
    bertscore_f1: 0.873,
    alignscore: 0.74,
    fkgl: 6.2,
    coverage_evidence: 0.87,
  },
  alerts: [
    {
      summary_id: 'TECH-332',
      level: 'critical',
      detail: 'TECH-332 requiere evidencia adicional por discrepancias con FACTOR-HF.',
    },
  ],
}

export const alignmentFindingsBySummaryId = {
  'TECH-332': {
    summary_id: 'TECH-332',
    hallucinations: [
      {
        severity: 'critical',
        excerpt:
          'El resumen menciona pacientes pediátricos cuando el ensayo original solo incluye adultos > 18 años.',
        evidence: 'ClinicalTrials.gov NCT04122111 · Eligibility Criteria',
      },
      {
        severity: 'warning',
        excerpt: 'Se omite la dosis titulada quincenal descrita en el protocolo original.',
        evidence: 'FACTOR-HF Study Protocol · Section 4.2',
      },
    ],
  },
}

export const ragMetrics = {
  datasets: [
    {
      name: 'ClinicalTrials Highlights',
      precision_at_5: 0.82,
      recall_at_5: 0.76,
      ndcg: 0.88,
    },
    {
      name: 'Cochrane Summaries',
      precision_at_5: 0.79,
      recall_at_5: 0.71,
      ndcg: 0.84,
    },
  ],
}

export const jobsStatus = {
  jobs: [
    {
      job: 'Traducción Helsinki batch 2024-Q4',
      owner: 'worker-03',
      started_at: '2024-11-22 09:01',
      duration: '17m',
      state: 'in_progress',
    },
    {
      job: 'AlignScore EC2 stack',
      owner: 'aws-ollama-eval',
      started_at: '2024-11-22 08:45',
      duration: '41m',
      state: 'active',
    },
    {
      job: 'RAG embeddings refresh',
      owner: 'worker-04',
      started_at: '2024-11-22 05:30',
      duration: 'Completado',
      state: 'ok',
    },
  ],
}

export const jobsCosts = {
  services: [
    { service: 'EC2 (GPU/CPU)', monthly_cost: 1240, trend: '+8.2%' },
    { service: 'S3 storage', monthly_cost: 310, trend: '+1.5%' },
    { service: 'Data transfer', monthly_cost: 190, trend: '-2.1%' },
    { service: 'Lambda misc', monthly_cost: 120, trend: '+0.5%' },
  ],
}
