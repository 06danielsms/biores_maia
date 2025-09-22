import { create } from 'zustand'

const DEFAULT_METRIC_IDS = ['coverage', 'alignscore', 'bertscore', 'fkgl']

const AVAILABLE_METRICS = [
  {
    id: 'coverage',
    label: 'Cobertura de evidencia',
    route: '/evaluation',
  },
  {
    id: 'alignscore',
    label: 'AlignScore promedio',
    route: '/evaluation',
  },
  {
    id: 'bertscore',
    label: 'BERTScore F1',
    route: '/evaluation',
  },
  {
    id: 'fkgl',
    label: 'Legibilidad FKGL',
    route: '/evaluation',
  },
  {
    id: 'jobsCompleted',
    label: 'Jobs completados esta semana',
    route: '/operations',
  },
]

export const useDashboardStore = create((set) => ({
  availableMetrics: AVAILABLE_METRICS,
  selectedMetricIds: DEFAULT_METRIC_IDS,
  setSelectedMetricIds: (ids) => {
    const uniqueIds = Array.from(new Set(ids))
    set({ selectedMetricIds: uniqueIds.slice(0, 4) })
  },
}))
