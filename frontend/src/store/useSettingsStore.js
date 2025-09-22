import { create } from 'zustand'

export const useSettingsStore = create((set) => ({
  environment: import.meta.env?.VITE_ENVIRONMENT ?? 'local',
  apiBaseUrl: import.meta.env?.VITE_API_URL ?? 'http://localhost:8000',
  mlflowUrl: import.meta.env?.VITE_MLFLOW_URL ?? 'http://localhost:5500',
  setEnvironment: (environment) => set({ environment }),
  setApiBaseUrl: (apiBaseUrl) => set({ apiBaseUrl }),
  setMlflowUrl: (mlflowUrl) => set({ mlflowUrl }),
}))
