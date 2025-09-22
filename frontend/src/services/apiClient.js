import axios from 'axios'
import { useSettingsStore } from '@/store/useSettingsStore.js'

export const createApiClient = () => {
  const apiBaseUrl = useSettingsStore.getState().apiBaseUrl
  let normalizedBaseUrl = apiBaseUrl?.replace(/\/$/, '') || 'http://localhost:8000'

  if (typeof window !== 'undefined') {
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    if (isLocalhost && normalizedBaseUrl?.includes('backend:')) {
      const url = new URL(normalizedBaseUrl)
      url.hostname = 'localhost'
      normalizedBaseUrl = url.toString().replace(/\/$/, '')
    }
  }

  const instance = axios.create({
    baseURL: normalizedBaseUrl,
    timeout: 15000,
  })

  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      if (!error.response) {
        return Promise.reject(
          new Error('No se pudo contactar al backend. Verifica VITE_API_URL o enciende el servicio.'),
        )
      }

      const message = error.response?.data?.detail ?? error.message
      return Promise.reject(new Error(message))
    },
  )

  return instance
}

export const apiClient = createApiClient()
