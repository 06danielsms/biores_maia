import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/services/apiClient.js'

export const useApi = ({ queryKey, url, enabled = true, params }) => {
  return useQuery({
    queryKey,
    enabled,
    queryFn: async () => {
      const response = await apiClient.get(url, { params })
      return response.data
    },
  })
}

export const useHealth = () =>
  useApi({
    queryKey: ['health'],
    url: '/health/',
  })
