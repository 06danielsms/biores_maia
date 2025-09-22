import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider, App as AntApp } from 'antd'
import { MemoryRouter } from 'react-router-dom'
import DashboardPage from '@/pages/Dashboard.jsx'

describe('DashboardPage', () => {
  const renderWithProviders = () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: 0,
          refetchOnWindowFocus: false,
        },
        mutations: {
          retry: 0,
        },
      },
    })
    return render(
      <QueryClientProvider client={queryClient}>
        <ConfigProvider>
          <AntApp>
            <MemoryRouter>
              <DashboardPage />
            </MemoryRouter>
          </AntApp>
        </ConfigProvider>
      </QueryClientProvider>,
    )
  }

  it('muestra los principales KPIs', () => {
    renderWithProviders()
    expect(screen.getByText(/Cobertura evidencia/i)).toBeInTheDocument()
    expect(screen.getByText(/AlignScore promedio/i)).toBeInTheDocument()
  })

  it('describe el propósito del panel', () => {
    renderWithProviders()
    expect(
      screen.getByText(/Visión global del estado de los pipelines/i),
    ).toBeInTheDocument()
  })
})
