import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider, App as AntApp, theme, unstableSetRender } from 'antd'
import App from './App.jsx'
import './index.css'

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

const enableMocks = import.meta.env?.VITE_ENABLE_MOCKS === 'true'

const configureReact19Compatibility = () => {
  const majorVersion = Number(React.version?.split('.')?.[0] ?? 0)
  if (Number.isNaN(majorVersion) || majorVersion < 19) {
    return
  }

  const rootCache = new WeakMap()

  unstableSetRender((node, container) => {
    let root = rootCache.get(container)
    if (!root) {
      root = ReactDOM.createRoot(container)
      rootCache.set(container, root)
    }

    root.render(node)

    return () =>
      Promise.resolve().then(() => {
        const cachedRoot = rootCache.get(container)
        if (cachedRoot) {
          cachedRoot.unmount()
          rootCache.delete(container)
        }
      })
  })
}

configureReact19Compatibility()

const bootstrap = async () => {
  if (import.meta.env.DEV && enableMocks) {
    const { worker } = await import('./mocks/browser.js')
    await worker.start({ onUnhandledRequest: 'bypass' })
  }

  ReactDOM.createRoot(document.getElementById('root')).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <ConfigProvider
          theme={{
            algorithm: theme.defaultAlgorithm,
            token: {
              colorPrimary: '#1677ff',
            },
          }}
        >
          <BrowserRouter>
            <AntApp>
              <App />
            </AntApp>
          </BrowserRouter>
        </ConfigProvider>
      </QueryClientProvider>
    </React.StrictMode>,
  )
}

bootstrap()
