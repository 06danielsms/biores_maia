import { Layout, Typography, Breadcrumb } from 'antd'
import { Outlet, useLocation } from 'react-router-dom'
import { useState } from 'react'
import Sidebar from '@/components/navigation/Sidebar.jsx'
import { navigationItems } from '@/config/navigation.js'
import { useHealth } from '@/hooks/useApi.js'
import { useSettingsStore } from '@/store/useSettingsStore.js'
import ApiStatusBar from '@/components/system/ApiStatusBar.jsx'
import ApiSettingsModal from '@/components/system/ApiSettingsModal.jsx'
import '@/App.css'

const { Header, Content } = Layout

const MainLayout = () => {
  const location = useLocation()
  const environment = useSettingsStore((state) => state.environment)
  const { isSuccess, isError, error, refetch, isFetching } = useHealth()
  const [isSettingsModalOpen, setSettingsModalOpen] = useState(false)

  const breadcrumbs = location.pathname
    .split('/')
    .filter(Boolean)
    .map((segment, index, segments) => {
      const path = `/${segments.slice(0, index + 1).join('/')}`
      const item = navigationItems.find((navItem) => navItem.path === path)
      return { title: item?.label ?? segment }
    })

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sidebar />
      <Layout>
        <Header style={{ background: '#fff', padding: '0 24px', borderBottom: '1px solid #e2e8f0' }} />
        <ApiStatusBar
          isOnline={isSuccess}
          environment={environment}
          onConfigure={() => setSettingsModalOpen(true)}
          onRetry={() => refetch()}
          isLoading={isFetching}
          errorMessage={error?.message}
        />
        <Content className="layout-content">
          {breadcrumbs.length > 0 && <Breadcrumb items={breadcrumbs} style={{ marginBottom: 16 }} />}
          <Outlet />
        </Content>
        <ApiSettingsModal open={isSettingsModalOpen} onClose={() => setSettingsModalOpen(false)} />
      </Layout>
    </Layout>
  )
}

export default MainLayout
