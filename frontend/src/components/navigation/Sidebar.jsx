import { Layout, Menu, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { useMemo } from 'react'
import { navigationItems } from '@/config/navigation.js'

const { Sider } = Layout

const Sidebar = () => {
  const location = useLocation()
  const navigate = useNavigate()

  const selectedKey = useMemo(() => {
    const match = navigationItems.find((item) =>
      item.path === '/' ? location.pathname === '/' : location.pathname.startsWith(item.path),
    )

    return match?.key ?? 'dashboard'
  }, [location.pathname])

  const menuItems = navigationItems
    .filter((item) => item.key !== 'alerts')
    .map((item) => ({
      key: item.key,
      label: item.label,
      icon: item.icon ? <item.icon /> : null,
      disabled: item.disabled,
    }))

  const handleClick = ({ key }) => {
    const item = navigationItems.find((navItem) => navItem.key === key)
    if (!item) return

    if (item.external) {
      window.open(item.path, '_blank', 'noopener,noreferrer')
      return
    }

    if (item.path) {
      navigate(item.path)
    }
  }

  return (
    <Sider width={240} breakpoint="lg" collapsedWidth={0} style={{ background: '#0f172a' }}>
      <div style={{ padding: '16px 20px' }}>
        <Typography.Title level={4} style={{ color: '#e2e8f0', marginBottom: 0 }}>
          BIORES Maia
        </Typography.Title>
        <Typography.Text style={{ color: '#94a3b8' }}>Inteligencia aplicada a salud</Typography.Text>
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[selectedKey]}
        items={menuItems}
        onClick={handleClick}
        style={{ background: '#0f172a' }}
      />
    </Sider>
  )
}

export default Sidebar
