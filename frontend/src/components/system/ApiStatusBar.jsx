import { Alert, Button, Space, Tag, Typography } from 'antd'
import { SettingOutlined, ReloadOutlined } from '@ant-design/icons'

const ApiStatusBar = ({ isOnline, environment, onConfigure, onRetry, isLoading, errorMessage }) => (
  <div className="api-status-bar">
    <div className="api-status-bar__status">
      <Space size="middle" wrap>
        <Tag color={isOnline ? 'green' : 'red'}>API {isOnline ? 'operativa' : 'sin conexión'}</Tag>
        <Tag color={environment === 'prod' ? 'volcano' : 'blue'}>Entorno: {environment}</Tag>
      </Space>
      <Space size="small" wrap>
        <Button icon={<SettingOutlined />} onClick={onConfigure}>
          Configurar API
        </Button>
        {!isOnline && (
          <Button type="primary" icon={<ReloadOutlined />} onClick={onRetry} loading={isLoading}>
            Reintentar
          </Button>
        )}
      </Space>
    </div>
    {!isOnline && (
      <Alert
        type="error"
        showIcon
        message="Backend no disponible"
        description={
          <Typography.Text>
            {errorMessage ?? 'No pudimos contactar al backend. Revisa la configuración o levanta el servicio correspondiente.'}
          </Typography.Text>
        }
      />
    )}
  </div>
)

export default ApiStatusBar
