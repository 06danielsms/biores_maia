import { Alert, Button, Space, Typography } from 'antd'

const BackendOfflineNotice = ({ onRetry, retrying = false, errorMessage }) => (
  <Alert
    type="error"
    showIcon
    message="Backend no disponible"
    description={
      <Space direction="vertical" size={8}>
        <Typography.Text>
          No pudimos contactar al backend. Revisa la configuración de la API o levanta el servicio.
        </Typography.Text>
        {errorMessage && (
          <Typography.Text type="secondary">Detalle: {errorMessage}</Typography.Text>
        )}
        <Button type="primary" onClick={onRetry} loading={retrying}>
          Reintentar
        </Button>
      </Space>
    }
  />
)

export default BackendOfflineNotice
