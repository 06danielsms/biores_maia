import { Form, Input, Modal, message } from 'antd'
import { useEffect } from 'react'
import { useSettingsStore } from '@/store/useSettingsStore.js'

const ApiSettingsModal = ({ open, onClose }) => {
  const [form] = Form.useForm()
  const apiBaseUrl = useSettingsStore((state) => state.apiBaseUrl)
  const setApiBaseUrl = useSettingsStore((state) => state.setApiBaseUrl)

  useEffect(() => {
    if (open) {
      form.setFieldsValue({ apiBaseUrl })
    }
  }, [apiBaseUrl, form, open])

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      setApiBaseUrl(values.apiBaseUrl.trim())
      message.success('Base URL actualizada')
      onClose()
    } catch (error) {
      // validation handled by form
    }
  }

  return (
    <Modal
      title="Configuración de API"
      open={open}
      onCancel={onClose}
      onOk={handleSubmit}
      okText="Guardar"
    >
      <Form layout="vertical" form={form} initialValues={{ apiBaseUrl }}>
        <Form.Item
          label="Base URL"
          name="apiBaseUrl"
          rules={[
            { required: true, message: 'La URL es obligatoria' },
            { type: 'url', message: 'Ingresa una URL válida' },
          ]}
        >
          <Input placeholder="http://localhost:8000" />
        </Form.Item>
      </Form>
    </Modal>
  )
}

export default ApiSettingsModal
