import {
  Alert,
  Button,
  Card,
  Col,
  Form,
  Input,
  List,
  message,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Switch,
  Tabs,
  Tag,
  Timeline,
  Typography,
} from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useCorpusDocument, useCorpusDocuments, useRecomputeTranslation, useTranslationMetrics } from '@/hooks/useBackend.js'

const highlightDifferences = (source, translation) => {
  const normalizedSource = new Set(
    (source || '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, '')
      .split(/\s+/)
      .filter(Boolean),
  )

  return (translation || '').split(/(\s+)/).map((token, index) => {
    if (/^\s+$/.test(token)) {
      return token
    }

    const cleanToken = token
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]/gu, '')

    if (!cleanToken) {
      return token
    }

    const isNewToken = !normalizedSource.has(cleanToken)
    if (isNewToken) {
      return (
        <Typography.Text mark key={`${token}-${index}`}>
          {token}
        </Typography.Text>
      )
    }

    return <span key={`${token}-${index}`}>{token}</span>
  })
}

const TranslationLabPage = () => {
  const [form] = Form.useForm()
  const { data: corpusData, isLoading: documentsLoading } = useCorpusDocuments({ limit: 200, offset: 0 })
  const documents = corpusData?.items ?? []
  const [selectedDocumentId, setSelectedDocumentId] = useState(documents[0]?.id ?? null)

  useEffect(() => {
    if (documents.length > 0 && !selectedDocumentId) {
      setSelectedDocumentId(documents[0].id)
    }
  }, [documents, selectedDocumentId])

  const { data: documentDetail, isFetching: detailLoading } = useCorpusDocument(selectedDocumentId, Boolean(selectedDocumentId))
  const { data: translationMetricsData, isLoading: metricsLoading } = useTranslationMetrics(selectedDocumentId, Boolean(selectedDocumentId))
  const recomputeMutation = useRecomputeTranslation()

  const [editedTranslation, setEditedTranslation] = useState('')

  useEffect(() => {
    setEditedTranslation(documentDetail?.translation ?? '')
  }, [documentDetail?.translation])

  const metrics = translationMetricsData?.metrics ?? {}
  const metricEntries = [
    { key: 'bleu', label: 'BLEU', precision: 1, suffix: ' pts' },
    { key: 'chrf2', label: 'chrF2', precision: 3 },
    { key: 'fkgl', label: 'FKGL', precision: 1 },
    { key: 'length_ratio', label: 'Ratio de longitud', precision: 2, suffix: 'x' },
  ].filter((entry) => typeof metrics[entry.key] === 'number')

  const highlightedTranslation = useMemo(
    () => highlightDifferences(documentDetail?.original ?? '', documentDetail?.translation ?? ''),
    [documentDetail?.original, documentDetail?.translation],
  )

  const highlightedCustomRevision = useMemo(
    () => highlightDifferences(documentDetail?.translation ?? '', editedTranslation ?? ''),
    [documentDetail?.translation, editedTranslation],
  )

  const handleLaunchJob = async () => {
    if (!selectedDocumentId) {
      message.warning('Selecciona un documento antes de lanzar el job')
      return
    }
    try {
      const values = form.getFieldsValue()
      await recomputeMutation.mutateAsync({
        document_id: selectedDocumentId,
        model: values.model,
        force: true,
        notify: values.notify ?? false,
      })
      message.success('Recomputo encolado correctamente')
    } catch (error) {
      message.error(error.message)
    }
  }

  const qualityAlerts = useMemo(() => {
    const result = []
    if (typeof metrics.bleu === 'number' && metrics.bleu < 40) {
      result.push({
        key: 'bleu',
        severity: 'alta',
        title: 'BLEU por debajo del umbral',
        description: `BLEU actual ${metrics.bleu.toFixed(1)} (< 40)` ,
      })
    }
    if (typeof metrics.chrf2 === 'number' && metrics.chrf2 < 0.65) {
      result.push({
        key: 'chrf2',
        severity: 'media',
        title: 'chrF2 bajo',
        description: `chrF2 actual ${metrics.chrf2.toFixed(3)} (< 0.65)`,
      })
    }
    return result
  }, [metrics])

  const timelineItems = (translationMetricsData?.history ?? []).map((item) => {
    const timestamp = item.timestamp ?? item.time
    const label = timestamp ? String(timestamp) : 'Sin marca de tiempo'
    return {
      color: item.color ?? 'blue',
      label,
      children: (
        <div>
          <strong>{item.event ?? item.title ?? 'Evento registrado'}</strong>
          {item.detail && <p style={{ marginBottom: 0 }}>{item.detail}</p>}
        </div>
      ),
    }
  })

  return (
    <div>
      <div className="page-header">
        <span className="page-header__title">Laboratorio de traducción</span>
        <span className="page-header__subtitle">
          Configura recomputos, compara métricas y supervisa la calidad lingüística
        </span>
      </div>

      {qualityAlerts.length > 0 && (
        <Alert
          type="warning"
          showIcon
          message="Se detectaron métricas por debajo del umbral"
          description={`Revisa ${qualityAlerts.map((alert) => alert.title).join(', ')}`}
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Card title="Configura un nuevo job" variant="borderless" style={{ borderRadius: 12 }}>
              <Form
                layout="vertical"
                form={form}
                initialValues={{
                  model: 'helsinki',
                  notify: true,
                }}
              >
                <Form.Item label="Documento" name="document" initialValue={selectedDocumentId}>
                  <Select
                    loading={documentsLoading}
                    options={documents.map((doc) => ({ label: doc.fileName ?? doc.id, value: doc.id }))}
                    onChange={(value) => setSelectedDocumentId(value)}
                    placeholder="Selecciona documento"
                  />
                </Form.Item>
                <Form.Item label="Modelo de traducción" name="model">
                  <Select
                    options={[
                      { label: 'Helsinki-NLP/opus-mt-en-es', value: 'helsinki' },
                      { label: 'NLLB 1.3B', value: 'nllb' },
                      { label: 'Custom LLM (Ollama phi3)', value: 'ollama-phi3' },
                    ]}
                  />
                </Form.Item>
                <Form.Item label="Notificar al completar" name="notify" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Button type="primary" onClick={handleLaunchJob} loading={recomputeMutation.isPending}>
                  Lanzar job
                </Button>
              </Form>
            </Card>

            <Card title="Métricas principales" variant="borderless" style={{ borderRadius: 12 }}>
              {metricsLoading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: 24 }}>
                  <Spin />
                </div>
              ) : metricEntries.length > 0 ? (
                <Row gutter={[16, 16]}>
                  {metricEntries.map((entry) => (
                    <Col key={entry.key} span={12}>
                      <Statistic
                        title={entry.label}
                        value={metrics[entry.key]}
                        precision={entry.precision}
                        suffix={entry.suffix}
                      />
                    </Col>
                  ))}
                </Row>
              ) : (
                <Typography.Text type="secondary">Sin métricas registradas para este documento.</Typography.Text>
              )}
            </Card>

            <Card title="Alertas de calidad" variant="borderless" style={{ borderRadius: 12 }}>
              <List
                dataSource={qualityAlerts}
                locale={{ emptyText: 'Sin alertas registradas para este documento' }}
                renderItem={(item) => (
                  <List.Item key={item.key}>
                    <List.Item.Meta
                      title={
                        <Space size="small">
                          <Tag color={item.severity === 'alta' ? 'red' : 'orange'}>Severidad {item.severity}</Tag>
                          <span>{item.title}</span>
                        </Space>
                      }
                      description={item.description}
                    />
                  </List.Item>
                )}
              />
            </Card>

            <Card title="Histórico de recomputos" variant="borderless" style={{ borderRadius: 12 }}>
              {timelineItems.length > 0 ? (
                <Timeline mode="left" items={timelineItems} />
              ) : (
                <Typography.Text type="secondary">Sin ejecuciones registradas.</Typography.Text>
              )}
            </Card>
          </Space>
        </Col>

        <Col xs={24} lg={14}>
          <Tabs
            defaultActiveKey="compare"
            items={[
              {
                key: 'compare',
                label: 'Comparador de traducciones',
                children: (
                  <Row gutter={[16, 16]}>
                    <Col span={24}>
                      <Card title="Texto original" variant="borderless" style={{ borderRadius: 12 }}>
                        {detailLoading ? (
                          <Spin />
                        ) : (
                          <Typography.Paragraph>
                            {documentDetail?.original || 'Sin contenido disponible'}
                          </Typography.Paragraph>
                        )}
                      </Card>
                    </Col>
                    <Col span={24}>
                      <Card
                        title="Traducción automática"
                        variant="borderless"
                        style={{ borderRadius: 12 }}
                        extra={
                          typeof metrics.bleu === 'number' ? (
                            <Tag color={metrics.bleu >= 40 ? 'blue' : 'orange'}>BLEU {metrics.bleu.toFixed(1)}</Tag>
                          ) : null
                        }
                      >
                        {detailLoading ? (
                          <Spin />
                        ) : documentDetail?.translation ? (
                          <Typography.Paragraph>{highlightedTranslation}</Typography.Paragraph>
                        ) : (
                          <Typography.Text type="secondary">Sin traducción disponible</Typography.Text>
                        )}
                      </Card>
                    </Col>
                    <Col span={24}>
                      <Card title="Revisión propuesta" variant="borderless" style={{ borderRadius: 12 }}>
                        <Typography.Paragraph strong>Editar versión propuesta</Typography.Paragraph>
                        <Input.TextArea
                          rows={6}
                          value={editedTranslation}
                          onChange={(event) => setEditedTranslation(event.target.value)}
                          placeholder="Introduce tu versión aprobada"
                          style={{ marginBottom: 12 }}
                        />
                        <Typography.Paragraph style={{ marginBottom: 0 }}>
                          {editedTranslation ? highlightedCustomRevision : 'Agrega una propuesta para comparar contra la traducción automática.'}
                        </Typography.Paragraph>
                      </Card>
                    </Col>
                  </Row>
                ),
              },
              {
                key: 'metadata',
                label: 'Metadatos',
                children: (
                  <Card variant="borderless" style={{ borderRadius: 12 }}>
                    <Typography.Paragraph>
                      <strong>Fuente:</strong> {documentDetail?.source ?? 'N/A'}
                    </Typography.Paragraph>
                    <Typography.Paragraph>
                      <strong>Tokens:</strong> {documentDetail?.tokens ?? 'N/A'}
                    </Typography.Paragraph>
                    <Typography.Paragraph>
                      <strong>Idioma:</strong> {documentDetail?.language ?? 'N/A'}
                    </Typography.Paragraph>
                  </Card>
                ),
              },
            ]}
          />
        </Col>
      </Row>
    </div>
  )
}

export default TranslationLabPage
