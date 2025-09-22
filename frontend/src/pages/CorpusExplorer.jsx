import {
  Badge,
  Card,
  Col,
  Descriptions,
  Divider,
  Drawer,
  Form,
  Input,
  List,
  Row,
  Select,
  Space,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useCorpusDocument, useCorpusDocuments, useDatasetState } from '@/hooks/useBackend.js'

const CorpusExplorerPage = () => {
  const [search, setSearch] = useState('')
  const [filters, setFilters] = useState({ source: 'all', translation: 'all', metrics: 'all' })
  const [selectedDocument, setSelectedDocument] = useState(null)
  const [isDrawerOpen, setDrawerOpen] = useState(false)

  const handleFiltersChange = (_, allValues) => setFilters(allValues)

  const apiFilters = useMemo(() => {
    return {
      source: filters.source !== 'all' ? filters.source : undefined,
      status: filters.translation !== 'all' ? filters.translation : undefined,
      metrics: filters.metrics !== 'all' ? (filters.metrics === 'published' ? 'published' : 'processing') : undefined,
      limit: 200,
      offset: 0,
    }
  }, [filters])

  const { data: documentsData, isLoading: isDocumentsLoading } = useCorpusDocuments(apiFilters)
  const { data: datasetState, isLoading: isDatasetsLoading } = useDatasetState()

  const documents = documentsData?.items ?? []

  useEffect(() => {
    if (!selectedDocument && documents.length > 0) {
      const first = documents[0]
      setSelectedDocument({ ...first, id: first.id ?? first.key })
    }
  }, [documents, selectedDocument])

  const filteredDocuments = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase()
    if (!normalizedSearch) return documents
    return documents.filter((doc) => {
      const file = doc.fileName?.toLowerCase() ?? ''
      const sourceValue = doc.source?.toLowerCase() ?? ''
      return file.includes(normalizedSearch) || sourceValue.includes(normalizedSearch)
    })
  }, [documents, search])

  const sourceFilters = useMemo(
    () =>
      Array.from(new Set(documents.map((doc) => doc.source).filter(Boolean))).map((source) => ({
        text: source,
        value: source,
      })),
    [documents],
  )

  const columns = useMemo(
    () => [
      {
        title: 'Fuente',
        dataIndex: 'source',
        filters: sourceFilters,
        onFilter: (value, record) => record.source === value,
      },
      { title: 'Archivo', dataIndex: 'fileName' },
      {
        title: 'Idioma',
        dataIndex: 'language',
        render: (value) => <Tag color={value === 'es' ? 'cyan' : 'blue'}>{value?.toUpperCase()}</Tag>,
      },
      {
        title: 'Legibilidad',
        dataIndex: 'readability',
        render: (value) => (value !== null && value !== undefined ? `${value} FKGL` : 'N/A'),
      },
      {
        title: 'Traducción',
        dataIndex: 'translated',
        render: (value) =>
          value ? <Badge status="success" text="Completa" /> : <Badge status="warning" text="Pendiente" />,
      },
      {
        title: 'Métricas',
        dataIndex: 'metricsReady',
        render: (value) => (value ? <Tag color="green">Publicadas</Tag> : <Tag color="orange">En proceso</Tag>),
      },
    ],
    [sourceFilters],
  )

  const alignmentColor = {
    bajo: 'green',
    medio: 'orange',
    alto: 'red',
    pendiente: 'default',
  }

  const openDrawer = (record) => {
    setSelectedDocument({ ...record, id: record.id ?? record.key })
    setDrawerOpen(true)
  }

  const selectedDocumentId = selectedDocument?.id || selectedDocument?.key
  const {
    data: selectedDocumentDetail,
    isFetching: isDetailLoading,
  } = useCorpusDocument(selectedDocumentId, isDrawerOpen)

  const renderDrawer = () => {
    if (!selectedDocument) return null

    const detail = {
      source: selectedDocumentDetail?.source ?? selectedDocument?.source,
      original: selectedDocumentDetail?.original ?? selectedDocument?.originalContent,
      translation: selectedDocumentDetail?.translation ?? selectedDocument?.translatedContent,
      metrics: selectedDocumentDetail?.metrics ?? selectedDocument?.metrics ?? {},
      comments: selectedDocumentDetail?.comments ?? selectedDocument?.comments ?? [],
      tokens: selectedDocumentDetail?.tokens ?? selectedDocument?.tokens,
      domain: selectedDocumentDetail?.domain ?? selectedDocument?.domain,
      updated_at: selectedDocumentDetail?.updated_at ?? selectedDocument?.updatedAt,
    }

    const metricsItems = [
      { label: 'BLEU', value: detail.metrics?.bleu ?? 'Pendiente' },
      { label: 'chrF2', value: detail.metrics?.chrf2 ?? 'Pendiente' },
      { label: 'FKGL', value: detail.metrics?.fkgl ?? 'Pendiente' },
      { label: 'BERTScore', value: detail.metrics?.bertscore ?? 'Pendiente' },
    ]

    const metadataItems = [
      { label: 'Fuente', value: detail.source ?? selectedDocument.source ?? 'N/A' },
      {
        label: 'Tokens',
        value: detail.tokens?.toLocaleString() ?? selectedDocument.tokens?.toLocaleString?.() ?? 'N/A',
      },
      { label: 'Dominio', value: detail.domain ?? selectedDocument.domain ?? 'N/A' },
      { label: 'Actualizado', value: detail.updated_at ?? selectedDocument.updatedAt ?? 'N/A' },
    ]

    const alignment = selectedDocument.alignmentRisk ?? 'pendiente'

    return (
      <Drawer
        title={`${selectedDocument.fileName}`}
        placement="right"
        width={720}
        onClose={() => setDrawerOpen(false)}
        open={isDrawerOpen}
        extra={<Tag color={alignmentColor[alignment] ?? 'default'}>Riesgo {alignment}</Tag>}
      >
        {isDetailLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '24px 0' }}>
            <Spin />
          </div>
        ) : (
          <Tabs
          defaultActiveKey="original"
          items={[
            {
              key: 'original',
              label: 'Original',
              children: (
                <Typography.Paragraph style={{ whiteSpace: 'pre-line' }}>
                  {detail.original ?? selectedDocument.originalContent}
                </Typography.Paragraph>
              ),
            },
            {
              key: 'translation',
              label: 'Traducción',
              children: (
                <Typography.Paragraph style={{ whiteSpace: 'pre-line' }}>
                  {detail.translation ?? selectedDocument.translatedContent}
                </Typography.Paragraph>
              ),
            },
            {
              key: 'metadata',
              label: 'Metadatos',
              children: (
                <Descriptions bordered size="small" column={2} items={metadataItems.map((item) => ({
                  key: item.label,
                  label: item.label,
                  children: item.value,
                }))} />
              ),
            },
            {
              key: 'metrics',
              label: 'Métricas destacadas',
              children: (
                <Descriptions bordered size="small" column={2} items={metricsItems.map((item) => ({
                  key: item.label,
                  label: item.label,
                  children: item.value,
                }))} />
              ),
            },
          ]}
        />
        )}

        <Divider>Comentarios y auditoría</Divider>

        <List
          itemLayout="horizontal"
          header={<strong>Historial de comentarios</strong>}
          dataSource={selectedDocumentDetail?.comments ?? selectedDocument.comments}
          locale={{ emptyText: 'Sin comentarios aún' }}
          renderItem={(comment) => (
            <List.Item key={`${comment.author ?? 'autor'}-${comment.timestamp ?? 'sin-fecha'}`}>
              <Space direction="vertical" size={2} style={{ width: '100%' }}>
                <Space size="small">
                  <Typography.Text strong>{comment.author ?? 'Sin autor'}</Typography.Text>
                  <Tag color="geekblue">{comment.role ?? 'Sin rol'}</Tag>
                  <Typography.Text type="secondary">{comment.timestamp ?? 'Sin fecha'}</Typography.Text>
                </Space>
                <Typography.Text>{comment.content}</Typography.Text>
              </Space>
            </List.Item>
          )}
        />
      </Drawer>
    )
  }

  return (
    <div>
      <div className="page-header">
        <span className="page-header__title">Explorador de corpus</span>
        <span className="page-header__subtitle">
          Filtra y navega la colección unificada con traducciones, métricas y auditoría
        </span>
      </div>
      <Row gutter={[16, 16]}>
        <Col xs={24} md={16}>
          <Card variant="borderless" style={{ borderRadius: 12 }}>
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              <Input.Search
                allowClear
                placeholder="Busca por nombre de archivo, fuente o ID"
                onSearch={setSearch}
                onChange={(event) => setSearch(event.target.value)}
              />
              <Form
                layout="inline"
                initialValues={filters}
                onValuesChange={handleFiltersChange}
                style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}
              >
                <Form.Item label="Fuente" name="source">
                  <Select
                    style={{ minWidth: 180 }}
                    options={[{ label: 'Todas', value: 'all' }, ...sourceFilters.map((item) => ({ label: item.text, value: item.value }))]}
                  />
                </Form.Item>
                <Form.Item label="Traducción" name="translation">
                  <Select
                    style={{ minWidth: 180 }}
                    options={[
                      { label: 'Todas', value: 'all' },
                      { label: 'Completas', value: 'translated' },
                      { label: 'Pendientes', value: 'pending' },
                    ]}
                  />
                </Form.Item>
                <Form.Item label="Métricas" name="metrics">
                  <Select
                    style={{ minWidth: 180 }}
                    options={[
                      { label: 'Todas', value: 'all' },
                      { label: 'Publicadas', value: 'published' },
                      { label: 'En proceso', value: 'processing' },
                    ]}
                  />
                </Form.Item>
              </Form>
              <Table
                columns={columns}
                dataSource={filteredDocuments.map((doc) => ({
                  key: doc.id ?? doc.key,
                  ...doc,
                }))}
                pagination={{ pageSize: 8 }}
                rowKey="key"
                loading={isDocumentsLoading}
                onRow={(record) => ({
                  onClick: () => openDrawer(record),
                })}
                locale={{ emptyText: 'Sin documentos que coincidan con los filtros' }}
              />
            </Space>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Card title="Estado de datasets" variant="borderless" style={{ borderRadius: 12 }}>
              {isDatasetsLoading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '24px 0' }}>
                  <Spin />
                </div>
              ) : (
                <Descriptions
                  size="small"
                  column={1}
                  items={(datasetState?.datasets ?? []).map((dataset) => ({
                    key: dataset.name,
                    label: dataset.name,
                    children: (
                      <Tag color={dataset.status === 'synced' ? 'green' : dataset.status === 'queued' ? 'blue' : 'orange'}>
                        {dataset.status}
                      </Tag>
                    ),
                  }))}
                />
              )}
            </Card>
            <Card title="Próximas integraciones" variant="borderless" style={{ borderRadius: 12 }}>
              <Space direction="vertical">
                <Tag color="blue">Exportar a DVC</Tag>
                <Tag color="magenta">Integración con MongoDB Atlas</Tag>
                <Tag color="purple">Análisis de desviaciones por fuente</Tag>
                <Tag color="geekblue">Auditoría de descargas</Tag>
              </Space>
            </Card>
          </Space>
        </Col>
      </Row>
      {renderDrawer()}
    </div>
  )
}

export default CorpusExplorerPage
