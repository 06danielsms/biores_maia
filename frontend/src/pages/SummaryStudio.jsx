import {
  Button,
  Card,
  Col,
  Descriptions,
  Form,
  Input,
  InputNumber,
  message,
  Progress,
  Statistic,
  Row,
  Select,
  Space,
  Switch,
  Tabs,
  Tag,
  Timeline,
  Typography,
  Empty,
  Spin,
} from 'antd'
import { useMemo, useState } from 'react'
import { useCreateSummaryJob, useSummaryJob } from '@/hooks/useBackend.js'

const summaryConfigOptions = {
  models: [
    { label: 'phi3:mini (Ollama)', value: 'ollama-phi3-mini' },
    { label: 'mixtral 8x7b', value: 'mixtral-8x7b' },
    { label: 'gpt-4o-mini', value: 'gpt-4o-mini' },
  ],
  tones: [
    { label: 'Coloquial', value: 'coloquial' },
    { label: 'Profesional', value: 'profesional' },
    { label: 'Neutro', value: 'neutro' },
  ],
  citations: [
    { label: 'Formato AMA', value: 'ama' },
    { label: 'Formato Vancouver', value: 'vancouver' },
    { label: 'Sin citas', value: 'none' },
  ],
}

const DEFAULT_JOB_ID = 'summary-job-4221'

const SummaryStudioPage = () => {
  const [jobId, setJobId] = useState(DEFAULT_JOB_ID)
  const { data: summaryJob, isLoading: summaryJobLoading } = useSummaryJob(jobId)
  const createSummaryJob = useCreateSummaryJob()
  const [form] = Form.useForm()

  const jobInfo = summaryJob?.job_id ? summaryJob : null
  const progress = jobInfo?.progress ?? 0
  const metrics = jobInfo?.metrics ?? {}
  const summaryMetricEntries = [
    { key: 'readability_fkgl', label: 'FKGL', precision: 2 },
    { key: 'coherence', label: 'Coherencia', precision: 2 },
    { key: 'alignscore', label: 'AlignScore', precision: 2 },
  ].filter((item) => typeof metrics[item.key] === 'number')
  const timelineItems = (jobInfo?.timeline ?? []).map((item, index) => ({
    color: item.color ?? 'blue',
    label: item.label ?? item.timestamp ?? `Paso ${index + 1}`,
    children: (
      <div>
        <strong>{item.title ?? item.event}</strong>
        {item.description && <p style={{ marginBottom: 0 }}>{item.description}</p>}
      </div>
    ),
  }))

  const jobStatusTag = useMemo(() => {
    const status = jobInfo?.status
    if (status === 'streaming') return { color: 'blue', label: 'En ejecución' }
    if (status === 'queued') return { color: 'orange', label: 'En cola' }
    if (status === 'succeeded' || status === 'completed') return { color: 'green', label: 'Completado' }
    return status ? { color: 'default', label: status } : null
  }, [jobInfo?.status])

  return (
    <div>
      <div className="page-header">
        <span className="page-header__title">Estudio de resúmenes</span>
        <span className="page-header__subtitle">
          Configura experimentos con LLM, monitorea métricas y aprueba entregables
        </span>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={9}>
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <Card title="Configura un nuevo job" variant="borderless" style={{ borderRadius: 12 }}>
              <Form
                layout="vertical"
                form={form}
                initialValues={{
                  corpus: 'pfizer-2024-q4',
                  model: 'ollama-phi3-mini',
                  tone: 'coloquial',
                  maxLength: 520,
                  includeMetrics: true,
                }}
              >
                <Form.Item label="Dataset / Lote" name="corpus" required>
                  <Select
                    options={[
                      { label: 'Pfizer Lote Q4', value: 'pfizer-2024-q4' },
                      { label: 'ClinicalTrials Highlights', value: 'ct-highlights' },
                      { label: 'Cochrane Evidence Pack', value: 'cochrane-pack' },
                    ]}
                  />
                </Form.Item>
                <Form.Item label="Modelo" name="model" required>
                  <Select options={summaryConfigOptions.models} />
                </Form.Item>
                <Form.Item label="Longitud máxima (tokens)" name="maxLength">
                  <InputNumber min={200} max={1200} step={20} style={{ width: '100%' }} />
                </Form.Item>
                <Form.Item label="Tono" name="tone">
                  <Select options={summaryConfigOptions.tones} />
                </Form.Item>
                <Form.Item label="Formato de citas" name="citations">
                  <Select options={summaryConfigOptions.citations} placeholder="Selecciona formato" />
                </Form.Item>
                <Form.Item label="Adjuntar métricas de evaluación" name="includeMetrics" valuePropName="checked">
                  <Switch />
                </Form.Item>
                <Space>
                  <Button
                    type="primary"
                    loading={createSummaryJob.isPending}
                    onClick={async () => {
                      try {
                        const values = await form.validateFields()
                        const payload = {
                          dataset: values.corpus,
                          model: values.model,
                          max_tokens: values.maxLength,
                          tone: values.tone,
                          include_metrics: values.includeMetrics,
                        }
                        const response = await createSummaryJob.mutateAsync(payload)
                        message.success(`Job ${response.job_id} encolado correctamente`)
                        setJobId(response.job_id)
                      } catch (error) {
                        message.error(error.message)
                      }
                    }}
                  >
                    Lanzar job
                  </Button>
                  <Button
                    onClick={() => {
                      form.resetFields()
                    }}
                  >
                    Limpiar
                  </Button>
                </Space>
              </Form>
            </Card>

            <Card title="Seleccionar job" variant="borderless" style={{ borderRadius: 12 }}>
              <Input.Search
                placeholder="Introduce ID de job"
                allowClear
                enterButton="Cargar"
                onSearch={(value) => value && setJobId(value.trim())}
                defaultValue={DEFAULT_JOB_ID}
              />
            </Card>
          </Space>
        </Col>

        <Col xs={24} lg={15}>
          <Card variant="borderless" style={{ borderRadius: 12 }}>
            {summaryJobLoading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 48 }}>
                <Spin />
              </div>
            ) : jobInfo ? (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                <Space align="center" size={12}>
                  <Typography.Title level={4} style={{ margin: 0 }}>
                    Job {jobInfo.job_id}
                  </Typography.Title>
                  {jobStatusTag && <Tag color={jobStatusTag.color}>{jobStatusTag.label}</Tag>}
                </Space>
                <Progress percent={progress} status={progress < 100 ? 'active' : 'normal'} />
                <Descriptions bordered size="small" column={2} items={[
                  { key: 'dataset', label: 'Dataset', children: jobInfo.dataset ?? 'N/A' },
                  { key: 'model', label: 'Modelo', children: jobInfo.model ?? 'N/A' },
                  { key: 'started', label: 'Inicio', children: jobInfo.started_at ?? 'N/A' },
                  { key: 'status', label: 'Estado', children: jobInfo.status ?? 'N/A' },
                ]} />

                <Row gutter={[16, 16]}>
                  {summaryMetricEntries.length > 0 ? (
                    summaryMetricEntries.map((entry) => (
                      <Col key={entry.key} span={8}>
                        <Statistic title={entry.label} value={metrics[entry.key]} precision={entry.precision} />
                      </Col>
                    ))
                  ) : (
                    <Col span={24}>
                      <Typography.Text type="secondary">Sin métricas registradas para este job</Typography.Text>
                    </Col>
                  )}
                </Row>

                <Tabs
                  defaultActiveKey="timeline"
                  items={[
                    {
                      key: 'timeline',
                      label: 'Timeline',
                      children:
                        timelineItems.length > 0 ? (
                          <Timeline mode="left" items={timelineItems} />
                        ) : (
                          <Empty description="Sin eventos registrados" />
                        ),
                    },
                    {
                      key: 'deliverable',
                      label: 'Resumen estructurado',
                      children: jobInfo.summary ? (
                        <Typography.Paragraph style={{ whiteSpace: 'pre-line' }}>
                          {jobInfo.summary}
                        </Typography.Paragraph>
                      ) : (
                        <Empty description="Sin resumen disponible" />
                      ),
                    },
                  ]}
                />
              </Space>
            ) : (
              <Empty description="No se encontró información para el job seleccionado" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default SummaryStudioPage
