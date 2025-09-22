import {
  Badge,
  Button,
  Card,
  Checkbox,
  Col,
  Drawer,
  List,
  Modal,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { BellOutlined, ClockCircleOutlined, SyncOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime.js'
import KpiCards from '@/components/overview/KpiCards.jsx'
import ActivityTimeline from '@/components/overview/ActivityTimeline.jsx'
import { useDatasetState, useEvaluationOverview, useJobsStatus } from '@/hooks/useBackend.js'
import { useDashboardStore } from '@/store/useDashboardStore.js'

dayjs.extend(relativeTime)

const pipelineStatusColor = {
  ok: 'green',
  OK: 'green',
  active: 'cyan',
  'En ejecución': 'blue',
  in_progress: 'blue',
  queued: 'orange',
  'En cola': 'orange',
}

const statusIcon = {
  queued: <ClockCircleOutlined style={{ color: '#f97316' }} />,
  in_progress: <SyncOutlined spin style={{ color: '#3b82f6' }} />,
  active: <SyncOutlined spin style={{ color: '#3b82f6' }} />,
  ok: <CheckCircleOutlined style={{ color: '#22c55e' }} />,
  success: <CheckCircleOutlined style={{ color: '#22c55e' }} />,
}

const pipelineColumns = [
  { title: 'Pipeline', dataIndex: 'name' },
  {
    title: 'Estado',
    dataIndex: 'status',
    render: (value) => (
      <Space>
        {statusIcon[value] ?? <ExclamationCircleOutlined style={{ color: '#f97316' }} />}
        <Tag color={pipelineStatusColor[value] || 'default'}>{value}</Tag>
      </Space>
    ),
  },
  {
    title: 'Última ejecución',
    dataIndex: 'lastRunRelative',
    render: (text, record) =>
      record.lastRunTooltip ? (
        <Tooltip title={record.lastRunTooltip}>{text}</Tooltip>
      ) : (
        <span>—</span>
      ),
  },
  { title: 'Duración', dataIndex: 'duration' },
  { title: 'Owner', dataIndex: 'owner' },
]

const DashboardPage = () => {
  const navigate = useNavigate()
  const { availableMetrics, selectedMetricIds, setSelectedMetricIds } = useDashboardStore()
  const { data: evaluationOverview } = useEvaluationOverview()
  const { data: datasetState, isLoading: datasetsLoading } = useDatasetState()
  const { data: jobsStatus, isLoading: jobsLoading } = useJobsStatus()

  const overviewMetrics = evaluationOverview?.metrics
  const coverageValue =
    typeof overviewMetrics?.coverage_evidence === 'number'
      ? Math.round(overviewMetrics.coverage_evidence * 100)
      : null
  const alignscoreValue =
    typeof overviewMetrics?.alignscore === 'number' ? overviewMetrics.alignscore : null
  const bertscoreValue =
    typeof overviewMetrics?.bertscore_f1 === 'number' ? overviewMetrics.bertscore_f1 : null
  const fkglValue = typeof overviewMetrics?.fkgl === 'number' ? overviewMetrics.fkgl : null

  const jobsList = Array.isArray(jobsStatus?.jobs) ? jobsStatus.jobs : []

  const [isCustomizeModalOpen, setCustomizeModalOpen] = useState(false)
  const [selectedMetricIdsDraft, setSelectedMetricIdsDraft] = useState(selectedMetricIds)
  const [isNotificationsOpen, setNotificationsOpen] = useState(false)

  useEffect(() => {
    setSelectedMetricIdsDraft(selectedMetricIds)
  }, [selectedMetricIds])

  const totalJobsCompleted = jobsList.filter((job) => job.state === 'ok').length
  const metricConfig = useMemo(() => {
    const entries = {}

    if (coverageValue !== null) {
      entries.coverage = {
        id: 'coverage',
        title: 'Cobertura evidencia',
        value: `${coverageValue}%`,
        progress: coverageValue,
        trend: Number((coverageValue - 85).toFixed(1)),
        trendDescription: 'Comparación con umbral interno (85%)',
        onClick: () => navigate('/evaluation'),
      }
    }

    if (alignscoreValue !== null) {
      entries.alignscore = {
        id: 'alignscore',
        title: 'AlignScore promedio',
        value: alignscoreValue,
        precision: 3,
        trend: Number(((alignscoreValue - 0.72) * 100).toFixed(1)),
        trendDescription: 'Diferencia vs resultado histórico (0.72)',
        onClick: () => navigate('/evaluation'),
      }
    }

    if (bertscoreValue !== null) {
      entries.bertscore = {
        id: 'bertscore',
        title: 'BERTScore F1',
        value: bertscoreValue,
        precision: 3,
        trend: Number(((bertscoreValue - 0.85) * 100).toFixed(1)),
        trendDescription: 'Diferencia vs benchmark 0.85',
        onClick: () => navigate('/evaluation'),
      }
    }

    if (fkglValue !== null) {
      entries.fkgl = {
        id: 'fkgl',
        title: 'Legibilidad FKGL',
        value: fkglValue,
        suffix: ' FKGL',
        precision: 2,
        trend: Number(((fkglValue - 6.5) * 100).toFixed(1)),
        trendDescription: 'Diferencia vs objetivo 6.5',
        onClick: () => navigate('/evaluation'),
      }
    }

    entries.jobsCompleted = {
      id: 'jobsCompleted',
      title: 'Jobs completados (7 días)',
      value: totalJobsCompleted,
      trend: totalJobsCompleted > 0 ? Number((totalJobsCompleted * 10).toFixed(1)) : 0,
      trendDescription: 'Comparado con la semana anterior',
      onClick: () => navigate('/operations'),
    }

    return entries
  }, [alignscoreValue, bertscoreValue, coverageValue, fkglValue, navigate, totalJobsCompleted])

  const kpiData = useMemo(
    () =>
      selectedMetricIds
        .map((id) => metricConfig[id])
        .filter((metric) => metric && metric.value !== undefined && metric.value !== null)
        .map((metric) => ({
          ...metric,
        })),
    [metricConfig, selectedMetricIds],
  )

  const pipelineData = useMemo(
    () =>
      (jobsStatus?.jobs ?? []).map((job, index) => {
        const startedAt = job.started_at ?? job.startedAt
        return {
          key: `${job.job}-${index}`,
          name: job.job,
          status: job.state ?? 'unknown',
          lastRunRelative: startedAt ? dayjs(startedAt).fromNow() : '—',
          lastRunTooltip: startedAt ?? null,
          duration: job.duration ?? '—',
          owner: job.owner ?? '—',
        }
      }),
    [jobsStatus],
  )

  const activityItems = useMemo(() => {
    return (jobsStatus?.jobs ?? [])
      .map((job) => {
        const startedAt = job.started_at ?? job.startedAt
        const startMoment = startedAt ? dayjs(startedAt) : null
        return {
          title: job.job,
          description: job.owner ? `${job.owner} · ${job.state}` : job.state,
          timestamp: startMoment ? startMoment.format('YYYY-MM-DD HH:mm') : 'Sin registro',
          color:
            job.state === 'ok'
              ? 'green'
              : job.state === 'in_progress' || job.state === 'active'
              ? 'blue'
              : job.state === 'queued'
              ? 'orange'
              : 'default',
          type: job.state,
          sortValue: startMoment ? startMoment.valueOf() : 0,
        }
      })
      .sort((a, b) => b.sortValue - a.sortValue)
      .slice(0, 6)
      .map(({ color, label, children }) => ({ color, label, children }))
  }, [jobsStatus])

  const qualityAlerts = (evaluationOverview?.alerts ?? []).map((alert, index) => ({
    key: alert.summary_id ?? index,
    severity: alert.level === 'critical' ? 'Crítica' : alert.level === 'warning' ? 'Alta' : alert.level,
    title: `Resumen ${alert.summary_id ?? 'N/A'}`,
    description: alert.detail,
  }))

  return (
    <div className="dashboard">
      <div className="page-header">
        <div>
          <span className="page-header__title">Panel operativo</span>
          <span className="page-header__subtitle">
            Visión global del estado de los pipelines, costos y calidad de los entregables
          </span>
        </div>
        <div className="page-header__actions">
          <Button type="primary" ghost icon={<BellOutlined />} onClick={() => setNotificationsOpen(true)}>
            Alertas ({qualityAlerts.length})
          </Button>
        </div>
      </div>
      <KpiCards items={kpiData} onCardClick={(metric) => metric.onClick?.(metric)} onConfigure={() => setCustomizeModalOpen(true)} />
      <Row gutter={[16, 16]} align="stretch" style={{ marginTop: 8 }}>
        <Col xs={24} xl={16} style={{ display: 'flex' }}>
          <Card
            title="Pipelines críticos"
            className="dashboard-section"
            styles={{ body: { padding: 16 } }}
            style={{ width: '100%' }}
          >
            {jobsLoading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 24 }}>
                <Spin />
              </div>
            ) : (
              <Table
                columns={pipelineColumns}
                dataSource={pipelineData}
                pagination={false}
                size="small"
                rowKey="key"
                locale={{ emptyText: 'Sin ejecuciones registradas' }}
              />
            )}
          </Card>
        </Col>
        <Col xs={24} xl={8} style={{ display: 'flex' }}>
          <Card
            title="Alertas prioritarias"
            className="dashboard-section"
            styles={{ body: { padding: 16, minHeight: 220 } }}
            extra={<Button size="small" onClick={() => navigate('/evaluation')}>Ver detalle</Button>}
            style={{ width: '100%' }}
          >
            <List
              dataSource={qualityAlerts}
              locale={{ emptyText: 'Sin alertas pendientes' }}
              renderItem={(item) => (
                <List.Item key={item.key}>
                  <Space direction="vertical" size={4} style={{ width: '100%' }}>
                    <Space align="center">
                      <Badge status={item.severity === 'Crítica' ? 'error' : 'warning'} />
                      <strong>{item.title}</strong>
                    </Space>
                    <Typography.Text type="secondary">{item.description}</Typography.Text>
                    <Button size="small" type="link" onClick={() => navigate('/evaluation')}>
                      Revisar en evaluación
                    </Button>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
      <Row gutter={[16, 16]} align="stretch">
        <Col xs={24} xl={16} style={{ display: 'flex' }}>
          <Card className="dashboard-section" style={{ width: '100%' }} title="Benchmarks comparativos">
            <Tabs
              defaultActiveKey="legibilidad"
              items={[
                {
                  key: 'legibilidad',
                  label: 'Legibilidad',
                  children:
                    fkglValue !== null ? (
                      <Statistic title="FKGL promedio" value={fkglValue} precision={2} suffix=" FKGL" />
                    ) : (
                      <Typography.Text type="secondary">Sin datos de legibilidad</Typography.Text>
                    ),
                },
                {
                  key: 'factualidad',
                  label: 'Factualidad',
                  children:
                    alignscoreValue !== null ? (
                      <Statistic title="AlignScore" value={alignscoreValue} precision={3} />
                    ) : (
                      <Typography.Text type="secondary">Sin datos de AlignScore</Typography.Text>
                    ),
                },
              ]}
            />
          </Card>
        </Col>
        <Col xs={24} xl={8} style={{ display: 'flex' }}>
          <ActivityTimeline activities={activityItems} />
        </Col>
      </Row>
      <Card
        className="dashboard-section"
        style={{ marginTop: 0 }}
        title="Estado de datasets"
      >
        {datasetsLoading ? (
          <Spin />
        ) : (
          <List
            grid={{ gutter: 16, column: 3 }}
            dataSource={datasetState?.datasets ?? []}
            locale={{ emptyText: 'Sin datasets registrados' }}
            renderItem={(item) => (
              <List.Item>
                <Card size="small" bordered={false} style={{ borderRadius: 12 }}>
                  <Typography.Text strong>{item.name}</Typography.Text>
                  <br />
                  <Typography.Text type="secondary">Versión: {item.version}</Typography.Text>
                  <br />
                  <Typography.Text type="secondary">Documentos: {item.document_count ?? '—'}</Typography.Text>
                  <br />
                  <Tag color={item.status === 'synced' ? 'green' : item.status === 'queued' ? 'blue' : 'orange'}>
                    {item.status}
                  </Tag>
                </Card>
              </List.Item>
            )}
          />
        )}
      </Card>
      <Modal
        title="Selecciona hasta 4 indicadores"
        open={isCustomizeModalOpen}
        onCancel={() => {
          setSelectedMetricIdsDraft(selectedMetricIds)
          setCustomizeModalOpen(false)
        }}
        onOk={() => {
          setSelectedMetricIds(selectedMetricIdsDraft)
          setCustomizeModalOpen(false)
        }}
      >
        <Checkbox.Group
          value={selectedMetricIdsDraft}
          onChange={(values) => {
            if (values.length > 4) {
              message.warning('Puedes seleccionar hasta 4 indicadores')
              return
            }
            setSelectedMetricIdsDraft(values)
          }}
          style={{ display: 'grid', gap: 8 }}
        >
          {availableMetrics.map((metric) => (
            <Checkbox key={metric.id} value={metric.id}>
              {metric.label}
            </Checkbox>
          ))}
        </Checkbox.Group>
      </Modal>
      <Drawer
        title="Centro de alertas"
        placement="right"
        open={isNotificationsOpen}
        onClose={() => setNotificationsOpen(false)}
      >
        <List
          dataSource={qualityAlerts}
          locale={{ emptyText: 'No hay alertas activas' }}
          renderItem={(item) => (
            <List.Item key={item.key}>
              <List.Item.Meta
                title={item.title}
                description={item.description}
                avatar={<Badge status={item.severity === 'Crítica' ? 'error' : 'warning'} />}
              />
            </List.Item>
          )}
        />
      </Drawer>
    </div>
  )
}

export default DashboardPage
