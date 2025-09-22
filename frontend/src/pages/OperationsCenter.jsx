import {
  Card,
  Col,
  Empty,
  List,
  Progress,
  Row,
  Space,
  Statistic,
  Table,
  Tabs,
  Tag,
  Timeline,
  Typography,
} from 'antd'
import { useJobsCosts, useJobsStatus } from '@/hooks/useBackend.js'

const OperationsCenterPage = () => {
  const { data: jobsStatusData, isLoading: jobsStatusLoading } = useJobsStatus()
  const { data: jobsCostData, isLoading: jobsCostLoading } = useJobsCosts()

  const statusLabel = (value) => {
    const map = {
      in_progress: 'En progreso',
      active: 'Activo',
      ok: 'OK',
      queued: 'En cola',
    }
    return map[value] ?? value
  }

  const jobStatusRows = (jobsStatusData?.jobs ?? []).map((item) => ({
    key: item.job ?? item.key,
    job: item.job,
    owner: item.owner,
    startedAt: item.started_at ?? item.startedAt,
    duration: item.duration,
    state: item.state ?? item.state,
  }))

  const costRows = (jobsCostData?.services ?? []).map((item) => ({
    key: item.service ?? item.key,
    service: item.service,
    monthly: item.monthly_cost ?? item.monthly,
    trend: item.trend,
  }))

  const totalMonthlyCost = costRows.reduce(
    (acc, item) => acc + (typeof item.monthly === 'number' ? item.monthly : 0),
    0,
  )
  const hasCostData = costRows.some((item) => typeof item.monthly === 'number')

  const activeJobsCount = jobStatusRows.filter((job) =>
    ['in_progress', 'active'].includes((job.state ?? '').toLowerCase()),
  ).length
  const completedJobsCount = jobStatusRows.filter((job) => (job.state ?? '').toLowerCase() === 'ok').length
  const completionPercent = jobStatusRows.length
    ? Math.round((completedJobsCount / jobStatusRows.length) * 100)
    : 0

  const timelineItems = jobStatusRows
    .filter((job) => job.startedAt)
    .map((job) => ({
      color:
        (job.state ?? '').toLowerCase() === 'ok'
          ? 'green'
          : ['in_progress', 'active'].includes((job.state ?? '').toLowerCase())
          ? 'blue'
          : 'orange',
      label: job.startedAt,
      children: (
        <div>
          <strong>{job.job}</strong>
          <p style={{ marginBottom: 0 }}>{job.owner ?? 'Sin owner'} · {statusLabel(job.state)}</p>
        </div>
      ),
      sort: new Date(job.startedAt ?? '').getTime() || 0,
    }))
    .sort((a, b) => b.sort - a.sort)

  const auditItems = jobStatusRows.map((job, index) => ({
    key: `${job.key}-${index}`,
    time: job.startedAt ?? 'Sin fecha',
    action: `${job.owner ?? 'N/A'} lanzó ${job.job} (${statusLabel(job.state)})`,
  }))

  return (
  <div>
    <div className="page-header">
      <span className="page-header__title">Operaciones y gobernanza</span>
      <span className="page-header__subtitle">
        Controla infraestructura, costos y evidencias de auditoría
      </span>
    </div>

    <Row gutter={[16, 16]}>
      <Col xs={24} lg={8}>
        <Card variant="borderless" style={{ borderRadius: 12 }}>
          {hasCostData ? (
            <Statistic title="Costo mensual AWS" prefix="US$" value={totalMonthlyCost.toFixed(0)} precision={0} />
          ) : (
            <Typography.Text type="secondary">Sin costos disponibles</Typography.Text>
          )}
          <Statistic title="Jobs activos" value={activeJobsCount} style={{ marginTop: 16 }} />
          <Statistic title="Jobs completados" value={completedJobsCount} style={{ marginTop: 16 }} />
          <div style={{ marginTop: 16 }}>
            <Typography.Text type="secondary">Avance jobs semana actual</Typography.Text>
            <Progress
              percent={completionPercent}
              strokeColor="#52c41a"
            />
          </div>
        </Card>
      </Col>
      <Col xs={24} lg={16}>
        <Card variant="borderless" style={{ borderRadius: 12 }}>
          <Tabs
            defaultActiveKey="jobs"
            items={[
              {
                key: 'jobs',
                label: 'Jobs activos',
                children: (
                  <Table
                    columns={[
                      { title: 'Job', dataIndex: 'job' },
                      { title: 'Owner', dataIndex: 'owner' },
                      { title: 'Inicio', dataIndex: 'startedAt' },
                      { title: 'Duración', dataIndex: 'duration' },
                      {
                        title: 'Estado',
                        dataIndex: 'state',
                        render: (value) => (
                          <Tag color={value === 'OK' || value === 'ok' ? 'green' : value === 'En progreso' || value === 'in_progress' ? 'blue' : value === 'Activo' || value === 'active' ? 'cyan' : 'orange'}>
                            {statusLabel(value)}
                          </Tag>
                        ),
                      },
                    ]}
                    dataSource={jobStatusRows}
                    loading={jobsStatusLoading}
                    pagination={false}
                    size="small"
                    locale={{ emptyText: 'Sin jobs registrados' }}
                  />
                ),
              },
              {
                key: 'costos',
                label: 'Costos AWS',
                children: (
                  <Table
                    columns={[
                      { title: 'Servicio', dataIndex: 'service' },
                      {
                        title: 'Costo mensual',
                        dataIndex: 'monthly',
                        render: (value) => (value ? `US$ ${value}` : '—'),
                      },
                      { title: 'Tendencia', dataIndex: 'trend' },
                    ]}
                    dataSource={costRows}
                    loading={jobsCostLoading}
                    pagination={false}
                    size="small"
                    locale={{ emptyText: 'Sin costos disponibles' }}
                  />
                ),
              },
            ]}
          />
        </Card>
      </Col>
    </Row>

    <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
      <Col xs={24} lg={12}>
        <Card variant="borderless" style={{ borderRadius: 12 }}>
          <Typography.Title level={5}>Línea de tiempo de infraestructura</Typography.Title>
          {timelineItems.length > 0 ? <Timeline mode="left" items={timelineItems} /> : <Empty description="Sin eventos registrados" />}
        </Card>
      </Col>
      <Col xs={24} lg={12}>
        <Card variant="borderless" style={{ borderRadius: 12 }}>
          <Typography.Title level={5}>Auditoría y gobernanza</Typography.Title>
          {auditItems.length > 0 ? (
            <List
              size="small"
              dataSource={auditItems}
              renderItem={(item) => (
                <List.Item>
                  <Space direction="vertical" size={0}>
                    <Typography.Text strong>{item.time}</Typography.Text>
                    <Typography.Text>{item.action}</Typography.Text>
                  </Space>
                </List.Item>
              )}
            />
          ) : (
            <Empty description="Sin actividades registradas" />
          )}
        </Card>
      </Col>
    </Row>
  </div>
)
}

export default OperationsCenterPage
