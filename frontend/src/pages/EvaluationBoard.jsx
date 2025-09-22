import {
  Alert,
  Card,
  Col,
  Descriptions,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Typography,
  Empty,
} from 'antd'
import { useAlignmentFindings, useEvaluationOverview, useRagMetrics } from '@/hooks/useBackend.js'

const EvaluationBoardPage = () => {
  const { data: overviewData, isLoading: overviewLoading } = useEvaluationOverview()
  const { data: alignmentData, isLoading: alignmentLoading } = useAlignmentFindings('TECH-332')
  const { data: ragMetricsData, isLoading: ragLoading } = useRagMetrics()

  const overviewMetrics = overviewData?.metrics ?? {}
  const coveragePercentage =
    typeof overviewMetrics.coverage_evidence === 'number'
      ? Math.round(overviewMetrics.coverage_evidence * 100)
      : null
  const evaluationStats = [
    { key: 'bertscore_f1', label: 'BERTScore F1', precision: 3 },
    { key: 'alignscore', label: 'AlignScore', precision: 3 },
    { key: 'fkgl', label: 'FKGL', precision: 2 },
  ].filter((stat) => typeof overviewMetrics[stat.key] === 'number')

  const ragRows = (ragMetricsData?.datasets ?? []).map((dataset, index) => ({
    key: dataset.name ?? `rag-${index}`,
    dataset: dataset.name ?? 'N/A',
    precision: dataset.precision_at_5,
    recall: dataset.recall_at_5,
    ndcg: dataset.ndcg,
  }))

  const hallucinations = alignmentData?.hallucinations ?? []
  const actions = alignmentData?.actions ?? []
  const alerts = overviewData?.alerts ?? []

  return (
    <div>
      <div className="page-header">
        <span className="page-header__title">Evaluación y QA</span>
        <span className="page-header__subtitle">
          Analiza métricas de factualidad, legibilidad y cumplimiento regulatorio
        </span>
      </div>

      {alerts.length > 0 && (
        <Alert
          type="warning"
          showIcon
          message="Alertas activas"
          description={alerts.map((alert) => alert.detail).join(' • ')}
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <Card variant="borderless" style={{ borderRadius: 12 }}>
            {overviewLoading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 24 }}>
                <Spin />
              </div>
            ) : (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                {evaluationStats.map((stat) => (
                  <Statistic
                    key={stat.key}
                    title={stat.label}
                    value={overviewMetrics[stat.key]}
                    precision={stat.precision}
                  />
                ))}
                {typeof coveragePercentage === 'number' ? (
                  <Statistic title="Cobertura de evidencia" value={coveragePercentage} suffix="%" />
                ) : (
                  <Typography.Text type="secondary">Sin cobertura de evidencia registrada</Typography.Text>
                )}
              </Space>
            )}
          </Card>
        </Col>
        <Col xs={24} lg={16}>
          <Card variant="borderless" style={{ borderRadius: 12 }}>
            <Typography.Title level={5}>Métricas RAG</Typography.Title>
            <Table
              columns={[
                { title: 'Dataset', dataIndex: 'dataset' },
                { title: 'Precision@5', dataIndex: 'precision', render: (value) => value?.toFixed?.(3) ?? '—' },
                { title: 'Recall@5', dataIndex: 'recall', render: (value) => value?.toFixed?.(3) ?? '—' },
                { title: 'nDCG', dataIndex: 'ndcg', render: (value) => value?.toFixed?.(3) ?? '—' },
              ]}
              loading={ragLoading}
              dataSource={ragRows}
              pagination={false}
              size="small"
              locale={{ emptyText: 'Sin métricas RAG registradas' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card variant="borderless" style={{ borderRadius: 12 }}>
            <Typography.Title level={5}>Hallazgos de alineación</Typography.Title>
            {alignmentLoading ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 24 }}>
                <Spin />
              </div>
            ) : hallucinations.length > 0 ? (
              <Space direction="vertical" size="large" style={{ width: '100%' }}>
                {hallucinations.map((item, index) => (
                  <Card key={index} type="inner" title={item.severity ?? 'hallucination'}>
                    <Typography.Paragraph>{item.excerpt}</Typography.Paragraph>
                    <Typography.Text type="secondary">{item.evidence}</Typography.Text>
                  </Card>
                ))}
              </Space>
            ) : (
              <Empty description="Sin hallazgos de alineación" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card variant="borderless" style={{ borderRadius: 12 }}>
            <Typography.Title level={5}>Acciones recomendadas</Typography.Title>
            {actions.length > 0 ? (
              <Descriptions bordered size="small" column={1} items={actions.map((action, index) => ({
                key: `action-${index}`,
                label: `Acción ${index + 1}`,
                children: action,
              }))} />
            ) : (
              <Empty description="Sin acciones registradas" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default EvaluationBoardPage
