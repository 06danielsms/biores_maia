import { Button, Card, Col, Progress, Row, Statistic, Tooltip, Typography } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined, SettingOutlined } from '@ant-design/icons'

const isNumeric = (value) => typeof value === 'number' && !Number.isNaN(value)

const trendColor = (value) => {
  if (!isNumeric(value)) return '#475569'
  if (value > 0) return '#16a34a'
  if (value < 0) return '#dc2626'
  return '#475569'
}

const formatTrend = (value) => {
  if (!isNumeric(value)) return '—'
  if (value === 0) return '0%'
  return value > 0 ? `+${value}%` : `${value}%`
}

const TrendIcon = ({ value }) =>
  !isNumeric(value) || value === 0 ? (
    <MinusOutlined style={{ color: '#475569' }} />
  ) : value > 0 ? (
    <ArrowUpOutlined style={{ color: '#16a34a' }} />
  ) : (
    <ArrowDownOutlined style={{ color: '#dc2626' }} />
  )

const KpiCards = ({ items, onCardClick, onConfigure }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
      <Typography.Title level={5} style={{ margin: 0 }}>
        Indicadores clave
      </Typography.Title>
      {onConfigure && (
        <Button size="small" icon={<SettingOutlined />} onClick={onConfigure}>
          Personalizar
        </Button>
      )}
    </div>
    <Row gutter={[16, 16]} align="stretch">
      {items.map((item) => (
        <Col key={item.id} xs={24} sm={12} xl={6} style={{ display: 'flex' }}>
          <Card
            size="small"
            hoverable
            variant="borderless"
            style={{
              borderRadius: 12,
              cursor: item.onClick || onCardClick ? 'pointer' : 'default',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              width: '100%',
              minHeight: 160,
            }}
            onClick={() => {
              if (item.onClick) {
                item.onClick(item)
                return
              }
              if (onCardClick) {
                onCardClick(item)
              }
            }}
          >
            <SpaceBetween title={item.title} subtitle={item.subtitle} />
            <Statistic
              value={item.value}
              prefix={item.prefix}
              suffix={item.suffix}
              precision={item.precision}
              valueStyle={{ fontSize: 28, fontWeight: 600 }}
            />
            {isNumeric(item.trend) ? (
              <Tooltip title={item.trendDescription ?? 'Comparado con la semana anterior'}>
                <Typography.Text style={{ color: trendColor(item.trend) }}>
                  <TrendIcon value={item.trend} /> {formatTrend(item.trend)} vs semana anterior
                </Typography.Text>
              </Tooltip>
            ) : (
              <Typography.Text style={{ color: trendColor(item.trend) }}>
                <TrendIcon value={item.trend} /> Sin variación registrada
              </Typography.Text>
            )}
            {item.progress !== undefined ? (
              <Progress percent={item.progress} strokeColor="#1677ff" size="small" style={{ marginTop: 12 }} />
            ) : (
              <div style={{ height: 8, marginTop: 12 }} />
            )}
          </Card>
        </Col>
      ))}
    </Row>
  </div>
)

const SpaceBetween = ({ title, subtitle }) => (
  <div style={{ marginBottom: 8 }}>
    <Typography.Text type="secondary" style={{ display: 'block' }}>
      {title}
    </Typography.Text>
    {subtitle && (
      <Typography.Text style={{ fontSize: 12, color: '#64748b' }}>{subtitle}</Typography.Text>
    )}
  </div>
)

export default KpiCards
