import { Card, Empty, Segmented, Timeline } from 'antd'
import { useMemo, useState } from 'react'

const ActivityTimeline = ({ activities }) => {
  const [filter, setFilter] = useState('all')

  const filteredActivities = useMemo(() => {
    if (filter === 'all') return activities
    return activities.filter((activity) => activity.type === filter)
  }, [activities, filter])

  return (
    <Card
      title="Actividad reciente"
      className="dashboard-section"
      styles={{ body: { padding: 16 } }}
      style={{ height: '100%', width: '100%' }}
      extra={
        <Segmented
          size="small"
          options={[
            { label: 'Todo', value: 'all' },
            { label: 'Traducciones', value: 'translation' },
            { label: 'Evaluación', value: 'evaluation' },
            { label: 'Alertas', value: 'alert' },
          ]}
          value={filter}
          onChange={setFilter}
        />
      }
    >
      {filteredActivities.length === 0 ? (
        <Empty description="Sin actividad registrada" image={Empty.PRESENTED_IMAGE_SIMPLE} />
      ) : (
        <Timeline
          mode="left"
          items={filteredActivities.map((activity) => ({
            color: activity.color,
            children: (
              <div>
                <strong>{activity.title}</strong>
                <p style={{ marginBottom: 0 }}>{activity.description}</p>
                <small style={{ color: '#94a3b8' }}>{activity.timestamp}</small>
              </div>
            ),
          }))}
        />
      )}
    </Card>
  )
}

export default ActivityTimeline
