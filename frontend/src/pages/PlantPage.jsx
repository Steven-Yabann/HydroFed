import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { Legend, TimeSeriesChart } from '../components/charts'
import {
  Button, Card, EmptyState, Loading, PageHeader, ScoreMeter, Segmented, StatTile, StatusBadge,
} from '../components/ui'
import { useApi } from '../lib/hooks'
import { duration, faultTypes, number, plantStatus, sensorValue } from '../lib/format'
import { navigate, useQueryParam } from '../lib/router'
import { AlertRow, LiveIndicator } from './FleetPage'

const windows = [
  { id: '300', label: '5 min' },
  { id: '900', label: '15 min' },
  { id: '1800', label: '30 min' },
  { id: '3600', label: '1 h' },
]

export default function PlantPage({ plantId }) {
  const [windowSeconds, setWindowSeconds] = useQueryParam('window', '900')
  const [showLabels, setShowLabels] = useState(false)
  const plant = useApi(`/api/plants/${plantId}`, { interval: 5000 })
  const telemetry = useApi(`/api/plants/${plantId}/telemetry?seconds=${windowSeconds}`, { interval: 5000 })
  const alerts = useApi(`/api/alerts?plant=${plantId}&limit=20`, { interval: 10000 })

  const sensorCharts = useMemo(() => {
    const data = telemetry.data
    if (!data) return []
    return data.sensors_meta.map((sensor) => ({
      ...sensor,
      values: data.sensors[sensor.key],
    }))
  }, [telemetry.data])

  if (plant.loading && !plant.data) return <Loading label="Loading plant…" />
  if (plant.error && !plant.data) {
    return (
      <EmptyState icon="alert" title="This plant could not be loaded" action={<Button onClick={() => navigate('/fleet')}>Back to the fleet</Button>}>
        {plant.error}
      </EmptyState>
    )
  }

  const current = plant.data
  const data = telemetry.data
  const alertsInView = data ? data.alert.filter(Boolean).length : 0
  const sensorLabels = Object.fromEntries((data?.sensors_meta || []).map((sensor) => [sensor.key, sensor.label.toLowerCase()]))

  return (
    <div className="page">
      <a className="back-link" href="#/fleet"><Icon name="arrowLeft" size={16} /> Fleet overview</a>
      <PageHeader
        eyebrow={<LiveIndicator time={current.updated_at} />}
        title={current.name}
        description={[current.location, current.capacity_mw ? `${current.capacity_mw} MW` : null, `Node ${current.id}`].filter(Boolean).join(' · ')}
        actions={<StatusBadge status={current.status} size="lg" />}
      />

      <div className="stat-grid">
        <article className={`stat-tile stat-score stat-${plantStatus[current.status]?.tone}`}>
          <div className="stat-label"><Icon name="signal" size={16} /><span>Anomaly score now</span></div>
          <strong>{number(current.score, 1)}</strong>
          <ScoreMeter score={current.score} threshold={current.threshold} />
          <span className="stat-detail">Threshold {number(current.threshold, 0)} · {plantStatus[current.status]?.description}</span>
        </article>
        <StatTile icon="alert" label="Open alerts" value={current.open_alerts} tone={current.open_alerts ? 'serious' : undefined} detail="awaiting review or in progress" />
        <StatTile icon="clock" label={`Alert time, last ${windows.find((item) => item.id === windowSeconds)?.label}`} value={duration(alertsInView)} detail={data ? `${Math.round((alertsInView / data.score.length) * 100)}% of the window` : ''} />
        <StatTile icon="wave" label="Replaying recording" value={current.recording} detail={faultTypes[current.recording.split('/')[0]] || 'SKAB test rig'} />
      </div>

      <Card
        title="Anomaly score"
        subtitle="Score in MAD units above this recording's normal baseline. Shaded periods raised an alert."
        actions={(
          <div className="toolbar">
            <label className="checkbox">
              <input type="checkbox" checked={showLabels} onChange={(event) => setShowLabels(event.target.checked)} />
              Show labelled faults
            </label>
            <Segmented label="Time window" size="sm" options={windows} value={windowSeconds} onChange={setWindowSeconds} />
          </div>
        )}
      >
        {!data ? <Loading /> : (
          <>
            <TimeSeriesChart
              times={data.time}
              series={[{ key: 'score', label: 'Anomaly score', values: data.score, color: 'var(--series-1)' }]}
              threshold={data.threshold}
              alerts={data.alert}
              labels={showLabels ? data.label : undefined}
              height={260}
              yLabel="Score (MAD)"
            />
            <Legend
              items={[
                { label: 'Anomaly score', color: 'var(--series-1)' },
                { label: 'Alert threshold', color: 'var(--ink-muted)', shape: 'dash' },
                { label: 'Alert raised', color: 'var(--critical-tint-strong)', shape: 'block' },
                ...(showLabels ? [{ label: 'Labelled fault (SKAB ground truth)', color: 'var(--label-band)', shape: 'block' }] : []),
              ]}
            />
          </>
        )}
      </Card>

      <div className="plant-layout">
        <Card title="Sensor trends" subtitle="Raw readings over the same window, in engineering units.">
          {!data ? <Loading /> : (
            <div className="sensor-grid">
              {sensorCharts.map((sensor) => (
                <div key={sensor.key} className="sensor-chart">
                  <div className="sensor-chart-head">
                    <span>{sensor.label}</span>
                    <strong>{sensorValue(sensor.values.at(-1), sensor.unit)}</strong>
                  </div>
                  <TimeSeriesChart
                    compact
                    times={data.time}
                    series={[{ key: sensor.key, label: sensor.label, values: sensor.values, color: 'var(--series-2)' }]}
                    alerts={data.alert}
                    unit={sensor.unit}
                    height={120}
                    formatValue={(value) => sensorValue(value)}
                  />
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card
          title="Alert history"
          subtitle="This replay cycle, newest first"
          padded={false}
          actions={<Button size="sm" variant="ghost" onClick={() => navigate(`/alerts?plant=${plantId}`)}>Review</Button>}
        >
          {!alerts.data?.alerts.length ? (
            <EmptyState icon="checkCircle" title="No alerts yet">This plant has not raised an alert in the current cycle.</EmptyState>
          ) : (
            <ul className="alert-list">
              {alerts.data.alerts.map((alert) => (
                <AlertRow
                  key={alert.id}
                  showPlant={false}
                  alert={{ ...alert, sensor_label: sensorLabels[alert.sensor] }}
                  onOpen={() => navigate(`/alerts?alert=${encodeURIComponent(alert.id)}`)}
                />
              ))}
            </ul>
          )}
        </Card>
      </div>
    </div>
  )
}
