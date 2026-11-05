import { Icon } from '../components/Icon'
import { Sparkline } from '../components/charts'
import {
  Button, Card, EmptyState, Loading, Notice, PageHeader, ReviewBadge, ScoreMeter, SeverityBadge, StatTile, StatusBadge,
} from '../components/ui'
import { useApi } from '../lib/hooks'
import { clock, duration, faultTypes, number, relative, sensorValue, severityInfo } from '../lib/format'
import { navigate } from '../lib/router'

const keySensors = ['current', 'pressure', 'flow_rate', 'temperature']
const REFRESH_MS = 5000

export function LiveIndicator({ time }) {
  return (
    <span className="live">
      <span className="live-dot" aria-hidden="true" />
      Live replay · {clock(time)}
    </span>
  )
}

function PlantCard({ plant, sensors }) {
  const sensorMeta = Object.fromEntries(sensors.map((sensor) => [sensor.key, sensor]))
  return (
    <a className={`plant-card plant-${plant.status}`} href={`#/plants/${plant.id}`}>
      <header>
        <div>
          <h3>{plant.name}</h3>
          <span>{plant.location || `Node ${plant.id}`}{plant.capacity_mw ? ` · ${plant.capacity_mw} MW` : ''}</span>
        </div>
        <StatusBadge status={plant.status} />
      </header>

      <div className="plant-score">
        <div>
          <span className="metric-label">Anomaly score</span>
          <strong>{number(plant.score, 1)}</strong>
          <span className="metric-unit">/ {number(plant.threshold, 0)} threshold</span>
        </div>
        <Sparkline values={plant.score_trend} threshold={plant.threshold} label={`${plant.name} anomaly score, last 5 minutes`} />
      </div>
      <ScoreMeter score={plant.score} threshold={plant.threshold} />

      <dl className="plant-sensors">
        {keySensors.map((key) => (
          <div key={key}>
            <dt>{sensorMeta[key]?.label}</dt>
            <dd>{sensorValue(plant.sensors[key], sensorMeta[key]?.unit)}</dd>
          </div>
        ))}
      </dl>

      <footer>
        {plant.open_alerts ? (
          <span className="plant-alerts">
            <SeverityBadge severity={plant.highest_open_severity} size="sm" />
            {plant.open_alerts} open alert{plant.open_alerts === 1 ? '' : 's'}
          </span>
        ) : (
          <span className="plant-alerts muted">No open alerts</span>
        )}
        <span className="plant-link">Details <Icon name="arrowRight" size={14} /></span>
      </footer>
    </a>
  )
}

export function AlertRow({ alert, onOpen, showPlant = true }) {
  return (
    <li className={`alert-row alert-${alert.severity}`}>
      <button type="button" onClick={() => onOpen(alert)}>
        <span className="alert-severity-bar" aria-hidden="true" />
        <span className="alert-main">
          <span className="alert-title">
            <SeverityBadge severity={alert.severity} size="sm" />
            {showPlant && <strong>{alert.plant_name}</strong>}
            {alert.ongoing && <span className="pill pill-live">Ongoing</span>}
          </span>
          <span className="alert-meta">
            {faultTypes[alert.fault_type] || alert.fault_type} pattern · largest change in {alert.sensor_label || alert.sensor}
          </span>
        </span>
        <span className="alert-side">
          <ReviewBadge status={alert.status} size="sm" />
          <span className="alert-time">{relative(alert.started_at)} · {duration(alert.duration_seconds)}</span>
        </span>
      </button>
    </li>
  )
}

export default function FleetPage() {
  const fleet = useApi('/api/fleet', { interval: REFRESH_MS })
  const alerts = useApi('/api/alerts?limit=50', { interval: REFRESH_MS })

  if (fleet.loading && !fleet.data) return <Loading label="Loading the fleet…" />
  if (fleet.error && !fleet.data) {
    return (
      <EmptyState icon="alert" title="The fleet could not be loaded" action={<Button onClick={() => fleet.reload()}>Try again</Button>}>
        {fleet.error}
      </EmptyState>
    )
  }

  const { plants, sensors, source } = fleet.data
  const sensorLabels = Object.fromEntries(sensors.map((sensor) => [sensor.key, sensor.label.toLowerCase()]))
  const counts = plants.reduce((total, plant) => ({ ...total, [plant.status]: (total[plant.status] || 0) + 1 }), {})
  const openAlerts = (alerts.data?.alerts || [])
    .filter((alert) => ['open', 'acknowledged', 'investigating'].includes(alert.status))
    .sort((a, b) => severityInfo[b.severity].rank - severityInfo[a.severity].rank || b.started_at - a.started_at)
  const unreviewed = openAlerts.filter((alert) => alert.status === 'open')
  const critical = openAlerts.filter((alert) => alert.severity === 'critical').length

  return (
    <div className="page">
      <PageHeader
        eyebrow={<LiveIndicator time={plants[0]?.updated_at} />}
        title="Fleet overview"
        description="Anomaly scores compare each plant with its own recent normal behaviour. A score above the threshold raises an alert."
      />

      {fleet.error && <Notice tone="warning" title="Connection interrupted">{fleet.error} Showing the last readings received.</Notice>}

      <div className="stat-grid">
        <StatTile icon="octagon" label="Plants in alarm" value={counts.alarm || 0} tone={counts.alarm ? 'critical' : undefined} detail={`of ${plants.length} plants`} />
        <StatTile icon="eye" label="On watch" value={counts.watch || 0} tone={counts.watch ? 'watch' : undefined} detail="above half the threshold" />
        <StatTile icon="checkCircle" label="Normal" value={counts.normal || 0} detail="within normal behaviour" />
        <StatTile icon="alert" label="Unreviewed alerts" value={unreviewed.length} tone={unreviewed.length ? 'serious' : undefined} detail={`${critical} critical in the open queue`} />
      </div>

      <div className="fleet-layout">
        <section aria-label="Plants" className="plant-grid">
          {plants.map((plant) => <PlantCard key={plant.id} plant={plant} sensors={sensors} />)}
        </section>

        <Card
          title="Open alerts"
          subtitle="Highest severity first"
          className="fleet-alerts"
          actions={<Button size="sm" variant="ghost" onClick={() => navigate('/alerts')}>View all</Button>}
          padded={false}
        >
          {!openAlerts.length ? (
            <EmptyState icon="checkCircle" title="No open alerts">Every alert so far has been resolved or dismissed.</EmptyState>
          ) : (
            <ul className="alert-list">
              {openAlerts.slice(0, 8).map((alert) => (
                <AlertRow
                  key={alert.id}
                  alert={{ ...alert, sensor_label: sensorLabels[alert.sensor] }}
                  onOpen={() => navigate(`/alerts?alert=${encodeURIComponent(alert.id)}`)}
                />
              ))}
            </ul>
          )}
        </Card>
      </div>

      <p className="data-source">
        <Icon name="database" size={14} />
        Readings replay the {source.dataset} test-rig recordings at one second per second, scored by the federated LSTM autoencoder
        with the frozen alert policy (threshold {source.threshold_mad} MAD above each recording's baseline).
      </p>
    </div>
  )
}
