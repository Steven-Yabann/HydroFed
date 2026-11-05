import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import {
  Button, Card, Drawer, EmptyState, Field, Loading, Notice, PageHeader, ReviewBadge, Segmented, SeverityBadge,
} from '../components/ui'
import { api } from '../lib/api'
import { useApi } from '../lib/hooks'
import { clock, dateTime, duration, faultTypes, number, relative, reviewInfo, severityInfo } from '../lib/format'
import { navigate, useQueryParam } from '../lib/router'

const views = [
  { id: 'active', label: 'Needs attention' },
  { id: 'closed', label: 'Closed' },
  { id: 'all', label: 'All' },
]
const activeStatuses = ['open', 'acknowledged', 'investigating']
const reviewOptions = [
  { id: 'acknowledged', description: 'Seen; the crew is aware.' },
  { id: 'investigating', description: 'Someone is checking the equipment.' },
  { id: 'resolved', description: 'A real fault, now dealt with.' },
  { id: 'false_alarm', description: 'Nothing was wrong; the alert was spurious.' },
  { id: 'open', description: 'Put the alert back in the queue.' },
]

function ReviewDrawer({ alert, sensorLabels, onClose, onSaved }) {
  const [status, setStatus] = useState(alert?.status === 'open' ? 'acknowledged' : alert?.status)
  const [note, setNote] = useState(alert?.note || '')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  const save = async () => {
    setSaving(true)
    try {
      const { alert: updated } = await api(`/api/alerts/${encodeURIComponent(alert.id)}/review`, {
        method: 'POST',
        body: { status, note },
      })
      onSaved(updated)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Drawer
      open={Boolean(alert)}
      onClose={onClose}
      title={`${severityInfo[alert.severity].label} alert · ${alert.plant_name}`}
      subtitle={`Started ${dateTime(alert.started_at)}`}
      footer={(
        <>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" icon="check" onClick={save} disabled={saving}>{saving ? 'Saving…' : 'Save review'}</Button>
        </>
      )}
    >
      <div className="badge-row">
        <SeverityBadge severity={alert.severity} />
        <ReviewBadge status={alert.status} />
        {alert.ongoing && <span className="pill pill-live">Ongoing</span>}
      </div>

      <dl className="detail-list">
        <div><dt>Plant</dt><dd><a href={`#/plants/${alert.plant}`}>{alert.plant_name}</a></dd></div>
        <div><dt>Started</dt><dd>{clock(alert.started_at)} ({relative(alert.started_at)})</dd></div>
        <div><dt>Ended</dt><dd>{alert.ongoing ? 'Still above threshold' : clock(alert.ended_at)}</dd></div>
        <div><dt>Duration</dt><dd>{duration(alert.duration_seconds)}</dd></div>
        <div><dt>Peak score</dt><dd>{alert.peak_score === null ? 'Still rising' : `${number(alert.peak_score, 1)} (threshold ${number(alert.threshold, 0)})`}</dd></div>
        <div><dt>Largest sensor change</dt><dd>{sensorLabels[alert.sensor] || alert.sensor}</dd></div>
        <div><dt>Pattern</dt><dd>{faultTypes[alert.fault_type] || alert.fault_type}</dd></div>
        <div>
          <dt>SKAB ground truth</dt>
          <dd>{alert.labelled_fault ? 'Overlaps a labelled fault' : 'No labelled fault (likely false alarm)'}</dd>
        </div>
      </dl>

      {alert.reviewed_by && (
        <p className="muted small">Last reviewed by {alert.reviewed_by}, {dateTime(alert.reviewed_at)}.</p>
      )}

      {error && <Notice tone="critical">{error}</Notice>}

      <fieldset className="radio-cards">
        <legend>Review status</legend>
        {reviewOptions.map((option) => (
          <label key={option.id} className={status === option.id ? 'selected' : ''}>
            <input type="radio" name="review-status" value={option.id} checked={status === option.id} onChange={() => setStatus(option.id)} />
            <span>
              <strong>{reviewInfo[option.id].label}</strong>
              <small>{option.description}</small>
            </span>
          </label>
        ))}
      </fieldset>

      <Field label="Note" hint="What was found or done. Visible to everyone reviewing alerts.">
        <textarea rows={4} value={note} maxLength={1000} onChange={(event) => setNote(event.target.value)} />
      </Field>

      <Button variant="ghost" icon="chart" onClick={() => navigate(`/plants/${alert.plant}`)}>Open the plant's trends</Button>
    </Drawer>
  )
}

export default function AlertsPage() {
  const [view, setView] = useQueryParam('view', 'active')
  const [plantFilter, setPlantFilter] = useQueryParam('plant', '')
  const [selectedId, setSelectedId] = useQueryParam('alert', '')
  const [severityFilter, setSeverityFilter] = useState('')
  const alerts = useApi('/api/alerts?limit=500', { interval: 10000 })
  const fleet = useApi('/api/fleet')

  const sensorLabels = useMemo(
    () => Object.fromEntries((fleet.data?.sensors || []).map((sensor) => [sensor.key, sensor.label])),
    [fleet.data],
  )
  const all = useMemo(() => alerts.data?.alerts || [], [alerts.data])
  const filtered = all.filter((alert) => {
    if (view === 'active' && !activeStatuses.includes(alert.status)) return false
    if (view === 'closed' && activeStatuses.includes(alert.status)) return false
    if (plantFilter && alert.plant !== plantFilter) return false
    if (severityFilter && alert.severity !== severityFilter) return false
    return true
  }).sort((a, b) => (view === 'active'
    ? severityInfo[b.severity].rank - severityInfo[a.severity].rank || b.started_at - a.started_at
    : b.started_at - a.started_at))
  const selected = all.find((alert) => alert.id === selectedId) || null
  const plants = fleet.data?.plants || []

  const counts = {
    open: all.filter((alert) => alert.status === 'open').length,
    inProgress: all.filter((alert) => ['acknowledged', 'investigating'].includes(alert.status)).length,
    falseAlarms: all.filter((alert) => alert.status === 'false_alarm').length,
  }

  return (
    <div className="page">
      <PageHeader
        title="Alerts"
        description="Review each alert so the team knows what has been seen, checked and resolved. Marking false alarms shows how often the detector cries wolf."
      />

      <div className="summary-strip">
        <span><b>{counts.open}</b> unreviewed</span>
        <span><b>{counts.inProgress}</b> in progress</span>
        <span><b>{counts.falseAlarms}</b> marked false alarm</span>
      </div>

      <Card padded={false}>
        <div className="filter-bar">
          <Segmented label="Alert view" options={views} value={view} onChange={setView} />
          <label className="select">
            <span className="sr-only">Plant</span>
            <select value={plantFilter} onChange={(event) => setPlantFilter(event.target.value)}>
              <option value="">All plants</option>
              {plants.map((plant) => <option key={plant.id} value={plant.id}>{plant.name}</option>)}
            </select>
          </label>
          <label className="select">
            <span className="sr-only">Severity</span>
            <select value={severityFilter} onChange={(event) => setSeverityFilter(event.target.value)}>
              <option value="">All severities</option>
              {Object.entries(severityInfo).reverse().map(([id, info]) => <option key={id} value={id}>{info.label}</option>)}
            </select>
          </label>
        </div>

        {alerts.loading && !alerts.data ? <Loading /> : alerts.error && !alerts.data ? (
          <EmptyState icon="alert" title="Alerts could not be loaded">{alerts.error}</EmptyState>
        ) : !filtered.length ? (
          <EmptyState icon="checkCircle" title={view === 'active' ? 'Nothing needs attention' : 'No alerts match these filters'}>
            {view === 'active' ? 'Every alert has been resolved or dismissed.' : 'Try another plant or severity.'}
          </EmptyState>
        ) : (
          <div className="table-scroll">
            <table className="data-table alert-table">
              <thead>
                <tr>
                  <th scope="col">Severity</th>
                  <th scope="col">Plant</th>
                  <th scope="col">Started</th>
                  <th scope="col">Duration</th>
                  <th scope="col">Largest change</th>
                  <th scope="col">Status</th>
                  <th scope="col"><span className="sr-only">Action</span></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((alert) => (
                  <tr key={alert.id} className={selectedId === alert.id ? 'selected' : ''}>
                    <td><SeverityBadge severity={alert.severity} size="sm" /></td>
                    <td>
                      <strong>{alert.plant_name}</strong>
                      {alert.ongoing && <span className="pill pill-live">Ongoing</span>}
                    </td>
                    <td><span title={dateTime(alert.started_at)}>{relative(alert.started_at)}</span></td>
                    <td className="numeric">{duration(alert.duration_seconds)}</td>
                    <td>{sensorLabels[alert.sensor] || alert.sensor}</td>
                    <td>
                      <ReviewBadge status={alert.status} size="sm" />
                      {alert.reviewed_by && <span className="cell-note">{alert.reviewed_by}</span>}
                    </td>
                    <td className="row-action">
                      <Button size="sm" variant={alert.status === 'open' ? 'primary' : 'secondary'} onClick={() => setSelectedId(alert.id)}>
                        {alert.status === 'open' ? 'Review' : 'Update'}
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <p className="data-source">
        <Icon name="info" size={14} />
        Severity follows the peak score: medium above the threshold, high above twice it, critical above four times it.
      </p>

      {selected && <ReviewDrawer
        key={selected.id}
        alert={selected}
        sensorLabels={sensorLabels}
        onClose={() => setSelectedId('')}
        onSaved={(updated) => {
          alerts.setData({ ...alerts.data, alerts: all.map((alert) => (alert.id === updated.id ? updated : alert)) })
          setSelectedId('')
        }}
      />}
    </div>
  )
}
