import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AdministrationWorkspace,
  MaintenanceWorkspace,
  OperatorWorkspace,
  ResearchWorkspace,
} from './workspaces'
import './App.css'

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const apiLabel = apiBaseUrl || 'same-origin /api'

const navigation = [
  { id: 'dashboard', label: 'Performance', role: 'Plant operator' },
  { id: 'operations', label: 'Plant operations', role: 'Plant operator' },
  { id: 'maintenance', label: 'Anomaly review', role: 'Maintenance engineer' },
  { id: 'administration', label: 'Administration', role: 'System administrator' },
  { id: 'research', label: 'Research', role: 'Researcher' },
]

function formatMetric(value) {
  return typeof value === 'number' ? value.toFixed(3) : '—'
}

function formatTimestamp(value) {
  if (!value) return '—'

  const date = new Date(`${value}Z`)
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString()
}

function runLabel(run) {
  return `${run.experiment} · ${run.status} · ${formatTimestamp(run.started_at)}`
}

function RunPicker({ loading, runs, selectedRunId, onChange, selectedRun }) {
  return (
    <section className="run-picker" aria-label="Experiment run selector">
      <div>
        <p className="eyebrow">Selected experiment</p>
        <h2>Choose a saved federated run</h2>
      </div>

      <div className="run-picker-control">
        <label htmlFor="experiment-run">Experiment run</label>
        <select
          id="experiment-run"
          value={selectedRunId}
          onChange={(event) => onChange(event.target.value)}
          disabled={loading || !runs.length}
        >
          {!runs.length && (
            <option value="">Legacy results without a tracked run</option>
          )}

          {runs.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {runLabel(run)}
            </option>
          ))}
        </select>
      </div>

      {selectedRun ? (
        <p className="run-detail">
          {selectedRun.result_count} saved result
          {selectedRun.result_count === 1 ? '' : 's'} · seed{' '}
          {selectedRun.config.seed ?? '—'} · rounds{' '}
          {selectedRun.config.rounds ?? '—'}
        </p>
      ) : (
        <p className="run-detail">
          Showing older rows that were saved before run tracking existed.
        </p>
      )}
    </section>
  )
}

function MetricSummary({ plants, macroMetrics }) {
  return (
    <section className="metric-grid" aria-label="Federated experiment summary">
      <article className="metric-card">
        <p>Evaluated plants</p>
        <strong>{plants.length}</strong>
        <span>Latest saved result per plant</span>
      </article>

      <article className="metric-card">
        <p>Macro F1</p>
        <strong>{formatMetric(macroMetrics.f1)}</strong>
        <span>Average across evaluated plants</span>
      </article>

      <article className="metric-card">
        <p>Macro AUC-ROC</p>
        <strong>{formatMetric(macroMetrics.aucRoc)}</strong>
        <span>Single-class results are excluded</span>
      </article>

      <article className="metric-card">
        <p>Backend address</p>
        <strong className="api-address">{apiLabel}</strong>
        <span>Configured at build time</span>
      </article>
    </section>
  )
}

function ResultsTable({ plants }) {
  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Plant</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>AUC-ROC</th>
            <th>Threshold</th>
            <th>Saved</th>
          </tr>
        </thead>
        <tbody>
          {plants.map((plant) => (
            <tr key={plant.id}>
              <td>{plant.client_id}</td>
              <td>{formatMetric(plant.metrics.precision)}</td>
              <td>{formatMetric(plant.metrics.recall)}</td>
              <td>{formatMetric(plant.metrics.f1)}</td>
              <td>{formatMetric(plant.metrics.auc_roc)}</td>
              <td>{formatMetric(plant.metrics.threshold)}</td>
              <td>{formatTimestamp(plant.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PerformanceDashboard({ error, loading, macroMetrics, plants }) {
  return (
    <>
      <MetricSummary plants={plants} macroMetrics={macroMetrics} />

      <section className="results-panel">
        <div className="panel-heading">
          <div>
            <h2>Plant evaluation results</h2>
            <p>Each row belongs to the selected experiment run.</p>
          </div>
        </div>

        {loading && <p className="state-message">Loading saved experiment results…</p>}

        {!loading && error && (
          <div className="error-message">
            <strong>Could not load the dashboard.</strong>
            <span>{error}</span>
          </div>
        )}

        {!loading && !error && plants.length === 0 && (
          <div className="empty-message">
            <strong>No federated evaluation has been saved yet.</strong>
            <span>Run a federated experiment with <code>--evaluate</code>, then refresh this page.</span>
          </div>
        )}

        {!loading && !error && plants.length > 0 && <ResultsTable plants={plants} />}
      </section>
    </>
  )
}

function App() {
  const [plants, setPlants] = useState([])
  const [runs, setRuns] = useState([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [activePage, setActivePage] = useState('dashboard')
  const [apiStatus, setApiStatus] = useState('checking')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  const loadDashboard = useCallback(async (requestedRunId = '') => {
    setLoading(true)
    setError('')

    try {
      const [healthResponse, runsResponse] = await Promise.all([
        fetch(`${apiBaseUrl}/api/health`),
        fetch(`${apiBaseUrl}/api/runs?experiment=federated`),
      ])

      if (!healthResponse.ok || !runsResponse.ok) {
        throw new Error('The HydroFed API could not return dashboard data.')
      }

      const runData = await runsResponse.json()
      const availableRuns = runData.runs
      const selectedRunStillExists = availableRuns.some(
        (run) => run.run_id === requestedRunId,
      )
      const defaultRun = availableRuns.find(
        (run) => run.status === 'completed',
      ) || availableRuns[0]
      const nextRunId = selectedRunStillExists
        ? requestedRunId
        : defaultRun?.run_id || ''

      const query = new URLSearchParams({
        experiment: 'federated',
        kind: 'evaluation',
      })

      if (nextRunId) query.set('run_id', nextRunId)

      const resultsResponse = await fetch(
        `${apiBaseUrl}/api/results?${query.toString()}`,
      )

      if (!resultsResponse.ok) {
        throw new Error('The HydroFed API could not return evaluation results.')
      }

      const resultData = await resultsResponse.json()
      const latestByPlant = new Map()

      for (const result of resultData.results) {
        if (result.client_id && !latestByPlant.has(result.client_id)) {
          latestByPlant.set(result.client_id, result)
        }
      }

      setRuns(availableRuns)
      setSelectedRunId(nextRunId)
      setPlants(
        [...latestByPlant.values()].sort((first, second) =>
          first.client_id.localeCompare(second.client_id),
        ),
      )
      setApiStatus('connected')
    } catch (requestError) {
      setApiStatus('unavailable')
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const refreshTimer = window.setTimeout(() => {
      loadDashboard()
    }, 0)

    return () => window.clearTimeout(refreshTimer)
  }, [loadDashboard])

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId),
    [runs, selectedRunId],
  )

  const macroMetrics = useMemo(() => {
    const average = (metricName) => {
      const values = plants
        .map((plant) => plant.metrics[metricName])
        .filter((value) => typeof value === 'number')

      if (!values.length) return null
      return values.reduce((sum, value) => sum + value, 0) / values.length
    }

    return { f1: average('f1'), aucRoc: average('auc_roc') }
  }, [plants])

  const pageProps = {
    plants,
    runs,
    selectedRun,
    macroMetrics,
    formatMetric,
    formatTimestamp,
  }

  const pageContent = {
    dashboard: <PerformanceDashboard error={error} loading={loading} {...pageProps} />,
    operations: <OperatorWorkspace {...pageProps} />,
    maintenance: <MaintenanceWorkspace {...pageProps} />,
    administration: <AdministrationWorkspace {...pageProps} />,
    research: <ResearchWorkspace {...pageProps} />,
  }

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <div>
          <p className="eyebrow">HydroFed monitoring</p>
          <h1>Federated anomaly detection</h1>
          <p className="subtitle">Operational workspaces for plant, maintenance, administrative, and research users.</p>
        </div>

        <div className="header-actions">
          <span className={`connection connection-${apiStatus}`}>
            <span className="connection-dot" />
            API {apiStatus}
          </span>
          <button type="button" onClick={() => loadDashboard(selectedRunId)} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh data'}
          </button>
        </div>
      </header>

      <nav className="workspace-nav" aria-label="HydroFed workspaces">
        {navigation.map((item) => (
          <button
            key={item.id}
            type="button"
            className={activePage === item.id ? 'active' : ''}
            onClick={() => setActivePage(item.id)}
          >
            <span>{item.label}</span>
            <small>{item.role}</small>
          </button>
        ))}
      </nav>

      <RunPicker
        loading={loading}
        runs={runs}
        selectedRun={selectedRun}
        selectedRunId={selectedRunId}
        onChange={loadDashboard}
      />

      {pageContent[activePage]}
    </main>
  )
}

export default App
