import { useEffect, useState } from 'react'

const REGIME_EXPERIMENT = 'regime_comparison'
const MODEL_EXPERIMENT = 'model_comparison'

const regimeNames = {
  federated: 'Federated (FedAvg)',
  centralised: 'Centralised',
  isolated: 'Isolated plants',
}

const modelNames = {
  lstm_autoencoder: 'LSTM autoencoder',
  transformer_autoencoder_d32: 'Transformer autoencoder (d=32)',
  transformer_autoencoder_d64: 'Transformer autoencoder (d=64)',
  federated_xgboost: 'XGBoost (federated bagging)',
  xgboost: 'XGBoost (centralised)',
  random_forest: 'Random Forest (centralised)',
  isolation_forest: 'Isolation Forest (centralised)',
}

// Fixed display order, so a model keeps its place whatever order rows arrive in.
const modelOrder = Object.keys(modelNames)

const trainingNames = {
  FedAvg: 'FedAvg',
  FedXgbBagging: 'Flower bagging',
  centralised: 'Centralised',
}

const roleOptions = [
  { id: 'final', label: 'Final recordings (held out)' },
  { id: 'dev', label: 'Development recordings' },
]

const scoreOptions = [
  { id: 'hybrid', label: 'Full detector' },
  { id: 'residual', label: 'Model only' },
]

const detectionCharts = [
  { metric: 'recall', title: 'Recall', note: 'Share of fault seconds that raised an alert. Higher is better.' },
  { metric: 'false_positive_rate', title: 'False alarm rate', note: 'Share of normal seconds that raised an alert. Lower is better.' },
  { metric: 'macro_recording_auc_roc', title: 'Within-recording AUC', note: 'Ranking quality inside each recording, independent of the threshold.' },
]

function formatValue(value, digits = 3) {
  return typeof value === 'number' ? value.toFixed(digits) : '—'
}

function formatSummary(summary, digits = 3) {
  if (!summary || typeof summary.mean !== 'number') return '—'
  return `${summary.mean.toFixed(digits)} ± ${(summary.sd ?? 0).toFixed(digits)}`
}

function formatMegabytes(bytes) {
  if (typeof bytes !== 'number') return '—'
  return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 2 : 1)} MB`
}

function formatEvents(metrics) {
  const detected = metrics?.fault_events_detected?.mean
  const total = metrics?.fault_events?.mean
  return typeof detected === 'number' && typeof total === 'number'
    ? `${detected.toFixed(1)} / ${total.toFixed(0)}`
    : '—'
}

function formatDelay(metrics) {
  const delay = metrics?.median_detection_delay?.mean
  return typeof delay === 'number' ? `${delay.toFixed(0)} s` : '—'
}

function useLatestComparison(apiBaseUrl, experiment, refreshKey) {
  const [state, setState] = useState({ status: 'loading', run: null, rows: [], error: '' })

  useEffect(() => {
    let cancelled = false

    async function load() {
      setState((previous) => ({ ...previous, status: 'loading', error: '' }))
      try {
        const runsResponse = await fetch(
          `${apiBaseUrl}/api/runs?${new URLSearchParams({ experiment, limit: '20' })}`,
        )
        if (!runsResponse.ok) throw new Error('The API could not list comparison runs.')

        const { runs } = await runsResponse.json()
        const run = runs.find((candidate) => candidate.status === 'completed')
        if (!run) {
          if (!cancelled) setState({ status: 'empty', run: null, rows: [], error: '' })
          return
        }

        const resultsResponse = await fetch(
          `${apiBaseUrl}/api/results?${new URLSearchParams({ run_id: run.run_id, limit: '1000' })}`,
        )
        if (!resultsResponse.ok) throw new Error('The API could not return comparison results.')

        const { results } = await resultsResponse.json()
        if (!cancelled) setState({ status: 'ready', run, rows: results, error: '' })
      } catch (error) {
        if (!cancelled) setState({ status: 'error', run: null, rows: [], error: error.message })
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [apiBaseUrl, experiment, refreshKey])

  return state
}

function findDetection(rows, clientId, score, role) {
  return rows.find(
    (row) =>
      row.kind === 'detection' &&
      row.client_id === clientId &&
      row.metrics.score === score &&
      row.metrics.role === role,
  )?.metrics
}

function SegmentedControl({ label, options, value, onChange }) {
  return (
    <div className="segmented-control" role="group" aria-label={label}>
      <span>{label}</span>
      <div>
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            className={value === option.id ? 'active' : ''}
            aria-pressed={value === option.id}
            onClick={() => onChange(option.id)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function Legend({ items }) {
  return (
    <ul className="chart-legend">
      {items.map((item) => (
        <li key={item.label}>
          <span className={`legend-swatch ${item.emphasis ? 'emphasis' : 'context'}`} aria-hidden="true" />
          {item.label}
        </li>
      ))}
    </ul>
  )
}

// Horizontal bars from a zero baseline. Every value is printed beside its bar
// and repeated in the table below, so the hover readout only adds detail.
function BarChart({ title, note, items, max, formatLabel }) {
  const scale = max || Math.max(...items.map((item) => item.value || 0), 1e-9)

  return (
    <figure className="bar-chart">
      <figcaption>
        <strong>{title}</strong>
        {note && <span>{note}</span>}
      </figcaption>
      <div className="bar-rows">
        {items.map((item) => {
          const width = typeof item.value === 'number' ? Math.max((item.value / scale) * 100, 0.5) : 0
          return (
            <div
              key={item.key}
              className="bar-row"
              tabIndex={0}
              aria-label={`${item.label}: ${formatLabel(item)}`}
            >
              <span className="bar-label">{item.label}</span>
              <span className="bar-track">
                <span
                  className={`bar-fill ${item.emphasis ? 'emphasis' : 'context'}`}
                  style={{ width: `${width}%` }}
                />
              </span>
              <span className="bar-value">{formatLabel(item)}</span>
              {item.detail && (
                <span className="bar-tooltip" role="tooltip">
                  <strong>{formatLabel(item)}</strong>
                  <span>{item.label}</span>
                  <span>{item.detail}</span>
                </span>
              )}
            </div>
          )
        })}
      </div>
    </figure>
  )
}

function DetectionCharts({ entries }) {
  return (
    <div className="chart-grid">
      {detectionCharts.map((chart) => (
        <BarChart
          key={chart.metric}
          title={chart.title}
          note={chart.note}
          max={1}
          formatLabel={(item) => formatSummary(item.summary)}
          items={entries.map((entry) => ({
            key: entry.key,
            label: entry.label,
            emphasis: entry.emphasis,
            summary: entry.metrics?.[chart.metric],
            value: entry.metrics?.[chart.metric]?.mean,
            detail: `${formatEvents(entry.metrics)} faults detected · median delay ${formatDelay(entry.metrics)}`,
          }))}
        />
      ))}
    </div>
  )
}

function ComparisonState({ state, command }) {
  if (state.status === 'loading') {
    return <p className="state-message">Loading saved comparison results…</p>
  }
  if (state.status === 'error') {
    return (
      <div className="error-message">
        <strong>Could not load this comparison.</strong>
        <span>{state.error}</span>
      </div>
    )
  }
  return (
    <div className="empty-message">
      <strong>No saved comparison yet.</strong>
      <span>
        Run <code>{command}</code>, then upload the database the API uses.
      </span>
    </div>
  )
}

function RunCaption({ run, roundsLabel = 'federated rounds' }) {
  const seeds = run.config.seeds?.join(', ') ?? '—'
  return (
    <p className="comparison-caption">
      Mean ± standard deviation over seeds {seeds} · {run.config.rounds ?? '—'} {roundsLabel} ·
      saved {new Date(`${run.finished_at || run.started_at}Z`).toLocaleDateString()}
    </p>
  )
}

function RegimeComparison({ state }) {
  const [score, setScore] = useState('hybrid')
  const [role, setRole] = useState('final')

  if (state.status !== 'ready') {
    return <ComparisonState state={state} command="scripts/run_comparison.py regimes --db artifacts/experiments.db" />
  }

  const { rows, run } = state
  const entries = Object.keys(regimeNames)
    .map((regime) => ({
      key: regime,
      label: regimeNames[regime],
      emphasis: regime === 'federated',
      metrics: findDetection(rows, regime, score, role),
    }))
    .filter((entry) => entry.metrics)

  const raw = rows.find((row) => row.kind === 'raw_data')?.metrics
  const traffic = rows
    .filter((row) => row.kind === 'communication')
    .map((row) => row.metrics)
    .sort((first, second) => first.rounds - second.rounds)
  const fullRun = traffic[traffic.length - 1]
  const roundEntries = traffic.map((item) => ({
    ...item,
    metrics: findDetection(
      rows,
      item.rounds === run.config.rounds ? 'federated' : `federated_${item.rounds}r`,
      score,
      role,
    ),
  }))

  const byteItems = [
    ...[...traffic].reverse().map((item) => ({
      key: `federated-${item.rounds}`,
      label: `Federated, ${item.rounds} rounds`,
      value: item.total_bytes,
      emphasis: true,
      detail: `${formatMegabytes(item.bytes_per_round)} per round · ${formatMegabytes(item.bytes_per_client)} per plant`,
    })),
    raw && { key: 'csv', label: 'Send all SKAB CSV files', value: raw.csv_bytes, detail: `${raw.csv_files} files as stored` },
    raw && { key: 'binary', label: 'Send all sensor rows', value: raw.binary_bytes, detail: `${raw.rows.toLocaleString()} rows × ${raw.sensors} sensors, float32` },
    raw && { key: 'baseline', label: 'Send baseline rows only', value: raw.baseline_binary_bytes, detail: 'What unsupervised training uses' },
  ].filter(Boolean)

  return (
    <>
      <RunCaption run={run} />

      <div className="comparison-controls">
        <SegmentedControl label="Score" options={scoreOptions} value={score} onChange={setScore} />
        <SegmentedControl label="Recordings" options={roleOptions} value={role} onChange={setRole} />
      </div>

      <Legend items={[{ label: 'Federated learning', emphasis: true }, { label: 'Alternatives that share data or train alone' }]} />
      <DetectionCharts entries={entries} />

      <div className="information-banner">
        <strong>How to read this</strong>
        <span>
          The full detector also measures how far sensor levels sit from each recording&apos;s baseline, which does not
          depend on the trained model. Choose “Model only” to compare what each training approach learned.
        </span>
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Training</th>
              <th>Recall</th>
              <th>Precision</th>
              <th>F1</th>
              <th>False alarm rate</th>
              <th>AUC</th>
              <th>Faults detected</th>
              <th>Median delay</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={entry.key}>
                <td>{entry.label}</td>
                <td>{formatSummary(entry.metrics.recall)}</td>
                <td>{formatSummary(entry.metrics.precision)}</td>
                <td>{formatSummary(entry.metrics.f1)}</td>
                <td>{formatSummary(entry.metrics.false_positive_rate)}</td>
                <td>{formatSummary(entry.metrics.macro_recording_auc_roc)}</td>
                <td>{formatEvents(entry.metrics)}</td>
                <td>{formatDelay(entry.metrics)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="comparison-subheading">Communication overhead</h3>

      {raw && fullRun && (
        <div className="workspace-stat-grid">
          <article>
            <span>Model weights</span>
            <strong>{formatMegabytes(raw.model_bytes)}</strong>
            <span>Sent up and down by each plant every round</span>
          </article>
          <article>
            <span>Federated training, {fullRun.rounds} rounds</span>
            <strong>{formatMegabytes(fullRun.total_bytes)}</strong>
            <span>{formatMegabytes(fullRun.bytes_per_round)} per round across {raw.clients} plants</span>
          </article>
          <article>
            <span>Break-even against streaming</span>
            <strong>{formatValue(fullRun.break_even_days_1hz, 1)} days</strong>
            <span>Of one plant streaming {raw.sensors} sensors at 1 Hz</span>
          </article>
        </div>
      )}

      <Legend items={[{ label: 'Federated weight traffic', emphasis: true }, { label: 'Sending raw data to a server' }]} />
      <BarChart
        title="Bytes moved to train once"
        note="SKAB is about 13 hours of data, so sending it is cheaper than one federated run."
        items={byteItems}
        formatLabel={(item) => formatMegabytes(item.value)}
      />

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Federated rounds</th>
              <th>Total traffic</th>
              <th>Per plant</th>
              <th>× SKAB CSV files</th>
              <th>Break-even at 1 Hz</th>
              <th>Recall</th>
              <th>False alarm rate</th>
              <th>AUC</th>
            </tr>
          </thead>
          <tbody>
            {roundEntries.map((item) => (
              <tr key={item.rounds}>
                <td>{item.rounds}</td>
                <td>{formatMegabytes(item.total_bytes)}</td>
                <td>{formatMegabytes(item.bytes_per_client)}</td>
                <td>{formatValue(item.multiple_of_csv, 0)}×</td>
                <td>{formatValue(item.break_even_days_1hz, 1)} days</td>
                <td>{formatSummary(item.metrics?.recall)}</td>
                <td>{formatSummary(item.metrics?.false_positive_rate)}</td>
                <td>{formatSummary(item.metrics?.macro_recording_auc_roc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="research-note">
        Byte counts cover the exchanged weight arrays, not gRPC framing. A plant that retrains less often than the
        break-even interval moves less data with federated learning than by streaming its raw telemetry.
      </p>
    </>
  )
}

function modelSize(candidate) {
  if (typeof candidate.parameters === 'number') return `${candidate.parameters.toLocaleString()} parameters`
  if (typeof candidate.trees === 'number') {
    return `${candidate.trees} trees from ${candidate.clients} plants, ${candidate.rounds} rounds`
  }
  return '—'
}

function communicationLabel(entry) {
  if (entry.separateTraining) return '—'
  if (entry.candidate.communication_bytes) return formatMegabytes(entry.candidate.communication_bytes)
  return 'Raw data sent to a server'
}

function ModelComparison({ state }) {
  const [score, setScore] = useState('hybrid')
  const [role, setRole] = useState('final')

  if (state.status !== 'ready') {
    return <ComparisonState state={state} command="scripts/run_comparison.py models --db artifacts/experiments.db" />
  }

  const { rows, run } = state
  const candidates = Object.fromEntries(
    rows.filter((row) => row.kind === 'candidate').map((row) => [row.client_id, row.metrics]),
  )

  // The federated models: the chosen autoencoder of each kind, plus XGBoost
  // trained with Flower's bagging. Centralised XGBoost is shown for reference.
  const federatedNames = Object.keys(candidates)
    .filter(
      (name) =>
        (candidates[name].training === 'FedAvg' && candidates[name].selected) ||
        candidates[name].training === 'FedXgbBagging',
    )
    .sort((first, second) => modelOrder.indexOf(first) - modelOrder.indexOf(second))
  const entryFor = (name, entryRole, extra = {}) => {
    const candidate = candidates[name]
    const entryScore = candidate.selection_score === 'residual' ? score : 'model'
    return {
      key: `${name}-${entryRole}`,
      name,
      label: modelNames[name] ?? name,
      candidate,
      metrics: findDetection(rows, name, entryScore, entryRole),
      ...extra,
    }
  }

  const entries = federatedNames.map((name) => entryFor(name, role, { emphasis: true }))
  const references = []
  if (candidates.xgboost) {
    references.push(entryFor('xgboost', role, { emphasis: false }))
  }
  if (candidates.federated_xgboost && role === 'final') {
    references.push(entryFor('federated_xgboost', 'final_unseen_fault_type', {
      emphasis: false,
      label: 'XGBoost (federated), unseen fault type',
      // Trained separately for each held-out fault type, so the main run's cost does not apply.
      separateTraining: true,
    }))
  }
  const allEntries = [...entries, ...references].filter((entry) => entry.metrics)

  return (
    <>
      <RunCaption run={run} roundsLabel="FedAvg rounds for the autoencoders" />

      <div className="comparison-controls">
        <SegmentedControl label="Autoencoder score" options={scoreOptions} value={score} onChange={setScore} />
        <SegmentedControl label="Recordings" options={roleOptions} value={role} onChange={setRole} />
      </div>

      <Legend items={[{ label: 'Trained with federated learning', emphasis: true }, { label: 'Reference: centralised, or a fault type not seen in training' }]} />
      <DetectionCharts entries={allEntries} />

      <div className="information-banner">
        <strong>Labels matter</strong>
        <span>
          XGBoost learns from labelled faults, and the final recordings contain the same fault types, so it has an
          advantage the autoencoders do not. The unseen fault type row trains without the fault being tested. Plants
          with no labelled faults cannot add trees to federated XGBoost.
        </span>
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Training</th>
              <th>Needs fault labels</th>
              <th>Size</th>
              <th>Communication</th>
              <th>Recall</th>
              <th>False alarm rate</th>
              <th>AUC</th>
              <th>Faults detected</th>
              <th>Median delay</th>
            </tr>
          </thead>
          <tbody>
            {allEntries.map((entry) => (
              <tr key={entry.key}>
                <td>{entry.label}</td>
                <td>{trainingNames[entry.candidate.training] ?? entry.candidate.training ?? '—'}</td>
                <td>{entry.candidate.needs_fault_labels ? 'Yes' : 'No'}</td>
                <td>{entry.separateTraining ? '—' : modelSize(entry.candidate)}</td>
                <td>{communicationLabel(entry)}</td>
                <td>{formatSummary(entry.metrics.recall)}</td>
                <td>{formatSummary(entry.metrics.false_positive_rate)}</td>
                <td>{formatSummary(entry.metrics.macro_recording_auc_roc)}</td>
                <td>{formatEvents(entry.metrics)}</td>
                <td>{formatDelay(entry.metrics)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="research-note">
        Each model&apos;s alert threshold was chosen on development recordings: the highest recall with at most 15% false
        alarms. Final recordings were not used to choose anything.
      </p>
    </>
  )
}

export function ComparisonWorkspace({ apiBaseUrl, refreshKey }) {
  const regimes = useLatestComparison(apiBaseUrl, REGIME_EXPERIMENT, refreshKey)
  const models = useLatestComparison(apiBaseUrl, MODEL_EXPERIMENT, refreshKey)

  return (
    <>
      <section className="workspace-panel comparison-panel">
        <div className="workspace-heading">
          <p className="eyebrow">Research comparison</p>
          <h2>Federated against centralised training</h2>
          <p>
            The same LSTM autoencoder trained three ways, with the same inputs, schedule and alert policy, on SKAB
            recordings.
          </p>
        </div>
        <RegimeComparison state={regimes} />
      </section>

      <section className="workspace-panel comparison-panel">
        <div className="workspace-heading">
          <p className="eyebrow">Research comparison</p>
          <h2>LSTM, transformer and XGBoost under federated learning</h2>
          <p>
            The autoencoders train with FedAvg. XGBoost trees cannot be averaged, so plants grow trees locally and the
            server appends them, as in Flower&apos;s bagging strategy.
          </p>
        </div>
        <ModelComparison state={models} />
      </section>
    </>
  )
}
