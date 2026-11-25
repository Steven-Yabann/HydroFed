import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { BarChart } from '../components/charts'
import { Card, EmptyState, Loading, Notice, PageHeader, Segmented } from '../components/ui'
import { useApi } from '../lib/hooks'
import { bytes, integer, number, percent } from '../lib/format'
import { useQueryParam } from '../lib/router'

const regimeColors = {
  federated: 'var(--series-1)',
  centralised: 'var(--series-2)',
  isolated: 'var(--series-3)',
  fedprox: 'var(--series-4)',
  federated_finetune: 'var(--series-5)',
}
const modelColors = {
  lstm_autoencoder: 'var(--series-1)',
  transformer_autoencoder_d32: 'var(--series-2)',
  xgboost: 'var(--series-3)',
  federated_xgboost: 'var(--series-4)',
  transformer_autoencoder_d64: 'var(--series-5)',
}
const coreRegimes = ['federated', 'centralised', 'isolated']
const roleOptions = [
  { id: 'final', label: 'Held-out recordings' },
  { id: 'dev', label: 'Development recordings' },
]
const scoreOptions = [
  { id: 'full', label: 'Full detector' },
  { id: 'model_only', label: 'Model only' },
]
const sections = [
  { id: 'regimes', label: 'Training regimes' },
  { id: 'models', label: 'Model families' },
]

// The results this page shows; personalised runs feed the regimes toggle.
const shown = ['regimes', 'models']

const mean = (summary, metric) => summary?.[metric]?.mean
const sd = (summary, metric) => summary?.[metric]?.sd

function MetricCell({ summary, metric, format = (value) => number(value, 3) }) {
  const value = mean(summary, metric)
  if (typeof value !== 'number') return <td className="numeric">—</td>
  const spread = sd(summary, metric)
  return (
    <td className="numeric">
      {format(value)}
      {spread > 0 && <span className="sd"> ± {format(spread)}</span>}
    </td>
  )
}

function DetectionTable({ rows, scoreKey, colorFor }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th scope="col">Approach</th>
            <th scope="col" className="numeric">Within-recording AUC</th>
            <th scope="col" className="numeric">Recall</th>
            <th scope="col" className="numeric">False alarm rate</th>
            <th scope="col" className="numeric">Precision</th>
            <th scope="col" className="numeric">F1</th>
            <th scope="col" className="numeric">Faults detected</th>
            <th scope="col" className="numeric">Median delay</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const summary = row[scoreKey] || row.full
            return (
              <tr key={row.id}>
                <th scope="row">
                  <span className="series-key" style={{ background: colorFor(row.id) }} />
                  {row.label}
                  {row.note && <span className="cell-note">{row.note}</span>}
                </th>
                <MetricCell summary={summary} metric="macro_recording_auc_roc" />
                <MetricCell summary={summary} metric="recall" />
                <MetricCell summary={summary} metric="false_positive_rate" />
                <MetricCell summary={summary} metric="precision" />
                <MetricCell summary={summary} metric="f1" />
                <td className="numeric">
                  {typeof mean(summary, 'fault_events_detected') === 'number'
                    ? `${number(mean(summary, 'fault_events_detected'), 1)} / ${integer(mean(summary, 'fault_events'))}`
                    : '—'}
                </td>
                <MetricCell summary={summary} metric="median_detection_delay" format={(value) => `${value.toFixed(0)} s`} />
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Findings({ data, scoreKey }) {
  const rows = Object.fromEntries((data.regimes?.rows || []).map((row) => [row.id, row]))
  const auc = (id, key = 'full') => mean(rows[id]?.[key], 'macro_recording_auc_roc')
  const fedFull = auc('federated')
  const centralFull = auc('centralised')
  const fedModel = auc('federated', 'model_only')
  const isolatedModel = auc('isolated', 'model_only')

  return (
    <div className="findings">
      <article>
        <span className="finding-icon"><Icon name="shield" size={18} /></span>
        <div>
          <strong>Federated training matches centralised training</strong>
          <p>
            On the full detector, FedAvg reaches a within-recording AUC of {number(fedFull, 3)} against {number(centralFull, 3)} for
            pooling every plant's data on one server, a gap of {number(Math.abs(fedFull - centralFull), 3)}, and no raw telemetry leaves a plant.
          </p>
        </div>
      </article>
      <article>
        <span className="finding-icon finding-neutral"><Icon name="info" size={18} /></span>
        <div>
          <strong>It does not beat plants training alone on SKAB</strong>
          <p>
            Scored by the model alone, isolated plants reach {number(isolatedModel, 3)} against {number(fedModel, 3)} for FedAvg.
            Each SKAB plant already has thousands of normal windows, and the per-recording baseline does most of the detecting.
          </p>
        </div>
      </article>
      {scoreKey === 'model_only' && (
        <Notice tone="info">“Model only” scores use the autoencoder's reconstruction residuals, isolating what training contributes.</Notice>
      )}
    </div>
  )
}

function RegimesSection({ data, scoreKey }) {
  const [showPersonalised, setShowPersonalised] = useState(false)
  const personalised = data.scenarios?.scenarios.find((scenario) => scenario.id === 'baseline' && scenario.plants === 'All plants')
  const rows = data.regimes.rows.filter((row) => coreRegimes.includes(row.id))
  const extraRegimes = showPersonalised
    ? (personalised?.regimes || []).filter((row) => !coreRegimes.includes(row.id))
    : []

  const bars = [
    ...rows.map((row) => ({
      id: row.id,
      label: row.label,
      value: mean(row[scoreKey], 'macro_recording_auc_roc'),
      sd: sd(row[scoreKey], 'macro_recording_auc_roc'),
      color: regimeColors[row.id],
      note: row.moves_raw_data ? 'Moves every plant\'s raw data to one server' : row.collaborates ? 'Shares model weights only' : 'No collaboration',
    })),
    ...extraRegimes.map((row) => ({
      id: row.id,
      label: row.label,
      value: row[scoreKey === 'full' ? 'full' : 'model_only'],
      sd: 0,
      color: regimeColors[row.id],
      note: 'Personalised federated variant',
    })),
  ].filter((row) => typeof row.value === 'number')

  return (
    <Card
      id="regimes"
      title="Federated vs centralised vs isolated training"
      subtitle={`The same LSTM autoencoder trained three ways: ${data.regimes.schedule.rounds} rounds × ${data.regimes.schedule.epochs} local epochs, seeds ${data.regimes.schedule.seeds.join(', ')}.`}
      actions={(
        <label className="checkbox">
          <input type="checkbox" checked={showPersonalised} onChange={(event) => setShowPersonalised(event.target.checked)} />
          Include FedProx and fine-tuning
        </label>
      )}
    >
      <h3 className="chart-title">Within-recording AUC (higher is better; 0.5 is guessing)</h3>
      <BarChart rows={bars} domain={[0.5, 1]} reference={personalised?.no_model_auc ?? undefined} />
      <DetectionTable rows={rows} scoreKey={scoreKey} colorFor={(id) => regimeColors[id]} />
      <p className="caption">
        Mean ± standard deviation over three seeds. Differences under about 0.02 are within run-to-run noise.
        Recall and false alarm rate are two ends of one trade-off, so read them together; F1 alone flatters a detector that alarms constantly.
      </p>
    </Card>
  )
}

function ModelsSection({ data, scoreKey }) {
  const [showAll, setShowAll] = useState(false)
  const selected = new Set(Object.values(data.models.selected))
  const rows = data.models.rows.filter((row) => showAll || selected.has(row.id))
  const scored = rows.map((row) => ({
    ...row,
    summary: row.model_only && scoreKey === 'model_only' ? row.model_only : row.full,
  }))
  const color = (id) => modelColors[id] || 'var(--series-other)'

  return (
    <Card
      id="models"
      title="Model families: LSTM vs transformer vs classical ML"
      subtitle="The best of each family, chosen on development recordings. Classical models score windows of summary statistics."
      actions={(
        <label className="checkbox">
          <input type="checkbox" checked={showAll} onChange={(event) => setShowAll(event.target.checked)} />
          Show every candidate
        </label>
      )}
    >
      <h3 className="chart-title">Within-recording AUC</h3>
      <BarChart
        rows={scored.map((row) => ({
          id: row.id,
          label: row.label,
          value: mean(row.summary, 'macro_recording_auc_roc'),
          sd: sd(row.summary, 'macro_recording_auc_roc'),
          color: color(row.id),
          note: row.needs_fault_labels ? 'Needs labelled faults to train' : 'Learns from normal data only',
        })).filter((row) => typeof row.value === 'number')}
        domain={[0.5, 1]}
      />

      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Training</th>
              <th scope="col">Fault labels</th>
              <th scope="col" className="numeric">Parameters</th>
              <th scope="col" className="numeric">Traffic per run</th>
              <th scope="col" className="numeric">Recall</th>
              <th scope="col" className="numeric">False alarms</th>
              <th scope="col" className="numeric">Median delay</th>
              <th scope="col" className="numeric">Unseen-fault AUC</th>
            </tr>
          </thead>
          <tbody>
            {scored.map((row) => (
              <tr key={row.id}>
                <th scope="row"><span className="series-key" style={{ background: color(row.id) }} />{row.label}</th>
                <td>{row.training}</td>
                <td>{row.needs_fault_labels ? 'Needed' : 'Not needed'}</td>
                <td className="numeric">{row.parameters ? integer(row.parameters) : '—'}</td>
                <td className="numeric">{row.communication_bytes ? bytes(row.communication_bytes) : 'Raw data pooled'}</td>
                <MetricCell summary={row.summary} metric="recall" />
                <MetricCell summary={row.summary} metric="false_positive_rate" />
                <MetricCell summary={row.summary} metric="median_detection_delay" format={(value) => `${value.toFixed(0)} s`} />
                <td className="numeric">{number(mean(row.unseen_fault_type, 'macro_recording_auc_roc'), 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="caption">
        Supervised models (XGBoost, Random Forest) need labelled faults, which a plant with no fault history does not have;
        the “unseen fault type” column tests them on a fault type left out of training. Autoencoders learn from normal operation alone.
      </p>
    </Card>
  )
}

export default function PerformancePage() {
  const [role, setRole] = useQueryParam('recordings', 'final')
  const [scoreKey, setScoreKey] = useQueryParam('score', 'full')
  const comparison = useApi(`/api/comparisons?role=${role}`)
  const data = comparison.data

  const missing = useMemo(() => (data ? Object.entries(data.available).filter(([name, present]) => shown.includes(name) && !present).map(([name]) => name) : []), [data])

  return (
    <div className="page">
      <PageHeader
        eyebrow="Research results · SKAB benchmark"
        title="Model performance"
        description="How federated learning compares with centralised and isolated training, and which model family detects faults best."
      />

      <div className="sticky-controls">
        <nav className="section-nav" aria-label="Sections">
          {sections.map((section) => <a key={section.id} href={`#/performance?recordings=${role}&score=${scoreKey}`} onClick={(event) => {
            event.preventDefault()
            document.getElementById(section.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }}>{section.label}</a>)}
        </nav>
        <div className="toolbar">
          <Segmented label="Recordings" size="sm" options={roleOptions} value={role} onChange={setRole} />
          <Segmented label="Score" size="sm" options={scoreOptions} value={scoreKey} onChange={setScoreKey} />
        </div>
      </div>

      {comparison.loading && !data ? <Loading label="Loading comparison results…" /> : comparison.error ? (
        <EmptyState icon="alert" title="Results could not be loaded">{comparison.error}</EmptyState>
      ) : (
        <>
          {missing.length > 0 && (
            <Notice tone="warning" title="Some results have not been generated">
              Missing: {missing.join(', ')}. Run <code>scripts/run_comparison.py</code> to create them.
            </Notice>
          )}
          {role === 'dev' && (
            <Notice tone="info">Development recordings were used to choose settings, so these numbers are optimistic. Report the held-out recordings.</Notice>
          )}
          {data.regimes && <Findings data={data} scoreKey={scoreKey} />}
          {data.regimes && <RegimesSection data={data} scoreKey={scoreKey} />}
          {data.models && <ModelsSection data={data} scoreKey={scoreKey} />}
          <p className="data-source">
            <Icon name="database" size={14} />
            {percent(0.54, 0)} of held-out test windows are anomalous, so an always-on alarm would score F1 ≈ 0.70.
            Within-recording AUC is the primary measure because it does not depend on a threshold.
          </p>
        </>
      )}
    </div>
  )
}
