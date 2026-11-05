import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { BarChart, DotPlot, Legend, LineChart } from '../components/charts'
import { Card, EmptyState, Loading, Notice, PageHeader, Segmented, StatTile } from '../components/ui'
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
  { id: 'scenarios', label: 'Heterogeneous & scarce plants' },
  { id: 'models', label: 'Model families' },
  { id: 'communication', label: 'Communication overhead' },
]

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
  const twenty = data.communication?.federated.at(-1)

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
      {twenty && (
        <article>
          <span className="finding-icon"><Icon name="upload" size={18} /></span>
          <div>
            <strong>{bytes(twenty.bytes_per_client)} per plant for a {twenty.rounds}-round run</strong>
            <p>
              That equals {number(twenty.break_even_days_1hz, 1)} days of streaming eight sensors at 1 Hz to a central server,
              so federation moves less data whenever plants retrain less often than that.
            </p>
          </div>
        </article>
      )}
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

function ScenariosSection({ data, scoreKey }) {
  const scenarios = data.scenarios.scenarios
  const regimeIds = [...new Set(scenarios.flatMap((scenario) => scenario.regimes.map((regime) => regime.id)))]
  const series = regimeIds.map((id) => ({
    id,
    label: scenarios.flatMap((scenario) => scenario.regimes).find((regime) => regime.id === id).label,
    color: regimeColors[id],
  }))
  const rows = scenarios.map((scenario) => ({
    id: `${scenario.id}-${scenario.plants}`,
    label: scenario.plants === 'All plants' ? scenario.label : `${scenario.label} (${scenario.plants.replaceAll('plant_', 'P')})`,
    values: Object.fromEntries(scenario.regimes.map((regime) => [regime.id, regime[scoreKey]])),
  }))
  const values = rows.flatMap((row) => Object.values(row.values)).filter(Number.isFinite)
  const domain = [Math.floor(Math.min(...values) * 20) / 20, Math.min(1, Math.ceil(Math.max(...values) * 20) / 20)]

  const heterogeneous = scenarios.find((scenario) => scenario.id === 'heterogeneous')
  const plants = heterogeneous ? Object.keys(heterogeneous.regimes[0]?.model_only_by_plant || {}) : []
  const profiles = data.scenarios.profiles || {}

  return (
    <Card
      id="scenarios"
      title="Heterogeneous and data-scarce plants"
      subtitle="The two conditions federated learning is meant to help with: plants that record differently, and plants with little training data."
    >
      <Legend items={series} />
      <DotPlot rows={rows} series={series} domain={domain} />
      <p className="caption">
        Within-recording AUC on held-out recordings. Scarce rows train plants 4 and 5 on only the first 100, 50 or 25 seconds of each recording.
      </p>

      {heterogeneous && (
        <>
          <h3 className="chart-title">Model-only AUC per plant when plants differ</h3>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Approach</th>
                  {plants.map((plant) => (
                    <th key={plant} scope="col" className="numeric">
                      {plant.replace('_', ' ')}
                      <span className="cell-note">{profiles[plant]?.description}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {heterogeneous.regimes.map((regime) => (
                  <tr key={regime.id}>
                    <th scope="row"><span className="series-key" style={{ background: regimeColors[regime.id] }} />{regime.label}</th>
                    {plants.map((plant) => {
                      const value = regime.model_only_by_plant?.[plant]
                      const top = value === Math.max(...heterogeneous.regimes.map((item) => item.model_only_by_plant?.[plant] ?? 0))
                      return <td key={plant} className={`numeric ${top ? 'best' : ''}`}>{number(value, 3)}{top ? ' ★' : ''}</td>
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="caption">★ marks the best approach for each plant. One shared model fits unlike plants worse than their own models do.</p>
        </>
      )}
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

function CommunicationSection({ data }) {
  const communication = data.communication
  const runs = communication.federated
  const longest = runs.at(-1)
  const perDay = communication.streaming_bytes_per_plant_day_1hz
  const maxDays = Math.max(10, Math.ceil(longest.break_even_days_1hz * 1.6))

  const volumeRows = [
    ...runs.map((run) => ({
      id: `fl-${run.rounds}`,
      label: `Federated, ${run.rounds} rounds`,
      value: run.total_bytes,
      color: 'var(--series-1)',
      note: `${bytes(run.bytes_per_client)} per plant · ${number(run.multiple_of_csv, 1)}× the SKAB CSV files`,
    })),
    {
      id: 'csv',
      label: 'Centralise SKAB CSVs',
      value: communication.raw.csv_bytes,
      color: 'var(--series-2)',
      note: `${communication.raw.csv_files} files, ${integer(communication.raw.rows)} rows`,
    },
    {
      id: 'binary',
      label: 'Centralise as float32',
      value: communication.raw.binary_bytes,
      color: 'var(--series-2)',
      note: `${communication.raw.sensors} sensors × ${integer(communication.raw.rows)} rows × 4 bytes`,
    },
  ]

  const days = Array.from({ length: maxDays + 1 }, (_, day) => day)
  const lines = [
    {
      id: 'stream',
      label: 'Streaming raw data',
      color: 'var(--series-2)',
      points: days.map((day) => [day, perDay * day]),
    },
    ...runs.map((run, index) => ({
      id: `run-${run.rounds}`,
      label: `${run.rounds}-round run`,
      color: index === runs.length - 1 ? 'var(--series-1)' : `var(--series-1-${index === 0 ? 'light' : 'mid'})`,
      points: days.map((day) => [day, run.bytes_per_client]),
    })),
  ]

  return (
    <Card
      id="communication"
      title="Communication overhead"
      subtitle="Bytes counted for the weight arrays each plant uploads and the global model sent back, without gRPC framing."
    >
      <div className="stat-grid stat-grid-compact">
        <StatTile label="Model size" value={bytes(communication.model_bytes)} detail="LSTM autoencoder weights" />
        <StatTile label="Per round, whole fleet" value={bytes(longest.bytes_per_round)} detail={`${communication.clients} plants up and down`} />
        <StatTile label={`${longest.rounds}-round run, whole fleet`} value={bytes(longest.total_bytes)} detail={`${bytes(longest.uplink_bytes)} up · ${bytes(longest.downlink_bytes)} down`} />
        <StatTile label="Break-even at 1 Hz" value={`${number(longest.break_even_days_1hz, 1)} days`} detail="of one plant streaming 8 sensors" />
      </div>

      <div className="two-column">
        <div>
          <h3 className="chart-title">Total data moved</h3>
          <BarChart rows={volumeRows} format={(value) => bytes(value)} unitLabel={(value) => bytes(value)} />
          <Legend items={[{ label: 'Federated (weights only)', color: 'var(--series-1)' }, { label: 'Centralised (raw data)', color: 'var(--series-2)' }]} />
        </div>
        <div>
          <h3 className="chart-title">One plant's traffic: retraining vs streaming</h3>
          <LineChart
            series={lines}
            xLabel="Days of operation"
            yFormat={(value) => bytes(value)}
            xFormat={(value) => `${value}`}
            markers={[{ x: longest.break_even_days_1hz, label: `${number(longest.break_even_days_1hz, 1)} d` }]}
          />
        </div>
      </div>

      {communication.per_model.length > 0 && (
        <>
          <h3 className="chart-title">Traffic per training run, by model</h3>
          <BarChart
            rows={communication.per_model.map((row) => ({
              id: row.id,
              label: row.label,
              value: row.communication_bytes,
              color: modelColors[row.id] || 'var(--series-other)',
              note: `${row.training}, ${row.rounds} rounds · model ${bytes(row.model_bytes)}`,
            }))}
            format={(value) => bytes(value)}
            unitLabel={(value) => bytes(value)}
          />
        </>
      )}
      <p className="caption">
        SKAB holds only about 13 hours of data, so one training run moves more bytes than its recordings. A real plant streams
        continuously ({bytes(perDay)} a day at 1 Hz), so federation saves bandwidth when retraining is less frequent than the break-even point.
        Higher sampling rates, such as vibration bursts, bring the break-even point earlier.
      </p>
    </Card>
  )
}

export default function PerformancePage() {
  const [role, setRole] = useQueryParam('recordings', 'final')
  const [scoreKey, setScoreKey] = useQueryParam('score', 'full')
  const comparison = useApi(`/api/comparisons?role=${role}`)
  const data = comparison.data

  const missing = useMemo(() => (data ? Object.entries(data.available).filter(([, present]) => !present).map(([name]) => name) : []), [data])

  return (
    <div className="page">
      <PageHeader
        eyebrow="Research results · SKAB benchmark"
        title="Model performance"
        description="How federated learning compares with centralised and isolated training, which model family detects faults best, and what federation costs in network traffic."
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
          {data.scenarios && <ScenariosSection data={data} scoreKey={scoreKey} />}
          {data.models && <ModelsSection data={data} scoreKey={scoreKey} />}
          {data.communication && <CommunicationSection data={data} />}
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
