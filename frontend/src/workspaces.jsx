function PageHeading({ eyebrow, title, description }) {
  return (
    <div className="workspace-heading">
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  )
}

function EmptyRunNotice() {
  return (
    <div className="empty-message">
      <strong>No selected evaluation results are available.</strong>
      <span>Select a completed federated run or run a new evaluation first.</span>
    </div>
  )
}

export function OperatorWorkspace({ macroMetrics, plants, formatMetric }) {
  const attentionPlants = plants.filter(
    (plant) => typeof plant.metrics.f1 === 'number' && plant.metrics.f1 < 0.2,
  )

  return (
    <section className="workspace-panel">
      <PageHeading
        eyebrow="Plant operator workspace"
        title="Plant operations overview"
        description="Review the latest evaluated model performance before acting on operational alerts."
      />

      {!plants.length ? <EmptyRunNotice /> : (
        <>
          <div className="workspace-stat-grid">
            <article>
              <span>Fleet macro F1</span>
              <strong>{formatMetric(macroMetrics.f1)}</strong>
            </article>
            <article>
              <span>Low-F1 plants</span>
              <strong>{attentionPlants.length}</strong>
            </article>
            <article>
              <span>Evaluated plants</span>
              <strong>{plants.length}</strong>
            </article>
          </div>

          <div className="information-banner">
            <strong>Operational note</strong>
            <span>
              This page reports model-evaluation evidence. Live anomaly-event alerts will appear here after the event-ingestion API is added.
            </span>
          </div>

          <div className="workspace-list">
            {plants.map((plant) => (
              <article key={plant.id} className="workspace-list-row">
                <div>
                  <strong>{plant.client_id}</strong>
                  <span>Reconstruction threshold {formatMetric(plant.metrics.threshold)}</span>
                </div>
                <div className="metric-pair">
                  <span>F1</span>
                  <strong>{formatMetric(plant.metrics.f1)}</strong>
                </div>
                <div className="metric-pair">
                  <span>AUC-ROC</span>
                  <strong>{formatMetric(plant.metrics.auc_roc)}</strong>
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

export function MaintenanceWorkspace({ plants, formatMetric }) {
  return (
    <section className="workspace-panel">
      <PageHeading
        eyebrow="Maintenance engineer workspace"
        title="Anomaly review queue"
        description="Use selected-run thresholds and evaluation results to prioritise investigation."
      />

      <div className="information-banner">
        <strong>Event labelling is not connected yet</strong>
        <span>
          The current backend stores evaluation summaries, not individual anomaly windows. The review controls are intentionally disabled until an anomaly-event API is implemented.
        </span>
      </div>

      {!plants.length ? <EmptyRunNotice /> : (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Plant</th>
                <th>Threshold</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Review state</th>
              </tr>
            </thead>
            <tbody>
              {plants.map((plant) => (
                <tr key={plant.id}>
                  <td>{plant.client_id}</td>
                  <td>{formatMetric(plant.metrics.threshold)}</td>
                  <td>{formatMetric(plant.metrics.precision)}</td>
                  <td>{formatMetric(plant.metrics.recall)}</td>
                  <td><span className="status-pill pending">Awaiting event data</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export function AdministrationWorkspace({ plants, selectedRun, formatMetric }) {
  const configuration = selectedRun?.config || {}

  return (
    <section className="workspace-panel">
      <PageHeading
        eyebrow="System administrator workspace"
        title="Client nodes and federation control"
        description="Inspect the clients represented in the selected experiment and review its training configuration."
      />

      <div className="admin-grid">
        <article className="sub-panel">
          <h3>Client nodes</h3>
          {!plants.length ? <EmptyRunNotice /> : (
            <ul className="client-list">
              {plants.map((plant) => (
                <li key={plant.id}>
                  <span className="status-dot reported" />
                  <div>
                    <strong>{plant.client_id}</strong>
                    <span>Evaluation recorded for the selected run</span>
                  </div>
                  <span>F1 {formatMetric(plant.metrics.f1)}</span>
                </li>
              ))}
            </ul>
          )}
        </article>

        <article className="sub-panel">
          <h3>Federation configuration</h3>
          <dl className="configuration-list">
            <div><dt>Rounds</dt><dd>{configuration.rounds ?? '—'}</dd></div>
            <div><dt>Local epochs</dt><dd>{configuration.local_epochs ?? '—'}</dd></div>
            <div><dt>Client count</dt><dd>{configuration.client_count ?? '—'}</dd></div>
            <div><dt>Device</dt><dd>{configuration.device ?? '—'}</dd></div>
            <div><dt>Seed</dt><dd>{configuration.seed ?? '—'}</dd></div>
          </dl>
          <button type="button" disabled>Start federation API pending</button>
          <p className="disabled-explanation">
            Starting or stopping training requires a background-job API and is not simulated by this interface.
          </p>
        </article>
      </div>
    </section>
  )
}

function csvCell(value) {
  return `"${String(value ?? '').replaceAll('"', '""')}"`
}

export function ResearchWorkspace({ plants, runs, selectedRun, formatMetric, formatTimestamp }) {
  const exportEvaluation = () => {
    if (!plants.length) return

    const header = ['plant', 'precision', 'recall', 'f1', 'auc_roc', 'threshold']
    const rows = plants.map((plant) => [
      plant.client_id,
      plant.metrics.precision,
      plant.metrics.recall,
      plant.metrics.f1,
      plant.metrics.auc_roc,
      plant.metrics.threshold,
    ])
    const csv = [header, ...rows]
      .map((row) => row.map(csvCell).join(','))
      .join('\n')

    const link = document.createElement('a')
    link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    link.download = `hydrofed-${selectedRun?.run_id || 'legacy'}-evaluation.csv`
    link.click()
    URL.revokeObjectURL(link.href)
  }

  return (
    <section className="workspace-panel">
      <PageHeading
        eyebrow="Researcher workspace"
        title="Experiment analysis and export"
        description="Review completed federated runs and export the selected plant-level evaluation metrics."
      />

      <div className="research-actions">
        <button type="button" onClick={exportEvaluation} disabled={!plants.length}>
          Export selected evaluation CSV
        </button>
        <span>{plants.length ? `${plants.length} plant rows ready to export` : 'No evaluation rows selected'}</span>
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Started</th>
              <th>Rounds</th>
              <th>Seed</th>
              <th>Saved records</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.run_id}>
                <td>{run.run_id.slice(0, 12)}…</td>
                <td><span className={`status-pill ${run.status}`}>{run.status}</span></td>
                <td>{formatTimestamp(run.started_at)}</td>
                <td>{run.config.rounds ?? '—'}</td>
                <td>{run.config.seed ?? '—'}</td>
                <td>{run.result_count}</td>
              </tr>
            ))}
            {!runs.length && (
              <tr><td colSpan="6">No tracked federated runs have been saved yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {selectedRun && (
        <p className="research-note">
          Selected run: {selectedRun.run_id}. Its displayed F1 is {formatMetric(
            plants.length
              ? plants.reduce((sum, plant) => sum + (plant.metrics.f1 || 0), 0) / plants.length
              : null,
          )}.
        </p>
      )}
    </section>
  )
}
