import { useEffect, useMemo, useState } from 'react'
import { Icon } from '../../components/Icon'
import { Badge, Button, Card, EmptyState, Field, Loading, Notice, PageHeader, StatTile } from '../../components/ui'
import { api } from '../../lib/api'
import { useApi } from '../../lib/hooks'
import { bytes, dateTime, integer } from '../../lib/format'

const groups = [
  {
    title: 'Federation',
    description: 'How many rounds run and how much each plant trains between them.',
    fields: [
      { key: 'strategy', label: 'Aggregation strategy', hint: 'Weighted by each plant\'s number of training windows.' },
      { key: 'rounds', label: 'Federation rounds', hint: 'Each round: plants train, upload weights, receive the new global model.' },
      { key: 'local_epochs', label: 'Local epochs per round', hint: 'Passes over a plant\'s own data before it uploads.' },
      { key: 'fraction_fit', label: 'Share of plants per round', hint: '1.0 trains every participating plant each round.', step: 0.1 },
    ],
  },
  {
    title: 'Optimisation',
    description: 'Settings for each plant\'s local training.',
    fields: [
      { key: 'batch_size', label: 'Batch size' },
      { key: 'learning_rate', label: 'Learning rate', step: 0.0001 },
      { key: 'seed', label: 'Random seed', hint: 'The same seed reproduces a run exactly.' },
      { key: 'device', label: 'Compute device', hint: 'Auto picks CUDA, Apple Metal or the CPU.' },
    ],
  },
  {
    title: 'LSTM autoencoder',
    description: 'Model size sets how many bytes travel each round.',
    fields: [
      { key: 'hidden_size', label: 'Hidden units' },
      { key: 'latent_size', label: 'Latent size', hint: 'A narrow bottleneck learns normal behaviour instead of copying input.' },
      { key: 'num_layers', label: 'LSTM layers' },
      { key: 'dropout', label: 'Dropout', step: 0.05 },
    ],
  },
]

// Trainable parameters of the LSTM autoencoder (matches PyTorch's count).
function lstmParameters({ hidden_size: h, latent_size: latent, num_layers: layers }, features = 8) {
  const lstm = (input) => 4 * h * (input + h) + 8 * h
  let total = 0
  for (let layer = 0; layer < layers; layer += 1) total += lstm(layer === 0 ? features : h)
  total += h * latent + latent
  for (let layer = 0; layer < layers; layer += 1) total += lstm(layer === 0 ? latent : h)
  total += h * features + features
  return total
}

function TrainingPanel({ training, participating, config, onChange }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const job = training?.job
  const running = job?.state === 'running'

  const act = async (method) => {
    setBusy(true)
    setError('')
    try {
      onChange(await api('/api/admin/training', { method }))
    } catch (failure) {
      setError(failure.message)
    } finally {
      setBusy(false)
    }
  }

  const parameters = lstmParameters(config)
  const perRound = parameters * 4 * 2 * Math.max(1, Math.round(config.fraction_fit * participating.length))

  return (
    <Card
      title="Run federated training"
      subtitle="Starts a Flower simulation with the saved settings. Each plant trains on its own partition."
      actions={job && <Badge tone={running ? 'info' : job.state === 'completed' ? 'normal' : 'critical'} icon={running ? 'clock' : undefined}>{job.state}</Badge>}
    >
      {!training?.available && <Notice tone="warning" title="Training is unavailable here">{training?.reason}</Notice>}
      {error && <Notice tone="critical">{error}</Notice>}

      <div className="stat-grid stat-grid-compact">
        <StatTile label="Participating plants" value={participating.length} detail={<a href="#/admin/nodes">Manage plant nodes</a>} />
        <StatTile label="Model parameters" value={integer(parameters)} detail={bytes(parameters * 4)} />
        <StatTile label="Traffic per round" value={bytes(perRound)} detail="uploads plus downloads" />
        <StatTile label="Traffic per run" value={bytes(perRound * config.rounds)} detail={`${config.rounds} rounds`} />
      </div>

      <div className="button-row">
        <Button variant="primary" icon="play" disabled={busy || running || !training?.available} onClick={() => act('POST')}>
          Start training run
        </Button>
        {running && <Button variant="danger" icon="stop" disabled={busy} onClick={() => act('DELETE')}>Stop run</Button>}
      </div>

      {job && (
        <div className="job">
          <p className="muted small">
            Started {dateTime(job.started_at)} by {job.started_by} · plants {job.clients.join(', ')}
          </p>
          {job.log_tail?.length > 0 && (
            <pre className="log" aria-label="Training log">{job.log_tail.join('\n')}</pre>
          )}
        </div>
      )}
    </Card>
  )
}

export default function FederationPage() {
  const settings = useApi('/api/admin/federation')
  const nodes = useApi('/api/admin/nodes')
  const running = settings.data?.training?.job?.state === 'running'
  const runs = useApi('/api/runs?experiment=federated&limit=10', { interval: running ? 5000 : 0 })
  const trainingPoll = useApi('/api/admin/training', { interval: running ? 4000 : 0, enabled: running })
  const [form, setForm] = useState(null)
  const [message, setMessage] = useState(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (settings.data && !form) setForm(settings.data.config)
  }, [settings.data, form])

  useEffect(() => {
    if (trainingPoll.data && settings.data) settings.setData({ ...settings.data, training: trainingPoll.data })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingPoll.data])

  const dirty = useMemo(
    () => form && settings.data && Object.keys(form).some((key) => String(form[key]) !== String(settings.data.config[key])),
    [form, settings.data],
  )

  if (settings.loading && !settings.data) return <Loading />
  if (settings.error && !settings.data) return <EmptyState icon="alert" title="Settings could not be loaded">{settings.error}</EmptyState>
  if (!form) return <Loading />

  const { limits, choices, defaults } = settings.data
  const participating = (nodes.data?.nodes || []).filter((node) => node.participates).map((node) => node.client_id)

  const save = async (event) => {
    event.preventDefault()
    setSaving(true)
    setMessage(null)
    try {
      const updated = await api('/api/admin/federation', { method: 'PUT', body: form })
      settings.setData(updated)
      setForm(updated.config)
      setMessage({ tone: 'normal', text: 'Settings saved. The next training run will use them.' })
    } catch (failure) {
      setMessage({ tone: 'critical', text: failure.message })
    } finally {
      setSaving(false)
    }
  }

  const input = (field) => {
    const { key } = field
    if (choices[key]) {
      return (
        <select id={key} value={form[key]} onChange={(event) => setForm({ ...form, [key]: event.target.value })}>
          {choices[key].map((choice) => <option key={choice} value={choice}>{choice}</option>)}
        </select>
      )
    }
    const [min, max] = limits[key]
    return (
      <input
        id={key}
        type="number"
        min={min}
        max={max}
        step={field.step || 1}
        value={form[key]}
        onChange={(event) => setForm({ ...form, [key]: event.target.value === '' ? '' : Number(event.target.value) })}
        required
      />
    )
  }

  return (
    <div className="page">
      <PageHeader
        eyebrow="Administration"
        title="Federated training"
        description="Configure how plants train the shared anomaly detector, then start a run. Plants share model weights only; raw telemetry stays on site."
      />

      <TrainingPanel
        training={settings.data.training}
        participating={participating}
        config={settings.data.config}
        onChange={(training) => {
          settings.setData({ ...settings.data, training })
          runs.reload({ quiet: true })
        }}
      />

      <form onSubmit={save}>
        <Card
          title="Training settings"
          subtitle={settings.data.updated_at ? `Last saved ${dateTime(settings.data.updated_at)}` : 'Using the settings frozen for the SKAB comparison.'}
          actions={(
            <div className="button-row">
              <Button variant="ghost" icon="refresh" onClick={() => setForm(defaults)}>Restore defaults</Button>
              <Button type="submit" variant="primary" icon="check" disabled={!dirty || saving}>{saving ? 'Saving…' : 'Save settings'}</Button>
            </div>
          )}
        >
          {message && <Notice tone={message.tone}>{message.text}</Notice>}
          {dirty && <Notice tone="info">You have unsaved changes. Training runs use the saved settings.</Notice>}
          <div className="settings-groups">
            {groups.map((group) => (
              <fieldset key={group.title} className="settings-group">
                <legend>{group.title}</legend>
                <p>{group.description}</p>
                {group.fields.map((field) => (
                  <Field
                    key={field.key}
                    htmlFor={field.key}
                    label={field.label}
                    hint={`${field.hint ? `${field.hint} ` : ''}Default ${defaults[field.key]}.`}
                  >
                    {input(field)}
                  </Field>
                ))}
              </fieldset>
            ))}
          </div>
        </Card>
      </form>

      <Card title="Recent federated runs" subtitle="Each run records its rounds and evaluation in the experiment database." padded={false}>
        {!runs.data?.runs.length ? (
          <EmptyState icon="database" title="No runs yet">Start a training run to see it here.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Started</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="numeric">Rounds</th>
                  <th scope="col" className="numeric">Local epochs</th>
                  <th scope="col">Plants</th>
                  <th scope="col" className="numeric">Saved results</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.runs.map((run) => (
                  <tr key={run.run_id}>
                    <td>{dateTime(run.started_at)}</td>
                    <td>
                      <Badge size="sm" tone={run.status === 'completed' ? 'normal' : run.status === 'failed' ? 'critical' : 'info'} icon={run.status === 'running' ? 'clock' : undefined}>
                        {run.status}
                      </Badge>
                    </td>
                    <td className="numeric">{run.config.rounds ?? '—'}</td>
                    <td className="numeric">{run.config.local_epochs ?? '—'}</td>
                    <td>{run.config.clients ? run.config.clients.join(', ') : `${run.config.client_count ?? '—'} plants`}</td>
                    <td className="numeric">{run.result_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <p className="data-source">
        <Icon name="info" size={14} />
        Runs use the SKAB-aligned evaluation: per-plant scaling, the hybrid score and normal-only training windows.
      </p>
    </div>
  )
}
