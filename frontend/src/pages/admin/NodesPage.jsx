import { useState } from 'react'
import { Badge, Button, Card, Drawer, EmptyState, Field, Loading, Notice, PageHeader, Switch } from '../../components/ui'
import { api } from '../../lib/api'
import { useApi } from '../../lib/hooks'
import { integer, number, percent, relative } from '../../lib/format'

function NodeDrawer({ node, onClose, onSaved }) {
  const [form, setForm] = useState(() => ({
    display_name: node.display_name,
    location: node.location,
    capacity_mw: node.capacity_mw ?? '',
    notes: node.notes,
  }))
  const [error, setError] = useState('')

  const save = async (event) => {
    event.preventDefault()
    try {
      const { node: updated } = await api(`/api/admin/nodes/${node.client_id}`, {
        method: 'PATCH',
        body: { ...form, capacity_mw: form.capacity_mw === '' ? null : Number(form.capacity_mw) },
      })
      onSaved(updated)
    } catch (failure) {
      setError(failure.message)
    }
  }

  return (
    <Drawer
      open
      onClose={onClose}
      title={`Edit ${node.display_name}`}
      subtitle={`Node ${node.client_id}`}
      footer={(
        <>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" icon="check" type="submit" form="node-form">Save</Button>
        </>
      )}
    >
      <form id="node-form" className="stack" onSubmit={save}>
        {error && <Notice tone="critical">{error}</Notice>}
        <Field label="Display name" hint="Shown to operators on the fleet overview.">
          <input required maxLength={200} value={form.display_name} onChange={(event) => setForm({ ...form, display_name: event.target.value })} />
        </Field>
        <Field label="Location" hint="For example the river or county.">
          <input maxLength={200} value={form.location} onChange={(event) => setForm({ ...form, location: event.target.value })} />
        </Field>
        <Field label="Installed capacity (MW)">
          <input type="number" min={0} step={0.1} value={form.capacity_mw} onChange={(event) => setForm({ ...form, capacity_mw: event.target.value })} />
        </Field>
        <Field label="Notes" hint="Visible to administrators only.">
          <textarea rows={3} maxLength={200} value={form.notes} onChange={(event) => setForm({ ...form, notes: event.target.value })} />
        </Field>
      </form>
    </Drawer>
  )
}

export default function NodesPage() {
  const nodes = useApi('/api/admin/nodes')
  const [editing, setEditing] = useState(null)
  const [error, setError] = useState('')

  const replace = (updated) => nodes.setData({
    nodes: nodes.data.nodes.map((node) => (node.client_id === updated.client_id ? { ...node, ...updated } : node)),
  })

  const toggle = async (node) => {
    setError('')
    try {
      const { node: updated } = await api(`/api/admin/nodes/${node.client_id}`, {
        method: 'PATCH',
        body: { participates: !node.participates },
      })
      replace(updated)
    } catch (failure) {
      setError(failure.message)
    }
  }

  const list = nodes.data?.nodes || []
  const participating = list.filter((node) => node.participates).length

  return (
    <div className="page">
      <PageHeader
        eyebrow="Administration"
        title="Plant nodes"
        description="Each node is a plant that keeps its telemetry on site and trains locally. Pause a node to leave it out of the next federated run, for example during maintenance or a network outage."
      />

      {error && <Notice tone="critical">{error}</Notice>}
      <div className="summary-strip">
        <span><b>{list.length}</b> nodes</span>
        <span><b>{participating}</b> participating in training</span>
        <span><b>{list.length - participating}</b> paused</span>
      </div>

      <Card padded={false}>
        {nodes.loading && !nodes.data ? <Loading /> : nodes.error ? (
          <EmptyState icon="alert" title="Nodes could not be loaded">{nodes.error}</EmptyState>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Plant</th>
                  <th scope="col">Location</th>
                  <th scope="col" className="numeric">Capacity</th>
                  <th scope="col" className="numeric">Local windows</th>
                  <th scope="col" className="numeric">Anomalous share</th>
                  <th scope="col">Last federated round</th>
                  <th scope="col">Training</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {list.map((node) => (
                  <tr key={node.client_id}>
                    <th scope="row">
                      {node.display_name}
                      <span className="cell-note">{node.client_id}</span>
                    </th>
                    <td>{node.location || <span className="muted">Not set</span>}</td>
                    <td className="numeric">{node.capacity_mw ? `${number(node.capacity_mw, 1)} MW` : '—'}</td>
                    <td className="numeric">{integer(node.windows)}</td>
                    <td className="numeric">{percent(node.anomaly_rate)}</td>
                    <td>
                      {node.last_round ? (
                        <>
                          Round {node.last_round}
                          <span className="cell-note">{relative(node.last_round_at)}{node.last_loss !== undefined && node.last_loss !== null ? ` · loss ${number(node.last_loss, 4)}` : ''}</span>
                        </>
                      ) : <span className="muted">Never</span>}
                    </td>
                    <td>
                      <div className="switch-cell">
                        <Switch checked={node.participates} onChange={() => toggle(node)} label={`${node.display_name} participates in training`} />
                        <Badge size="sm" tone={node.participates ? 'normal' : 'muted'} icon={node.participates ? 'check' : 'stop'}>
                          {node.participates ? 'Participating' : 'Paused'}
                        </Badge>
                      </div>
                    </td>
                    <td className="row-action">
                      <Button size="sm" icon="edit" onClick={() => setEditing(node)}>Edit</Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {editing && (
        <NodeDrawer
          key={editing.client_id}
          node={editing}
          onClose={() => setEditing(null)}
          onSaved={(updated) => {
            replace(updated)
            setEditing(null)
          }}
        />
      )}
    </div>
  )
}
