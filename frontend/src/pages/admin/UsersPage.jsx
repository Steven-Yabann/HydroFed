import { useState } from 'react'
import { Badge, Button, Card, Drawer, EmptyState, Field, Loading, Notice, PageHeader } from '../../components/ui'
import { api } from '../../lib/api'
import { useAuth } from '../../lib/auth'
import { useApi } from '../../lib/hooks'
import { dateTime, initials, relative } from '../../lib/format'

const permissionLabels = {
  view_fleet: 'Watch the fleet and plant trends',
  review_alerts: 'Acknowledge and review alerts',
  view_performance: 'Read model performance results',
  manage_federation: 'Configure and start federated training',
  manage_nodes: 'Manage plant nodes',
  manage_users: 'Add and manage users',
}
const providerLabels = { password: 'Password', google: 'Google', microsoft: 'Microsoft' }

function RolePicker({ roles, value, onChange }) {
  return (
    <fieldset className="radio-cards">
      <legend>Role</legend>
      {roles.map((role) => (
        <label key={role.id} className={value === role.id ? 'selected' : ''}>
          <input type="radio" name="role" value={role.id} checked={value === role.id} onChange={() => onChange(role.id)} />
          <span>
            <strong>{role.label}</strong>
            <small>{role.permissions.map((permission) => permissionLabels[permission] || permission).join(' · ')}</small>
          </span>
        </label>
      ))}
    </fieldset>
  )
}

function NewUserDrawer({ roles, onClose, onCreated }) {
  const [form, setForm] = useState({ name: '', email: '', role: 'operator', method: 'sso', password: '' })
  const [error, setError] = useState('')

  const submit = async (event) => {
    event.preventDefault()
    try {
      const { user } = await api('/api/admin/users', {
        method: 'POST',
        body: {
          name: form.name,
          email: form.email,
          role: form.role,
          password: form.method === 'password' ? form.password : null,
        },
      })
      onCreated(user)
    } catch (failure) {
      setError(failure.message)
    }
  }

  return (
    <Drawer
      open
      onClose={onClose}
      title="Add a user"
      subtitle="They can sign in as soon as the account exists."
      footer={(
        <>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" icon="plus" type="submit" form="new-user">Add user</Button>
        </>
      )}
    >
      <form id="new-user" className="stack" onSubmit={submit}>
        {error && <Notice tone="critical">{error}</Notice>}
        <Field label="Full name">
          <input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
        </Field>
        <Field label="Work email" hint="Must match their Google or Microsoft account to use single sign-on.">
          <input type="email" required value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
        </Field>
        <RolePicker roles={roles} value={form.role} onChange={(role) => setForm({ ...form, role })} />
        <fieldset className="radio-cards">
          <legend>How they sign in</legend>
          <label className={form.method === 'sso' ? 'selected' : ''}>
            <input type="radio" name="method" checked={form.method === 'sso'} onChange={() => setForm({ ...form, method: 'sso' })} />
            <span><strong>Single sign-on only</strong><small>Google or Microsoft with the email above. No password to manage.</small></span>
          </label>
          <label className={form.method === 'password' ? 'selected' : ''}>
            <input type="radio" name="method" checked={form.method === 'password'} onChange={() => setForm({ ...form, method: 'password' })} />
            <span><strong>Set a starting password</strong><small>Share it securely; they can change it from their account.</small></span>
          </label>
        </fieldset>
        {form.method === 'password' && (
          <Field label="Starting password" hint="At least 10 characters.">
            <input type="password" autoComplete="new-password" minLength={10} required value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />
          </Field>
        )}
      </form>
    </Drawer>
  )
}

function EditUserDrawer({ user, roles, isSelf, onClose, onSaved, onDeleted }) {
  const [form, setForm] = useState({ name: user.name, role: user.role, status: user.status, password: '' })
  const [error, setError] = useState('')
  const [confirming, setConfirming] = useState(false)

  const save = async (event) => {
    event.preventDefault()
    try {
      const { user: updated } = await api(`/api/admin/users/${user.user_id}`, {
        method: 'PATCH',
        body: { name: form.name, role: form.role, status: form.status, password: form.password || null },
      })
      onSaved(updated)
    } catch (failure) {
      setError(failure.message)
    }
  }

  const remove = async () => {
    try {
      await api(`/api/admin/users/${user.user_id}`, { method: 'DELETE' })
      onDeleted(user.user_id)
    } catch (failure) {
      setError(failure.message)
      setConfirming(false)
    }
  }

  return (
    <Drawer
      open
      onClose={onClose}
      title={user.name}
      subtitle={user.email}
      footer={(
        <>
          {!isSelf && (confirming ? (
            <Button variant="danger" icon="trash" onClick={remove}>Confirm delete</Button>
          ) : (
            <Button variant="ghost" icon="trash" className="danger-text" onClick={() => setConfirming(true)}>Delete</Button>
          ))}
          <span className="spacer" />
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button variant="primary" icon="check" type="submit" form="edit-user">Save</Button>
        </>
      )}
    >
      <form id="edit-user" className="stack" onSubmit={save}>
        {error && <Notice tone="critical">{error}</Notice>}
        {confirming && <Notice tone="warning">Deleting removes {user.name}'s access and cannot be undone. Disabling keeps the record.</Notice>}
        <Field label="Full name">
          <input required value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
        </Field>
        <RolePicker roles={roles} value={form.role} onChange={(role) => setForm({ ...form, role })} />
        <fieldset className="radio-cards radio-inline">
          <legend>Access</legend>
          {['active', 'disabled'].map((status) => (
            <label key={status} className={form.status === status ? 'selected' : ''}>
              <input type="radio" name="status" checked={form.status === status} onChange={() => setForm({ ...form, status })} disabled={isSelf} />
              <span><strong>{status === 'active' ? 'Active' : 'Disabled'}</strong><small>{status === 'active' ? 'Can sign in' : 'Signed out and blocked'}</small></span>
            </label>
          ))}
        </fieldset>
        <Field label={user.has_password ? 'Reset password' : 'Set a password'} hint="Leave empty to keep the current sign-in method.">
          <input type="password" autoComplete="new-password" minLength={10} value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} />
        </Field>
        <dl className="detail-list">
          <div><dt>Created</dt><dd>{dateTime(user.created_at)}</dd></div>
          <div><dt>Last sign-in</dt><dd>{user.last_login_at ? `${dateTime(user.last_login_at)} via ${providerLabels[user.last_provider] || user.last_provider}` : 'Never'}</dd></div>
        </dl>
      </form>
    </Drawer>
  )
}

export default function UsersPage() {
  const { user: me } = useAuth()
  const users = useApi('/api/admin/users')
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState(null)
  const [query, setQuery] = useState('')

  const list = (users.data?.users || []).filter((user) => `${user.name} ${user.email}`.toLowerCase().includes(query.toLowerCase()))
  const roles = users.data?.roles || []
  const update = (next) => users.setData({ ...users.data, users: next })

  return (
    <div className="page">
      <PageHeader
        eyebrow="Administration"
        title="Users & access"
        description="Only people added here can sign in, whether with a password or with Google or Microsoft. Roles decide what each person can see and change."
        actions={<Button variant="primary" icon="plus" onClick={() => setAdding(true)}>Add user</Button>}
      />

      <Card padded={false}>
        <div className="filter-bar">
          <label className="search">
            <span className="sr-only">Search users</span>
            <input type="search" placeholder="Search by name or email" value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
        </div>
        {users.loading && !users.data ? <Loading /> : users.error ? (
          <EmptyState icon="alert" title="Users could not be loaded">{users.error}</EmptyState>
        ) : !list.length ? (
          <EmptyState icon="users" title="No users match">Try another name or email.</EmptyState>
        ) : (
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Person</th>
                  <th scope="col">Role</th>
                  <th scope="col">Sign-in</th>
                  <th scope="col">Access</th>
                  <th scope="col">Last active</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {list.map((user) => (
                  <tr key={user.user_id}>
                    <th scope="row">
                      <span className="person">
                        <span className="avatar avatar-sm">{initials(user.name)}</span>
                        <span>
                          {user.name}{user.user_id === me.user_id && <span className="pill">You</span>}
                          <span className="cell-note">{user.email}</span>
                        </span>
                      </span>
                    </th>
                    <td>{user.role_label}</td>
                    <td>{user.has_password ? 'Password or SSO' : 'Single sign-on'}</td>
                    <td>
                      <Badge size="sm" tone={user.status === 'active' ? 'normal' : 'muted'} icon={user.status === 'active' ? 'check' : 'lock'}>
                        {user.status === 'active' ? 'Active' : 'Disabled'}
                      </Badge>
                    </td>
                    <td>{user.last_login_at ? relative(user.last_login_at) : <span className="muted">Never</span>}</td>
                    <td className="row-action"><Button size="sm" icon="edit" onClick={() => setEditing(user)}>Edit</Button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {adding && (
        <NewUserDrawer
          roles={roles}
          onClose={() => setAdding(false)}
          onCreated={(user) => {
            update([...users.data.users, user].sort((a, b) => a.name.localeCompare(b.name)))
            setAdding(false)
          }}
        />
      )}
      {editing && (
        <EditUserDrawer
          key={editing.user_id}
          user={editing}
          roles={roles}
          isSelf={editing.user_id === me.user_id}
          onClose={() => setEditing(null)}
          onSaved={(updated) => {
            update(users.data.users.map((user) => (user.user_id === updated.user_id ? updated : user)))
            setEditing(null)
          }}
          onDeleted={(id) => {
            update(users.data.users.filter((user) => user.user_id !== id))
            setEditing(null)
          }}
        />
      )}
    </div>
  )
}
