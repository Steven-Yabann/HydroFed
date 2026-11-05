import { useState } from 'react'
import { Button, Drawer, Field, Notice } from '../components/ui'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { dateTime, initials } from '../lib/format'

const providerNames = { password: 'Email and password', google: 'Google', microsoft: 'Microsoft' }

export default function AccountDrawer({ open, onClose }) {
  const { user, signOut } = useAuth()
  const [form, setForm] = useState({ current: '', next: '' })
  const [message, setMessage] = useState(null)

  const changePassword = async (event) => {
    event.preventDefault()
    try {
      await api('/api/auth/password', {
        method: 'POST',
        body: { current_password: form.current, new_password: form.next },
      })
      setForm({ current: '', next: '' })
      setMessage({ tone: 'normal', text: 'Password changed.' })
    } catch (error) {
      setMessage({ tone: 'critical', text: error.message })
    }
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title="Your account"
      footer={<Button variant="danger" icon="logout" onClick={() => signOut()}>Sign out</Button>}
    >
      <div className="account-summary">
        <span className="avatar avatar-lg">{initials(user.name)}</span>
        <div>
          <strong>{user.name}</strong>
          <span>{user.email}</span>
          <span className="pill">{user.role_label}</span>
        </div>
      </div>
      <dl className="detail-list">
        <div><dt>Last signed in with</dt><dd>{providerNames[user.last_provider] || '—'}</dd></div>
        <div><dt>Last sign-in</dt><dd>{dateTime(user.last_login_at)}</dd></div>
        <div><dt>Member since</dt><dd>{dateTime(user.created_at)}</dd></div>
      </dl>

      {user.has_password ? (
        <form className="stack" onSubmit={changePassword}>
          <h3>Change password</h3>
          {message && <Notice tone={message.tone}>{message.text}</Notice>}
          <Field label="Current password">
            <input type="password" autoComplete="current-password" required value={form.current} onChange={(event) => setForm({ ...form, current: event.target.value })} />
          </Field>
          <Field label="New password" hint="At least 10 characters.">
            <input type="password" autoComplete="new-password" minLength={10} required value={form.next} onChange={(event) => setForm({ ...form, next: event.target.value })} />
          </Field>
          <Button type="submit" variant="primary">Update password</Button>
        </form>
      ) : (
        <Notice tone="info">You sign in with single sign-on, so there is no HydroFed password to change.</Notice>
      )}
    </Drawer>
  )
}
