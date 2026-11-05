import { useEffect, useState } from 'react'
import { GoogleMark, HydroFedMark, Icon, MicrosoftMark } from '../components/Icon'
import { Notice } from '../components/ui'
import { api, oauthStartUrl } from '../lib/api'
import { useAuth } from '../lib/auth'

const providerMarks = { google: <GoogleMark />, microsoft: <MicrosoftMark /> }
const knownProviders = [
  { id: 'google', label: 'Google' },
  { id: 'microsoft', label: 'Microsoft' },
]

export default function LoginPage() {
  const { acceptSession, notice } = useAuth()
  const [providers, setProviders] = useState(null)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(notice)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    api('/api/auth/providers')
      .then((data) => setProviders(data.providers))
      .catch((failure) => {
        setProviders([])
        setError(failure.message)
      })
  }, [])

  const submit = async (event) => {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const session = await api('/api/auth/login', { method: 'POST', body: { email, password } })
      await acceptSession(session)
    } catch (failure) {
      setError(failure.message)
      setSubmitting(false)
    }
  }

  const configured = new Set((providers || []).map((provider) => provider.id))
  // In development, show unconfigured providers greyed out with a setup hint.
  const visible = import.meta.env.DEV ? knownProviders : knownProviders.filter((provider) => configured.has(provider.id))

  return (
    <div className="login">
      <section className="login-hero" aria-hidden="true">
        <div className="login-waves">
          <span />
          <span />
          <span />
          <span />
        </div>
        <div className="login-hero-content">
          <div className="login-brand">
            <HydroFedMark size={44} />
            <span>HydroFed</span>
          </div>
          <h2>Early warning for every turbine in the fleet.</h2>
          <p>
            Each plant trains on its own telemetry and shares only model weights.
            Operators see anomalies as they develop, without raw sensor data ever leaving the site.
          </p>
          <ul>
            <li><Icon name="signal" size={18} /> Live anomaly scores for every plant</li>
            <li><Icon name="alert" size={18} /> Prioritised alerts with review tracking</li>
            <li><Icon name="shield" size={18} /> Raw telemetry stays at the plant</li>
          </ul>
        </div>
      </section>

      <section className="login-panel">
        <form className="login-card" onSubmit={submit}>
          <div className="login-mobile-brand">
            <HydroFedMark size={36} />
            <span>HydroFed</span>
          </div>
          <h1>Sign in</h1>
          <p className="login-lead">Use the account your administrator set up for you.</p>

          {error && <Notice tone="critical">{error}</Notice>}

          {visible.length > 0 && (
            <div className="sso-buttons">
              {visible.map((provider) => {
                const enabled = configured.has(provider.id)
                return enabled ? (
                  <a key={provider.id} className="sso-button" href={oauthStartUrl(provider.id)}>
                    {providerMarks[provider.id]}
                    Continue with {provider.label}
                  </a>
                ) : (
                  <span
                    key={provider.id}
                    className="sso-button disabled"
                    title={`Set HYDROFED_${provider.id.toUpperCase()}_CLIENT_ID and _CLIENT_SECRET on the API to enable`}
                  >
                    {providerMarks[provider.id]}
                    Continue with {provider.label}
                    <small>not configured</small>
                  </span>
                )
              })}
            </div>
          )}

          {visible.length > 0 && <div className="divider"><span>or use your email</span></div>}

          <label className="field">
            <span className="field-label">Work email</span>
            <input type="email" autoComplete="username" required value={email} onChange={(event) => setEmail(event.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">Password</span>
            <input type="password" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          <button type="submit" className="button button-primary button-lg login-submit" disabled={submitting}>
            {submitting ? 'Signing in…' : 'Sign in'}
          </button>
          <p className="login-foot">
            <Icon name="lock" size={14} />
            Access is by invitation. New accounts are added by a system administrator.
          </p>
        </form>
      </section>
    </div>
  )
}
