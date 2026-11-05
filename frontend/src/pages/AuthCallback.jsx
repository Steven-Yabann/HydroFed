import { useEffect, useState } from 'react'
import { HydroFedMark } from '../components/Icon'
import { Loading, Notice } from '../components/ui'
import { useAuth } from '../lib/auth'
import { navigate } from '../lib/router'

// The API redirects here after Google or Microsoft sign-in, with the session
// token (or an error) in the URL fragment.
export default function AuthCallback({ query }) {
  const { acceptSession } = useAuth()
  const [error] = useState(query.error || '')

  useEffect(() => {
    if (query.token) {
      acceptSession({ token: query.token, expires_at: query.expires_at }).then(() => {
        navigate('/fleet', { replace: true })
      })
    }
  }, [query.token, query.expires_at, acceptSession])

  return (
    <div className="boot">
      <div className="callback-card">
        <HydroFedMark size={40} />
        {error ? (
          <>
            <Notice tone="critical" title="Sign-in did not complete">{error}</Notice>
            <a className="button button-primary button-md" href="#/login">Back to sign in</a>
          </>
        ) : (
          <Loading label="Finishing sign-in…" />
        )}
      </div>
    </div>
  )
}
