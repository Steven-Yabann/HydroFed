import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { api, onUnauthorised, readSession, saveSession } from './api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [state, setState] = useState(() => ({
    status: readSession() ? 'checking' : 'signed-out',
    user: null,
    notice: '',
  }))

  const signOut = useCallback((notice = '') => {
    saveSession(null)
    setState({ status: 'signed-out', user: null, notice })
  }, [])

  const loadUser = useCallback(async () => {
    try {
      const { user } = await api('/api/auth/me')
      setState({ status: 'signed-in', user, notice: '' })
    } catch (error) {
      if (error.status === 401) signOut(error.message)
      else setState({ status: 'signed-out', user: null, notice: error.message })
    }
  }, [signOut])

  useEffect(() => {
    onUnauthorised((message) => signOut(message || 'Your session has ended. Sign in again.'))
    if (readSession()) loadUser()
  }, [loadUser, signOut])

  const acceptSession = useCallback(async (session) => {
    saveSession(session)
    if (session.user) setState({ status: 'signed-in', user: session.user, notice: '' })
    else await loadUser()
  }, [loadUser])

  const value = useMemo(() => ({
    ...state,
    can: (permission) => Boolean(state.user?.permissions?.includes(permission)),
    acceptSession,
    signOut,
    refreshUser: loadUser,
  }), [state, acceptSession, signOut, loadUser])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// eslint-disable-next-line react/only-export-components
export function useAuth() {
  return useContext(AuthContext)
}
