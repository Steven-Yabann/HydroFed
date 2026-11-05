import { useEffect, useState } from 'react'
import { Icon, HydroFedMark } from './components/Icon'
import { Button, Loading } from './components/ui'
import { useAuth } from './lib/auth'
import { initials } from './lib/format'
import { navigate, useRoute } from './lib/router'
import AccountDrawer from './pages/AccountDrawer'
import AlertsPage from './pages/AlertsPage'
import AuthCallback from './pages/AuthCallback'
import FederationPage from './pages/admin/FederationPage'
import NodesPage from './pages/admin/NodesPage'
import UsersPage from './pages/admin/UsersPage'
import FleetPage from './pages/FleetPage'
import LoginPage from './pages/LoginPage'
import PerformancePage from './pages/PerformancePage'
import PlantPage from './pages/PlantPage'
import './App.css'

const navigation = [
  {
    group: 'Operations',
    items: [
      { path: '/fleet', label: 'Fleet overview', icon: 'fleet', permission: 'view_fleet' },
      { path: '/alerts', label: 'Alerts', icon: 'alert', permission: 'review_alerts' },
    ],
  },
  {
    group: 'Insights',
    items: [
      { path: '/performance', label: 'Model performance', icon: 'chart', permission: 'view_performance' },
    ],
  },
  {
    group: 'Administration',
    items: [
      { path: '/admin/federation', label: 'Federated training', icon: 'settings', permission: 'manage_federation' },
      { path: '/admin/nodes', label: 'Plant nodes', icon: 'nodes', permission: 'manage_nodes' },
      { path: '/admin/users', label: 'Users & access', icon: 'users', permission: 'manage_users' },
    ],
  },
]

const pages = [
  { match: (s) => s[0] === 'fleet', permission: 'view_fleet', render: () => <FleetPage /> },
  { match: (s) => s[0] === 'plants' && s[1], permission: 'view_fleet', render: (s) => <PlantPage plantId={s[1]} /> },
  { match: (s) => s[0] === 'alerts', permission: 'review_alerts', render: () => <AlertsPage /> },
  { match: (s) => s[0] === 'performance', permission: 'view_performance', render: () => <PerformancePage /> },
  { match: (s) => s[0] === 'admin' && s[1] === 'federation', permission: 'manage_federation', render: () => <FederationPage /> },
  { match: (s) => s[0] === 'admin' && s[1] === 'nodes', permission: 'manage_nodes', render: () => <NodesPage /> },
  { match: (s) => s[0] === 'admin' && s[1] === 'users', permission: 'manage_users', render: () => <UsersPage /> },
]

function Sidebar({ route, open, onClose, onAccount }) {
  const { user, can } = useAuth()
  return (
    <>
      {open && <button type="button" className="sidebar-scrim" aria-label="Close menu" onClick={onClose} />}
      <aside className={`sidebar ${open ? 'open' : ''}`} aria-label="Main navigation">
        <div className="sidebar-layers" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <a className="brand" href="#/fleet" onClick={onClose}>
          <HydroFedMark size={36} />
          <span>
            <strong>HydroFed</strong>
            <small>Federated plant monitoring</small>
          </span>
        </a>
        <nav>
          {navigation.map((section) => {
            const items = section.items.filter((item) => can(item.permission))
            if (!items.length) return null
            return (
              <div key={section.group} className="nav-group">
                <p>{section.group}</p>
                {items.map((item) => {
                  const active = route.path === item.path
                    || (item.path === '/fleet' && route.segments[0] === 'plants')
                  return (
                    <a
                      key={item.path}
                      href={`#${item.path}`}
                      className={active ? 'active' : ''}
                      aria-current={active ? 'page' : undefined}
                      onClick={onClose}
                    >
                      <Icon name={item.icon} size={18} />
                      {item.label}
                    </a>
                  )
                })}
              </div>
            )
          })}
        </nav>
        <button type="button" className="sidebar-user" onClick={onAccount}>
          <span className="avatar">{initials(user.name)}</span>
          <span>
            <strong>{user.name}</strong>
            <small>{user.role_label}</small>
          </span>
          <Icon name="settings" size={16} />
        </button>
      </aside>
    </>
  )
}

function Shell({ route }) {
  const { can } = useAuth()
  const [menuOpen, setMenuOpen] = useState(false)
  const [accountOpen, setAccountOpen] = useState(false)
  const page = pages.find((candidate) => candidate.match(route.segments))

  useEffect(() => {
    if (!page) navigate('/fleet', { replace: true })
  }, [page])

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [route.path])

  return (
    <div className="shell">
      <a className="skip-link" href="#main">Skip to content</a>
      <Sidebar route={route} open={menuOpen} onClose={() => setMenuOpen(false)} onAccount={() => setAccountOpen(true)} />
      <div className="shell-main">
        <header className="topbar">
          <Button variant="ghost" icon="menu" className="menu-button" onClick={() => setMenuOpen(true)} aria-label="Open menu" />
          <a className="topbar-brand" href="#/fleet"><HydroFedMark size={28} /> HydroFed</a>
          <span className="advisory">
            <Icon name="shield" size={14} />
            Advisory monitoring · does not control plant equipment
          </span>
        </header>
        <main id="main" tabIndex={-1}>
          {!page ? <Loading /> : can(page.permission) ? page.render(route.segments) : (
            <div className="forbidden">
              <Icon name="lock" size={28} />
              <h1>This page is for administrators</h1>
              <p>Your role does not include access to it. Ask an administrator if you need it.</p>
              <Button variant="primary" onClick={() => navigate('/fleet')}>Back to the fleet</Button>
            </div>
          )}
        </main>
      </div>
      <AccountDrawer open={accountOpen} onClose={() => setAccountOpen(false)} />
    </div>
  )
}

export default function App() {
  const { status } = useAuth()
  const route = useRoute()

  if (route.path === '/auth/callback') return <AuthCallback query={route.query} />
  if (status === 'checking') return <div className="boot"><Loading label="Opening HydroFed…" /></div>
  if (status !== 'signed-in') return <LoginPage />
  if (route.path === '/' || route.path === '/login') {
    navigate('/fleet', { replace: true })
    return null
  }
  return <Shell route={route} />
}
