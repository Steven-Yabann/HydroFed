// Stroke icons drawn on a 24px grid. Decorative unless given a label.
const paths = {
  fleet: 'M3 20h18M5 20V9l4 3V9l4 3V6l6 4v10M9 16h1M13 16h1M17 16h1',
  alert: 'M12 3 2.5 20h19L12 3Zm0 6v5m0 3h.01',
  chart: 'M4 20V4M4 20h16M8 16v-4M12 16V8M16 16v-6',
  settings: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7 7 0 0 0-2-1.2L14.5 3h-5l-.4 2.6a7 7 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7 7 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7 7 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2Z',
  nodes: 'M12 5a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM5 23a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm14 0a2 2 0 1 0 0-4 2 2 0 0 0 0 4ZM12 5v6m0 0-6 8m6-8 6 8',
  users: 'M16 20v-1.5a3.5 3.5 0 0 0-3.5-3.5h-5A3.5 3.5 0 0 0 4 18.5V20M10 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm10 9v-1.5a3.5 3.5 0 0 0-2.5-3.3M15 4.2a3.5 3.5 0 0 1 0 6.6',
  logout: 'M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 16l-4-4 4-4M6 12h10',
  check: 'm5 12.5 4.5 4.5L19 7',
  checkCircle: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm-4-9 2.8 2.8L16.5 9',
  eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Zm10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
  octagon: 'M8 2.5h8l5.5 5.5v8L16 21.5H8L2.5 16V8L8 2.5ZM12 8v5m0 3h.01',
  close: 'M6 6l12 12M18 6 6 18',
  menu: 'M4 7h16M4 12h16M4 17h16',
  arrowLeft: 'M19 12H5m6-6-6 6 6 6',
  arrowRight: 'M5 12h14m-6-6 6 6-6 6',
  refresh: 'M20 11a8 8 0 0 0-14.6-4.5L4 8m0-5v5h5M4 13a8 8 0 0 0 14.6 4.5L20 16m0 5v-5h-5',
  play: 'M7 4.5v15l12-7.5-12-7.5Z',
  stop: 'M6 6h12v12H6z',
  plus: 'M12 5v14M5 12h14',
  edit: 'M4 20h4L19 9l-4-4L4 16v4Zm9-13 4 4',
  trash: 'M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3',
  lock: 'M6 11h12v9H6v-9Zm2 0V8a4 4 0 1 1 8 0v3',
  mail: 'M3 6h18v12H3V6Zm0 0 9 7 9-7',
  info: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-10v6m0-9h.01',
  wave: 'M2 15c2.5 0 2.5-3 5-3s2.5 3 5 3 2.5-3 5-3 2.5 3 5 3M2 19c2.5 0 2.5-3 5-3s2.5 3 5 3 2.5-3 5-3 2.5 3 5 3M12 3v6m-3-3 3 3 3-3',
  signal: 'M5 18a10 10 0 0 1 0-12m14 0a10 10 0 0 1 0 12M8.5 15a5 5 0 0 1 0-6m7 0a5 5 0 0 1 0 6M12 12h.01',
  shield: 'M12 3 4 6v6c0 4.5 3.4 8.3 8 9 4.6-.7 8-4.5 8-9V6l-8-3Zm-3 9 2 2 4-4',
  clock: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-13v4l3 2',
  database: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3Zm0 0v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  upload: 'M12 16V4m-5 5 5-5 5 5M4 20h16',
  key: 'M14.5 9.5a4.5 4.5 0 1 1-1.3-3.2M14.5 9.5 21 16v3h-3v-2h-2v-2l-2.8-2.8',
}

export function Icon({ name, size = 18, label, className = '', strokeWidth = 1.8 }) {
  return (
    <svg
      className={`icon ${className}`}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <path d={paths[name] || paths.info} />
    </svg>
  )
}

export function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
      <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5Z" />
      <path fill="#FF3D00" d="m6.3 14.7 6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7Z" />
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2A11.9 11.9 0 0 1 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44Z" />
      <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3a12 12 0 0 1-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5Z" />
    </svg>
  )
}

export function MicrosoftMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 21 21" aria-hidden="true">
      <path fill="#F25022" d="M1 1h9v9H1z" />
      <path fill="#7FBA00" d="M11 1h9v9h-9z" />
      <path fill="#00A4EF" d="M1 11h9v9H1z" />
      <path fill="#FFB900" d="M11 11h9v9h-9z" />
    </svg>
  )
}

export function HydroFedMark({ size = 32 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <rect width="40" height="40" rx="10" fill="#0f5132" />
      <path d="M8 25c3 0 3-3 6-3s3 3 6 3 3-3 6-3 3 3 6 3" stroke="#6ee7b7" strokeWidth="2.4" fill="none" strokeLinecap="round" />
      <path d="M8 30c3 0 3-3 6-3s3 3 6 3 3-3 6-3 3 3 6 3" stroke="#a7f3d0" strokeWidth="2.4" fill="none" strokeLinecap="round" opacity="0.7" />
      <path d="M20 8c-3 4.5-5 7.3-5 9.6a5 5 0 0 0 10 0C25 15.3 23 12.5 20 8Z" fill="#d1fae5" />
    </svg>
  )
}
