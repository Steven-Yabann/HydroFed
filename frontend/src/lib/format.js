export const plantStatus = {
  normal: { label: 'Normal', tone: 'normal', description: 'Score well below the alert threshold' },
  watch: { label: 'Watch', tone: 'watch', description: 'Score above half the alert threshold' },
  alarm: { label: 'Alarm', tone: 'critical', description: 'Score above the alert threshold' },
  unknown: { label: 'No data', tone: 'muted', description: 'No reading yet' },
}

export const severityInfo = {
  critical: { label: 'Critical', tone: 'critical', rank: 3 },
  high: { label: 'High', tone: 'serious', rank: 2 },
  medium: { label: 'Medium', tone: 'warning', rank: 1 },
}

export const reviewInfo = {
  open: { label: 'Open', tone: 'serious' },
  acknowledged: { label: 'Acknowledged', tone: 'info' },
  investigating: { label: 'Investigating', tone: 'warning' },
  resolved: { label: 'Resolved', tone: 'normal' },
  false_alarm: { label: 'False alarm', tone: 'muted' },
}

export const faultTypes = {
  valve1: 'Outlet valve closure',
  valve2: 'Inlet valve closure',
  other: 'Rotor, cavitation or fluid fault',
  'anomaly-free': 'Normal operation',
}

export function number(value, digits = 2) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

export function percent(value, digits = 1) {
  return typeof value === 'number' ? `${(value * 100).toFixed(digits)}%` : '—'
}

export function bytes(value) {
  if (typeof value !== 'number') return '—'
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GB`
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e8 ? 0 : 1)} MB`
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)} kB`
  return `${value} B`
}

export function integer(value) {
  return typeof value === 'number' ? Math.round(value).toLocaleString() : '—'
}

export function duration(seconds) {
  if (typeof seconds !== 'number') return '—'
  if (seconds < 60) return `${Math.round(seconds)} s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  if (minutes < 60) return rest ? `${minutes} min ${rest} s` : `${minutes} min`
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`
}

// The API sends Unix seconds for replay times and SQLite UTC strings elsewhere.
export function toDate(value) {
  if (value === null || value === undefined || value === '') return null
  if (typeof value === 'number') return new Date(value * 1000)
  const text = /[zZ]|[+-]\d\d:?\d\d$/.test(value) ? value : `${value.replace(' ', 'T')}Z`
  const date = new Date(text)
  return Number.isNaN(date.getTime()) ? null : date
}

export function clock(value) {
  const date = toDate(value)
  return date ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'
}

export function dateTime(value) {
  const date = toDate(value)
  return date ? date.toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—'
}

export function relative(value) {
  const date = toDate(value)
  if (!date) return '—'
  const seconds = Math.round((Date.now() - date.getTime()) / 1000)
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds} s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`
  return dateTime(value)
}

export function sensorValue(value, unit) {
  if (typeof value !== 'number') return '—'
  const digits = Math.abs(value) >= 100 ? 1 : Math.abs(value) >= 10 ? 2 : 3
  return `${value.toFixed(digits)}${unit ? ` ${unit}` : ''}`
}

export function initials(name = '') {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0].toUpperCase()).join('') || '?'
}
