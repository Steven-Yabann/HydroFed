import { useEffect, useId, useRef } from 'react'
import { plantStatus, reviewInfo, severityInfo } from '../lib/format'
import { Icon } from './Icon'

const toneIcons = {
  normal: 'checkCircle',
  watch: 'eye',
  warning: 'alert',
  serious: 'alert',
  critical: 'octagon',
  info: 'info',
  muted: 'info',
}

// Status is never colour alone: every badge carries an icon and a word.
export function Badge({ tone = 'muted', icon, children, size = 'md' }) {
  return (
    <span className={`badge badge-${tone} badge-${size}`}>
      <Icon name={icon || toneIcons[tone] || 'info'} size={size === 'sm' ? 12 : 14} strokeWidth={2.2} />
      {children}
    </span>
  )
}

export function StatusBadge({ status, size }) {
  const info = plantStatus[status] || plantStatus.unknown
  return <Badge tone={info.tone} size={size}>{info.label}</Badge>
}

export function SeverityBadge({ severity, size }) {
  const info = severityInfo[severity]
  return info ? <Badge tone={info.tone} size={size}>{info.label}</Badge> : null
}

export function ReviewBadge({ status, size }) {
  const info = reviewInfo[status] || reviewInfo.open
  const icon = { open: 'alert', acknowledged: 'eye', investigating: 'clock', resolved: 'check', false_alarm: 'close' }[status]
  return <Badge tone={info.tone} icon={icon} size={size}>{info.label}</Badge>
}

export function Card({ title, subtitle, actions, children, className = '', padded = true, id }) {
  return (
    <section className={`card ${padded ? 'card-padded' : ''} ${className}`} id={id} aria-labelledby={title ? `${id || title}-title` : undefined}>
      {(title || actions) && (
        <header className="card-header">
          <div>
            {title && <h2 id={`${id || title}-title`}>{title}</h2>}
            {subtitle && <p>{subtitle}</p>}
          </div>
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function StatTile({ label, value, detail, tone, icon }) {
  return (
    <article className={`stat-tile ${tone ? `stat-${tone}` : ''}`}>
      <div className="stat-label">
        {icon && <Icon name={icon} size={16} />}
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
      {detail && <span className="stat-detail">{detail}</span>}
    </article>
  )
}

export function Button({ variant = 'secondary', icon, children, className = '', size = 'md', ...props }) {
  return (
    <button type="button" className={`button button-${variant} button-${size} ${className}`} {...props}>
      {icon && <Icon name={icon} size={size === 'sm' ? 14 : 16} />}
      {children && <span>{children}</span>}
    </button>
  )
}

export function PageHeader({ eyebrow, title, description, actions, children }) {
  return (
    <div className="page-header">
      <div className="page-header-text">
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
        {children}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  )
}

export function Notice({ tone = 'info', title, children }) {
  return (
    <div className={`notice notice-${tone}`} role={tone === 'critical' ? 'alert' : 'note'}>
      <Icon name={toneIcons[tone] || 'info'} size={18} />
      <div>
        {title && <strong>{title}</strong>}
        <span>{children}</span>
      </div>
    </div>
  )
}

export function Loading({ label = 'Loading…' }) {
  return (
    <div className="loading" role="status">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({ icon = 'info', title, children, action }) {
  return (
    <div className="empty-state">
      <span className="empty-icon"><Icon name={icon} size={22} /></span>
      <strong>{title}</strong>
      {children && <p>{children}</p>}
      {action}
    </div>
  )
}

export function Segmented({ label, options, value, onChange, size = 'md' }) {
  return (
    <div className={`segmented segmented-${size}`} role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          role="radio"
          aria-checked={value === option.id}
          className={value === option.id ? 'active' : ''}
          onClick={() => onChange(option.id)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function Field({ label, hint, error, children, htmlFor }) {
  return (
    <label className="field" htmlFor={htmlFor}>
      <span className="field-label">{label}</span>
      {children}
      {hint && !error && <span className="field-hint">{hint}</span>}
      {error && <span className="field-error">{error}</span>}
    </label>
  )
}

export function Switch({ checked, onChange, label, disabled }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className={`switch ${checked ? 'on' : ''}`}
      onClick={() => onChange(!checked)}
      disabled={disabled}
    >
      <span />
    </button>
  )
}

// A side panel that traps focus while open and closes on Escape.
export function Drawer({ open, title, subtitle, onClose, children, footer }) {
  const panel = useRef(null)
  const titleId = useId()

  useEffect(() => {
    if (!open) return undefined
    const previous = document.activeElement
    panel.current?.querySelector('input, select, textarea, button')?.focus()
    const onKey = (event) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      previous?.focus?.()
    }
  }, [open, onClose])

  if (!open) return null
  return (
    <div className="drawer-layer">
      <button type="button" className="drawer-scrim" aria-label="Close panel" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-labelledby={titleId} ref={panel}>
        <header className="drawer-header">
          <div>
            <h2 id={titleId}>{title}</h2>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <Button variant="ghost" icon="close" onClick={onClose} aria-label="Close" />
        </header>
        <div className="drawer-body">{children}</div>
        {footer && <footer className="drawer-footer">{footer}</footer>}
      </aside>
    </div>
  )
}

export function ScoreMeter({ score, threshold }) {
  // A bar out to twice the threshold, with the threshold marked at the middle.
  const ratio = typeof score === 'number' ? Math.max(0, Math.min(score / (2 * threshold), 1)) : 0
  const tone = score > threshold ? 'critical' : score > threshold / 2 ? 'watch' : 'normal'
  return (
    <div className="score-meter" aria-hidden="true">
      <div className={`score-meter-fill score-${tone}`} style={{ width: `${ratio * 100}%` }} />
      <div className="score-meter-threshold" />
    </div>
  )
}
