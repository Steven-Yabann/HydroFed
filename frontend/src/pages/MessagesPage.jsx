import { Fragment, useEffect, useMemo, useRef, useState } from 'react'
import { Icon } from '../components/Icon'
import { Button, Card, EmptyState, Loading, Notice, PageHeader, SeverityBadge } from '../components/ui'
import { api } from '../lib/api'
import { useAuth } from '../lib/auth'
import { MESSAGES_READ, useApi } from '../lib/hooks'
import { dateTime, initials, relative, toDate } from '../lib/format'
import { useQueryParam } from '../lib/router'

const MAX_LENGTH = 2000

function timeOfDay(value) {
  const date = toDate(value)
  return date ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''
}

function dayLabel(value) {
  const date = toDate(value)
  if (!date) return ''
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)
  if (date.toDateString() === today.toDateString()) return 'Today'
  if (date.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return date.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })
}

function AlertReference({ alert }) {
  return (
    <a className="message-alert" href={`#/alerts?alert=${encodeURIComponent(alert.id)}`}>
      <SeverityBadge severity={alert.severity} size="sm" />
      <span>Alert from {dateTime(alert.started_at)}</span>
    </a>
  )
}

function ThreadList({ threads, active }) {
  return (
    <Card padded={false} className="thread-list">
      <ul>
        {threads.map((thread) => {
          const last = thread.last_message
          return (
            <li key={thread.plant}>
              <a href={`#/messages/${thread.plant}`} className={active === thread.plant ? 'active' : ''} aria-current={active === thread.plant ? 'page' : undefined}>
                <span className="thread-title">
                  <strong>{thread.plant_name}</strong>
                  {last && <small>{relative(last.created_at)}</small>}
                </span>
                <span className="thread-preview">
                  <span>{last ? `${last.mine ? 'You' : last.author.name}: ${last.body}` : 'No messages yet'}</span>
                  {thread.unread > 0 && <span className="unread-count" aria-label={`${thread.unread} unread`}>{thread.unread}</span>}
                </span>
              </a>
            </li>
          )
        })}
      </ul>
    </Card>
  )
}

function Composer({ plantId, onSent }) {
  const [alertId, setAlertId] = useQueryParam('alert', '')
  const alerts = useApi(alertId ? `/api/alerts?plant=${plantId}&limit=1000` : null)
  const referenced = alerts.data?.alerts.find((alert) => alert.id === alertId)
  const [draft, setDraft] = useState('')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const box = useRef(null)

  useEffect(() => {
    if (alertId) box.current?.focus()
  }, [alertId])

  const send = async (event) => {
    event?.preventDefault()
    if (!draft.trim() || sending) return
    setSending(true)
    setError('')
    try {
      const { message } = await api(`/api/messages/${plantId}`, {
        method: 'POST',
        body: { body: draft, alert_id: referenced ? referenced.id : null },
      })
      setDraft('')
      setAlertId('')
      onSent(message)
    } catch (failure) {
      setError(failure.message)
    } finally {
      setSending(false)
    }
  }

  return (
    <form className="composer" onSubmit={send}>
      {error && <Notice tone="critical">{error}</Notice>}
      {referenced && (
        <div className="composer-alert">
          <AlertReference alert={referenced} />
          <Button size="sm" variant="ghost" icon="close" onClick={() => setAlertId('')} aria-label="Remove the alert from this message" />
        </div>
      )}
      <label htmlFor="message-box" className="sr-only">Message</label>
      <textarea
        id="message-box"
        ref={box}
        rows={3}
        value={draft}
        maxLength={MAX_LENGTH}
        placeholder="Write a message…"
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) send(event)
        }}
      />
      <div className="composer-actions">
        <small className="muted">
          {draft.length > MAX_LENGTH - 200 ? `${MAX_LENGTH - draft.length} characters left` : 'Enter to send, Shift + Enter for a new line'}
        </small>
        <Button type="submit" variant="primary" icon="send" disabled={!draft.trim() || sending}>
          {sending ? 'Sending…' : 'Send'}
        </Button>
      </div>
    </form>
  )
}

function Conversation({ plantId, showBack, onRead }) {
  const thread = useApi(`/api/messages/${plantId}`, { interval: 5000 })
  const messages = useMemo(() => thread.data?.messages || [], [thread.data])
  const lastId = messages.length ? messages[messages.length - 1].id : 0
  const list = useRef(null)

  useEffect(() => {
    if (list.current) list.current.scrollTop = list.current.scrollHeight
    if (lastId) onRead()
  }, [lastId, onRead])

  return (
    <Card padded={false} className="conversation">
      <header className="conversation-header">
        {showBack && <a className="conversation-back" href="#/messages"><Icon name="arrowLeft" size={16} /> All plants</a>}
        <div>
          <h2>{thread.data?.plant_name || 'Conversation'}</h2>
          <p>Seen by this plant's operator and every administrator.</p>
        </div>
      </header>
      <div className="message-scroll" ref={list}>
        {thread.loading && !thread.data ? <Loading /> : thread.error && !thread.data ? (
          <EmptyState icon="message" title="Messages could not be loaded">{thread.error}</EmptyState>
        ) : !messages.length ? (
          <EmptyState icon="message" title="No messages yet">
            Ask a question, report something on site, or follow up on an alert.
          </EmptyState>
        ) : (
          <ol className="message-list">
            {messages.map((message, index) => {
              const day = dayLabel(message.created_at)
              const newDay = index === 0 || day !== dayLabel(messages[index - 1].created_at)
              return (
                <Fragment key={message.id}>
                  {newDay && <li className="message-day"><span>{day}</span></li>}
                  <li className={`message ${message.mine ? 'mine' : ''}`}>
                    {!message.mine && <span className="avatar avatar-sm" aria-hidden="true">{initials(message.author.name)}</span>}
                    <div className="message-bubble">
                      <div className="message-meta">
                        <strong>{message.mine ? 'You' : message.author.name}</strong>
                        <span>{message.author.role_label}</span>
                        <time dateTime={toDate(message.created_at)?.toISOString()} title={dateTime(message.created_at)}>
                          {timeOfDay(message.created_at)}
                        </time>
                      </div>
                      {message.alert && <AlertReference alert={message.alert} />}
                      <p>{message.body}</p>
                    </div>
                  </li>
                </Fragment>
              )
            })}
          </ol>
        )}
      </div>
      <Composer
        plantId={plantId}
        onSent={(message) => thread.setData({ ...thread.data, messages: [...messages, message] })}
      />
    </Card>
  )
}

export default function MessagesPage({ plantId }) {
  const { user, can } = useAuth()
  const seesAll = can('view_all_plants')
  const threads = useApi(seesAll ? '/api/messages' : null, { interval: 10000 })
  const active = plantId || (seesAll ? null : user.plant_id)
  const reloadThreads = threads.reload

  const onRead = useMemo(() => () => {
    if (seesAll) reloadThreads({ quiet: true })
    window.dispatchEvent(new Event(MESSAGES_READ))
  }, [seesAll, reloadThreads])

  if (!seesAll && !user.plant_id) {
    return (
      <div className="forbidden">
        <Icon name="message" size={28} />
        <h1>No plant assigned yet</h1>
        <p>Messages belong to a plant. An administrator needs to assign yours first.</p>
      </div>
    )
  }

  return (
    <div className="page">
      <PageHeader
        title="Messages"
        description={seesAll
          ? 'Each plant has one conversation with its operator. Every administrator can read and reply.'
          : 'Talk to the administrators about your plant. Every administrator can read and reply.'}
      />
      <div className={`messages-layout ${seesAll ? 'with-list' : ''} ${active ? 'has-thread' : ''}`}>
        {seesAll && (threads.loading && !threads.data ? <Card><Loading /></Card> : threads.error && !threads.data ? (
          <Card><EmptyState icon="message" title="Conversations could not be loaded">{threads.error}</EmptyState></Card>
        ) : <ThreadList threads={threads.data?.threads || []} active={active} />)}
        {active ? (
          <Conversation key={active} plantId={active} showBack={seesAll} onRead={onRead} />
        ) : (
          <Card className="conversation-placeholder">
            <EmptyState icon="message" title="Choose a plant">Pick a conversation from the list to read and reply.</EmptyState>
          </Card>
        )}
      </div>
    </div>
  )
}
