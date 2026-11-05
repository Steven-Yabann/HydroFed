import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'

// Load an API path, optionally refreshing on an interval. Polling pauses
// while the tab is hidden so idle dashboards do not keep the server busy.
export function useApi(path, { interval = 0, enabled = true } = {}) {
  const [state, setState] = useState({ data: null, error: '', loading: Boolean(path && enabled) })
  const latestPath = useRef(path)
  useEffect(() => {
    latestPath.current = path
  }, [path])

  const load = useCallback(async ({ quiet = false } = {}) => {
    if (!path || !enabled) return
    if (!quiet) setState((previous) => ({ ...previous, loading: true }))
    try {
      const data = await api(path)
      if (latestPath.current === path) setState({ data, error: '', loading: false })
    } catch (error) {
      if (latestPath.current === path) setState((previous) => ({ ...previous, error: error.message, loading: false }))
    }
  }, [path, enabled])

  useEffect(() => {
    load()
    if (!interval) return undefined
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible') load({ quiet: true })
    }, interval)
    return () => window.clearInterval(timer)
  }, [load, interval])

  return { ...state, reload: load, setData: (data) => setState((previous) => ({ ...previous, data })) }
}

export function useElementWidth() {
  const ref = useRef(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    if (!ref.current) return undefined
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width]
}
