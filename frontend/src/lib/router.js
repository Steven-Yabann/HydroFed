import { useCallback, useEffect, useState } from 'react'

// A small hash router: #/plants/plant_2?window=1800 becomes
// { path: '/plants/plant_2', segments: ['plants', 'plant_2'], query }.
// Hash routes work on any static host without server rewrites.

function parse() {
  const raw = window.location.hash.replace(/^#/, '') || '/'
  const [pathPart, queryPart = ''] = raw.split('?')
  const path = pathPart.startsWith('/') ? pathPart : `/${pathPart}`
  return {
    path,
    segments: path.split('/').filter(Boolean),
    query: Object.fromEntries(new URLSearchParams(queryPart)),
  }
}

export function navigate(path, { replace = false } = {}) {
  const target = `#${path}`
  if (replace) window.history.replaceState(null, '', target)
  else window.history.pushState(null, '', target)
  window.dispatchEvent(new HashChangeEvent('hashchange'))
}

export function useRoute() {
  const [route, setRoute] = useState(parse)
  useEffect(() => {
    const update = () => setRoute(parse())
    window.addEventListener('hashchange', update)
    window.addEventListener('popstate', update)
    return () => {
      window.removeEventListener('hashchange', update)
      window.removeEventListener('popstate', update)
    }
  }, [])
  return route
}

export function useQueryParam(name, fallback) {
  const route = useRoute()
  const value = route.query[name] ?? fallback
  const setValue = useCallback((next) => {
    const query = new URLSearchParams(parse().query)
    if (next === fallback || next === undefined || next === null) query.delete(name)
    else query.set(name, next)
    const search = query.toString()
    navigate(`${parse().path}${search ? `?${search}` : ''}`, { replace: true })
  }, [name, fallback])
  return [value, setValue]
}
