import { useMemo, useState } from 'react'
import { clock } from '../lib/format'
import { useElementWidth } from '../lib/hooks'

// Hand-built SVG charts. Marks are thin, gridlines are solid hairlines and
// every chart has a hover layer. Series colours come from CSS custom
// properties (--series-*) so they follow the entity, not its rank.

function linear(domain, range) {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0 || 1
  const scale = (value) => r0 + ((value - d0) / span) * (r1 - r0)
  scale.invert = (pixel) => d0 + ((pixel - r0) / (r1 - r0)) * span
  return scale
}

function niceTicks(min, max, count = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return []
  if (min === max) return [min]
  const raw = (max - min) / count
  const power = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((candidate) => candidate >= raw) || raw
  const ticks = []
  for (let tick = Math.ceil(min / step) * step; tick <= max + step * 1e-9; tick += step) {
    ticks.push(Number(tick.toFixed(10)))
  }
  ticks.step = step
  return ticks
}

// Show as many decimals as the tick spacing needs, so 88.5 and 89 stay distinct.
function formatTick(value, step) {
  const magnitude = Math.abs(value)
  if (magnitude >= 10000) return `${(value / 1000).toFixed(0)}k`
  if (step) {
    const decimals = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)))
    return value.toFixed(decimals)
  }
  if (magnitude >= 10 || value === 0) return value.toFixed(0)
  if (magnitude >= 1) return value.toFixed(1)
  return value.toFixed(2)
}

function runs(flags) {
  // Contiguous [start, end] index ranges where flags are truthy.
  const ranges = []
  let start = null
  flags.forEach((flag, index) => {
    if (flag && start === null) start = index
    if (!flag && start !== null) {
      ranges.push([start, index - 1])
      start = null
    }
  })
  if (start !== null) ranges.push([start, flags.length - 1])
  return ranges
}

function Tooltip({ x, y, width, children }) {
  const left = Math.min(Math.max(x + 12, 4), width - 184)
  return (
    <div className="chart-tooltip" style={{ left, top: Math.max(4, y - 8) }}>
      {children}
    </div>
  )
}

/**
 * Time series with an optional threshold and shaded alert periods.
 * `times` are Unix seconds; `series` is [{ key, label, values, color }].
 */
export function TimeSeriesChart({
  times,
  series,
  height = 220,
  threshold,
  thresholdLabel = 'Alert threshold',
  alerts,
  labels,
  unit = '',
  yLabel,
  compact = false,
  formatValue = (value) => formatTick(value),
}) {
  const [ref, width] = useElementWidth()
  const [hover, setHover] = useState(null)
  const margin = compact ? { top: 8, right: 8, bottom: 20, left: 40 } : { top: 12, right: 16, bottom: 28, left: 48 }
  const plotWidth = Math.max(0, width - margin.left - margin.right)
  const plotHeight = height - margin.top - margin.bottom

  const geometry = useMemo(() => {
    if (!times?.length || !plotWidth) return null
    const values = series.flatMap((item) => item.values).filter(Number.isFinite)
    if (threshold !== undefined) values.push(threshold)
    let low = Math.min(...values)
    let high = Math.max(...values)
    const pad = (high - low) * 0.08 || Math.abs(high) * 0.1 || 1
    low -= pad
    high += pad
    const x = linear([times[0], times[times.length - 1]], [0, plotWidth])
    const y = linear([low, high], [plotHeight, 0])
    const paths = series.map((item) => {
      let path = ''
      item.values.forEach((value, index) => {
        if (!Number.isFinite(value)) return
        path += `${path ? 'L' : 'M'}${x(times[index]).toFixed(1)},${y(value).toFixed(1)}`
      })
      return { ...item, path }
    })
    const step = times.length > 1 ? x(times[1]) - x(times[0]) : plotWidth
    const band = (ranges) => ranges.map(([start, end]) => ({
      x: x(times[start]) - step / 2,
      width: Math.max(2, x(times[end]) - x(times[start]) + step),
    }))
    const timeTicks = niceTimeTicks(times[0], times[times.length - 1], compact ? 3 : 5)
    return {
      x, y, paths,
      yTicks: niceTicks(low, high, compact ? 3 : 4),
      timeTicks,
      alertBands: alerts ? band(runs(alerts)) : [],
      labelBands: labels ? band(runs(labels)) : [],
    }
  }, [times, series, threshold, alerts, labels, plotWidth, plotHeight, compact])

  const onMove = (event) => {
    if (!geometry) return
    const box = event.currentTarget.getBoundingClientRect()
    const time = geometry.x.invert(event.clientX - box.left)
    let index = 0
    let lowIndex = 0
    let highIndex = times.length - 1
    while (lowIndex <= highIndex) {
      const middle = (lowIndex + highIndex) >> 1
      if (times[middle] < time) lowIndex = middle + 1
      else highIndex = middle - 1
    }
    index = Math.min(times.length - 1, Math.max(0, lowIndex))
    if (index > 0 && Math.abs(times[index - 1] - time) < Math.abs(times[index] - time)) index -= 1
    setHover(index)
  }

  return (
    <div className="chart" ref={ref} style={{ height }}>
      {geometry && (
        <svg width={width} height={height} role="img" aria-label={`${series.map((item) => item.label).join(', ')} over time`}>
          <g transform={`translate(${margin.left},${margin.top})`}>
            {geometry.labelBands.map((band, index) => (
              <rect key={`label-${index}`} className="chart-label-band" x={band.x} y={0} width={band.width} height={plotHeight} />
            ))}
            {geometry.alertBands.map((band, index) => (
              <rect key={`alert-${index}`} className="chart-alert-band" x={band.x} y={0} width={band.width} height={plotHeight} />
            ))}
            {geometry.yTicks.map((tick) => (
              <g key={tick} transform={`translate(0,${geometry.y(tick)})`}>
                <line className="chart-grid" x2={plotWidth} />
                <text className="chart-tick" x={-8} dy="0.32em" textAnchor="end">{formatTick(tick, geometry.yTicks.step)}</text>
              </g>
            ))}
            {geometry.timeTicks.map((tick) => (
              <text key={tick} className="chart-tick" x={geometry.x(tick)} y={plotHeight + (compact ? 14 : 18)} textAnchor="middle">
                {new Date(tick * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </text>
            ))}
            <line className="chart-axis" y1={plotHeight} y2={plotHeight} x2={plotWidth} />
            {threshold !== undefined && (
              <g>
                <line className="chart-threshold" y1={geometry.y(threshold)} y2={geometry.y(threshold)} x2={plotWidth} />
                {!compact && (
                  <text className="chart-threshold-label" x={plotWidth - 4} y={geometry.y(threshold) - 6} textAnchor="end">
                    {thresholdLabel} · {formatTick(threshold)}
                  </text>
                )}
              </g>
            )}
            {geometry.paths.map((item) => (
              <path key={item.key} d={item.path} className={`chart-line ${compact ? 'chart-line-thin' : ''}`} style={{ stroke: item.color }} />
            ))}
            {hover !== null && (
              <g>
                <line className="chart-crosshair" x1={geometry.x(times[hover])} x2={geometry.x(times[hover])} y2={plotHeight} />
                {series.map((item) => Number.isFinite(item.values[hover]) && (
                  <circle
                    key={item.key}
                    className="chart-point"
                    cx={geometry.x(times[hover])}
                    cy={geometry.y(item.values[hover])}
                    r={4}
                    style={{ fill: item.color }}
                  />
                ))}
              </g>
            )}
            {yLabel && !compact && (
              <text className="chart-axis-label" transform={`translate(${-margin.left + 12},${plotHeight / 2}) rotate(-90)`} textAnchor="middle">
                {yLabel}
              </text>
            )}
            <rect
              className="chart-hit"
              width={plotWidth}
              height={plotHeight}
              onMouseMove={onMove}
              onMouseLeave={() => setHover(null)}
            />
          </g>
        </svg>
      )}
      {geometry && hover !== null && (
        <Tooltip x={margin.left + geometry.x(times[hover])} y={margin.top} width={width}>
          <span className="tooltip-title">{clock(times[hover])}</span>
          {series.map((item) => (
            <span key={item.key} className="tooltip-row">
              <i style={{ background: item.color }} />
              {item.label}
              <b>{Number.isFinite(item.values[hover]) ? `${formatValue(item.values[hover])}${unit ? ` ${unit}` : ''}` : '—'}</b>
            </span>
          ))}
          {alerts?.[hover] ? <span className="tooltip-flag tooltip-alert">Alert raised</span> : null}
          {labels?.[hover] ? <span className="tooltip-flag">Labelled fault in SKAB</span> : null}
        </Tooltip>
      )}
    </div>
  )
}

function niceTimeTicks(start, end, count) {
  const span = end - start
  if (span <= 0) return [start]
  const steps = [15, 30, 60, 120, 300, 600, 900, 1800, 3600]
  const step = steps.find((candidate) => span / candidate <= count) || 3600
  const ticks = []
  for (let tick = Math.ceil(start / step) * step; tick <= end; tick += step) ticks.push(tick)
  return ticks
}

export function Sparkline({ values, threshold, height = 36, label }) {
  const [ref, width] = useElementWidth()
  const finite = values.filter(Number.isFinite)
  if (!finite.length) return <div ref={ref} className="sparkline" style={{ height }} />
  const high = Math.max(...finite, threshold ?? -Infinity)
  const low = Math.min(...finite, 0)
  const x = linear([0, values.length - 1], [1, Math.max(1, width - 1)])
  const y = linear([low, high * 1.05 || 1], [height - 2, 2])
  const path = values.map((value, index) => `${index ? 'L' : 'M'}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join('')
  const over = threshold !== undefined && values[values.length - 1] > threshold
  return (
    <div ref={ref} className="sparkline" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={label}>
          {threshold !== undefined && <line className="chart-threshold" x1={0} x2={width} y1={y(threshold)} y2={y(threshold)} />}
          <path d={path} className={`sparkline-path ${over ? 'over' : ''}`} />
          <circle cx={x(values.length - 1)} cy={y(values[values.length - 1])} r={3} className={`sparkline-end ${over ? 'over' : ''}`} />
        </svg>
      )}
    </div>
  )
}

/**
 * Horizontal bars of mean ± sd, one row per entity.
 * rows: [{ id, label, value, sd, color, note }]
 */
export function BarChart({ rows, domain, format = (value) => value.toFixed(3), reference, referenceLabel, unitLabel }) {
  const [ref, width] = useElementWidth()
  const [hover, setHover] = useState(null)
  const labelWidth = Math.min(200, Math.max(120, width * 0.32))
  const rowHeight = 30
  const margin = { top: reference !== undefined ? 22 : 6, right: 24, bottom: 26 }
  const plotWidth = Math.max(0, width - labelWidth - margin.right)
  const height = margin.top + rows.length * rowHeight + margin.bottom
  const max = domain?.[1] ?? Math.max(...rows.map((row) => row.value + (row.sd || 0))) * 1.08
  const min = domain?.[0] ?? 0
  const x = linear([min, max], [0, plotWidth])
  const ticks = niceTicks(min, max, 4)

  return (
    <div className="chart" ref={ref} style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={rows.map((row) => `${row.label} ${format(row.value)}`).join('; ')}>
          <g transform={`translate(${labelWidth},${margin.top})`}>
            {ticks.map((tick) => (
              <g key={tick} transform={`translate(${x(tick)},0)`}>
                <line className="chart-grid" y2={rows.length * rowHeight} />
                <text className="chart-tick" y={rows.length * rowHeight + 16} textAnchor="middle">{unitLabel ? unitLabel(tick) : formatTick(tick)}</text>
              </g>
            ))}
            {reference !== undefined && (
              <g transform={`translate(${x(reference)},0)`}>
                <line className="chart-threshold" y1={-6} y2={rows.length * rowHeight} />
                <text className="chart-threshold-label" y={-10} textAnchor="middle">{referenceLabel}</text>
              </g>
            )}
            {rows.map((row, index) => {
              const top = index * rowHeight + 7
              const barWidth = Math.max(2, x(row.value) - x(min))
              return (
                <g key={row.id} onMouseEnter={() => setHover(index)} onMouseLeave={() => setHover(null)}>
                  <rect className="chart-hit" x={-labelWidth} y={index * rowHeight} width={labelWidth + plotWidth} height={rowHeight} />
                  <text className="chart-row-label" x={-12} y={top + 9} dy="0.32em" textAnchor="end">{row.label}</text>
                  <rect x={0} y={top} width={barWidth} height={16} rx={4} className={`chart-bar ${hover === index ? 'hover' : ''}`} style={{ fill: row.color }} />
                  {row.sd > 0 && (
                    <g className="chart-error">
                      <line x1={x(Math.max(min, row.value - row.sd))} x2={x(row.value + row.sd)} y1={top + 8} y2={top + 8} />
                      <line x1={x(row.value + row.sd)} x2={x(row.value + row.sd)} y1={top + 3} y2={top + 13} />
                    </g>
                  )}
                </g>
              )
            })}
          </g>
        </svg>
      )}
      {hover !== null && rows[hover] && (
        <Tooltip x={labelWidth + x(rows[hover].value)} y={margin.top + hover * rowHeight} width={width}>
          <span className="tooltip-title">{rows[hover].label}</span>
          <span className="tooltip-row">
            <i style={{ background: rows[hover].color }} />
            {format(rows[hover].value)}
            {rows[hover].sd > 0 && <b>± {format(rows[hover].sd)}</b>}
          </span>
          {rows[hover].note && <span className="tooltip-note">{rows[hover].note}</span>}
        </Tooltip>
      )}
    </div>
  )
}

/**
 * Dot plot: one row per category, one dot per series on a shared axis.
 * Good for close values such as AUCs between 0.75 and 0.9.
 */
export function DotPlot({ rows, series, domain, format = (value) => value.toFixed(3), reference, referenceLabel }) {
  const [ref, width] = useElementWidth()
  const [hover, setHover] = useState(null)
  const labelWidth = Math.min(210, Math.max(130, width * 0.34))
  const rowHeight = 34
  const margin = { top: reference !== undefined ? 22 : 8, right: 20, bottom: 26 }
  const plotWidth = Math.max(0, width - labelWidth - margin.right)
  const height = margin.top + rows.length * rowHeight + margin.bottom
  const x = linear(domain, [0, plotWidth])
  const ticks = niceTicks(domain[0], domain[1], 5)

  return (
    <div className="chart" ref={ref} style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Comparison dot plot">
          <g transform={`translate(${labelWidth},${margin.top})`}>
            {ticks.map((tick) => (
              <g key={tick} transform={`translate(${x(tick)},0)`}>
                <line className="chart-grid" y2={rows.length * rowHeight} />
                <text className="chart-tick" y={rows.length * rowHeight + 16} textAnchor="middle">{formatTick(tick)}</text>
              </g>
            ))}
            {reference !== undefined && (
              <g transform={`translate(${x(reference)},0)`}>
                <line className="chart-threshold" y1={-6} y2={rows.length * rowHeight} />
                <text className="chart-threshold-label" y={-10} textAnchor="middle">{referenceLabel}</text>
              </g>
            )}
            {rows.map((row, rowIndex) => {
              const cy = rowIndex * rowHeight + rowHeight / 2
              const present = series.filter((item) => Number.isFinite(row.values[item.id]))
              const xs = present.map((item) => x(row.values[item.id]))
              return (
                <g key={row.id}>
                  <text className="chart-row-label" x={-12} y={cy} dy="0.32em" textAnchor="end">{row.label}</text>
                  {xs.length > 1 && <line className="chart-range" x1={Math.min(...xs)} x2={Math.max(...xs)} y1={cy} y2={cy} />}
                  {present.map((item) => (
                    <circle
                      key={item.id}
                      cx={x(row.values[item.id])}
                      cy={cy}
                      r={hover?.row === rowIndex && hover?.series === item.id ? 7 : 5.5}
                      className="chart-dot"
                      style={{ fill: item.color }}
                      onMouseEnter={() => setHover({ row: rowIndex, series: item.id })}
                      onMouseLeave={() => setHover(null)}
                    />
                  ))}
                </g>
              )
            })}
          </g>
        </svg>
      )}
      {hover && (
        <Tooltip x={labelWidth + x(rows[hover.row].values[hover.series])} y={margin.top + hover.row * rowHeight} width={width}>
          <span className="tooltip-title">{rows[hover.row].label}</span>
          {series.filter((item) => Number.isFinite(rows[hover.row].values[item.id])).map((item) => (
            <span key={item.id} className={`tooltip-row ${item.id === hover.series ? 'strong' : ''}`}>
              <i style={{ background: item.color }} />
              {item.label}
              <b>{format(rows[hover.row].values[item.id])}</b>
            </span>
          ))}
        </Tooltip>
      )}
    </div>
  )
}

/** Simple x/y line chart for non-time data, such as bytes over days. */
export function LineChart({ series, xLabel, yFormat = formatTick, xFormat = formatTick, height = 240, markers = [] }) {
  const [ref, width] = useElementWidth()
  const [hover, setHover] = useState(null)
  const margin = { top: 14, right: 20, bottom: 40, left: 56 }
  const plotWidth = Math.max(0, width - margin.left - margin.right)
  const plotHeight = height - margin.top - margin.bottom
  const points = series.flatMap((item) => item.points)
  const xMax = Math.max(...points.map((point) => point[0]))
  const yMax = Math.max(...points.map((point) => point[1])) * 1.08
  const x = linear([0, xMax], [0, plotWidth])
  const y = linear([0, yMax], [plotHeight, 0])
  const xs = [...new Set(series[0]?.points.map((point) => point[0]))]

  const onMove = (event) => {
    const box = event.currentTarget.getBoundingClientRect()
    const value = x.invert(event.clientX - box.left)
    const nearest = xs.reduce((best, candidate) => (Math.abs(candidate - value) < Math.abs(best - value) ? candidate : best), xs[0])
    setHover(nearest)
  }

  return (
    <div className="chart" ref={ref} style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={series.map((item) => item.label).join(' and ')}>
          <g transform={`translate(${margin.left},${margin.top})`}>
            {niceTicks(0, yMax, 4).map((tick) => (
              <g key={tick} transform={`translate(0,${y(tick)})`}>
                <line className="chart-grid" x2={plotWidth} />
                <text className="chart-tick" x={-8} dy="0.32em" textAnchor="end">{yFormat(tick)}</text>
              </g>
            ))}
            {niceTicks(0, xMax, 6).map((tick) => (
              <text key={tick} className="chart-tick" x={x(tick)} y={plotHeight + 18} textAnchor="middle">{xFormat(tick)}</text>
            ))}
            <line className="chart-axis" y1={plotHeight} y2={plotHeight} x2={plotWidth} />
            <text className="chart-axis-label" x={plotWidth / 2} y={plotHeight + 34} textAnchor="middle">{xLabel}</text>
            {markers.map((marker) => (
              <g key={marker.label} transform={`translate(${x(marker.x)},0)`}>
                <line className="chart-threshold" y2={plotHeight} />
                <text className="chart-threshold-label" x={6} y={12}>{marker.label}</text>
              </g>
            ))}
            {series.map((item) => (
              <g key={item.id}>
                <path
                  className="chart-line"
                  style={{ stroke: item.color }}
                  d={item.points.map((point, index) => `${index ? 'L' : 'M'}${x(point[0])},${y(point[1])}`).join('')}
                />
                <text className="chart-direct-label" x={x(item.points.at(-1)[0]) - 4} y={y(item.points.at(-1)[1]) - 8} textAnchor="end">
                  {item.label}
                </text>
              </g>
            ))}
            {hover !== null && (
              <g>
                <line className="chart-crosshair" x1={x(hover)} x2={x(hover)} y2={plotHeight} />
                {series.map((item) => {
                  const point = item.points.find((candidate) => candidate[0] === hover)
                  return point && <circle key={item.id} className="chart-point" cx={x(point[0])} cy={y(point[1])} r={4} style={{ fill: item.color }} />
                })}
              </g>
            )}
            <rect className="chart-hit" width={plotWidth} height={plotHeight} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
          </g>
        </svg>
      )}
      {hover !== null && (
        <Tooltip x={margin.left + x(hover)} y={margin.top} width={width}>
          <span className="tooltip-title">{xFormat(hover)} {xLabel?.toLowerCase()}</span>
          {series.map((item) => {
            const point = item.points.find((candidate) => candidate[0] === hover)
            return point && (
              <span key={item.id} className="tooltip-row">
                <i style={{ background: item.color }} />
                {item.label}
                <b>{yFormat(point[1])}</b>
              </span>
            )
          })}
        </Tooltip>
      )}
    </div>
  )
}

export function Legend({ items }) {
  return (
    <ul className="legend">
      {items.map((item) => (
        <li key={item.id || item.label}>
          <i className={item.shape ? `legend-${item.shape}` : ''} style={{ background: item.color }} />
          {item.label}
        </li>
      ))}
    </ul>
  )
}
