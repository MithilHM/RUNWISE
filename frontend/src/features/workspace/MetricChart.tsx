import { useMemo } from 'react'
import type { MetricValue } from '../../api/types'
import { formatMetric, titleCase } from '../../lib/format'

interface MetricChartProps {
  experiments: { id: string; name: string; model_family?: string | null; metrics: MetricValue[] }[]
  metricName: string
  limit?: number
}

const FAMILY_COLORS = [
  'var(--color-cat-1)',
  'var(--color-cat-2)',
  'var(--color-cat-3)',
  'var(--color-cat-4)',
  'var(--color-cat-5)',
  'var(--color-cat-6)',
]

const ROW = 22
const GUTTER = 210
const PADDING_RIGHT = 64
const AXIS = 30
/* Matches the rendered width of the panel on desktop so the viewBox scale factor
   stays near 1. Below this the chart scrolls rather than shrinking its own text
   into illegibility, because a chart you cannot read is not a compact chart. */
const VIEW_W = 1400
const VIEW_W_MIN = 760

function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min
  if (!(span > 0)) return [min]

  const rawStep = span / (count - 1)
  const magnitude = 10 ** Math.floor(Math.log10(rawStep))
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rawStep) ?? magnitude * 10

  const ticks: number[] = []
  // Anchored on the low end and stepped up, so ticks are never emitted below the
  // plotted range. A gridline outside the data, or running up through the name
  // gutter, misrepresents where the scale starts.
  const first = Math.ceil(min / step) * step
  for (let t = first, i = 0; t <= max + step / 1000 && i < 40; t += step, i += 1) {
    ticks.push(Number(t.toFixed(10)))
  }
  return ticks
}

/** Decimal places needed to distinguish adjacent ticks at this step. */
function decimalsFor(step: number): number {
  if (!(step > 0)) return 2
  return Math.min(6, Math.max(0, Math.ceil(-Math.log10(step))))
}

/**
 * Metric comparison across runs.
 *
 * A dot plot rather than bars, because these scores sit in a narrow band near the
 * top of their range. Bars would need either a truncated baseline, which
 * overstates the gap between runs, or a zero baseline, which flattens the
 * differences the researcher is looking for. A dot carries position without
 * implying magnitude, so the axis can be zoomed honestly.
 *
 * A run with no recorded value for the selected metric is listed as unrecorded,
 * never plotted at zero. A blank row means unmeasured; plotting it at 0 would
 * make an unmeasured run look like the worst possible result.
 */
export function MetricChart({ experiments, metricName, limit = 12 }: MetricChartProps) {
  const direction = useMemo(() => {
    const recorded = experiments.find((e) => e.metrics.some((m) => m.name === metricName))
    return recorded?.metrics.find((m) => m.name === metricName)?.direction ?? 'maximize'
  }, [experiments, metricName])

  const rows = useMemo(() => {
    const families = Array.from(
      new Set(
        experiments
          .map((e) => e.model_family)
          .filter((f): f is string => Boolean(f))
          .sort(),
      ),
    )
    const colorFor = (family?: string | null) =>
      family
        ? FAMILY_COLORS[Math.max(0, families.indexOf(family)) % FAMILY_COLORS.length]
        : 'var(--color-ink-500)'

    const shape = (experiment: (typeof experiments)[number]) => {
      const metric = experiment.metrics.find((m) => m.name === metricName)
      return {
        id: experiment.id,
        label: experiment.name.length > 26 ? `${experiment.name.slice(0, 25)}…` : experiment.name,
        color: colorFor(experiment.model_family),
        value: metric?.value ?? null,
        display: formatMetric({
          value: metric?.value ?? null,
          direction: metric?.direction ?? direction,
        }),
      }
    }

    /* Runs that actually recorded the metric come first. Taking the first `limit`
       of whatever order the API returned meant a metric held only by a late run
       produced an all-null window and the chart vanished. */
    const measured = experiments.filter((e) =>
      e.metrics.some((m) => m.name === metricName && m.value !== null),
    )
    const unmeasured = experiments.filter(
      (e) => !e.metrics.some((m) => m.name === metricName && m.value !== null),
    )

    return [...measured, ...unmeasured].slice(0, limit).map(shape)
  }, [experiments, metricName, limit, direction])

  const plot = useMemo(() => {
    const measured = rows.map((r) => r.value).filter((v): v is number => v !== null)
    if (!measured.length) return null

    const min = Math.min(...measured)
    const max = Math.max(...measured)
    const pad = (max - min || Math.abs(max) * 0.05 || 0.01) * 0.18
    const lo = min - pad
    const hi = max + pad

    const ticks = niceTicks(lo, hi)
    const tickStep = ticks.length > 1 ? ticks[1] - ticks[0] : hi - lo
    const decimals = decimalsFor(tickStep)
    const mean = measured.reduce((a, b) => a + b, 0) / measured.length

    const plotLeft = GUTTER
    const plotRight = VIEW_W - PADDING_RIGHT
    const x = (value: number) => plotLeft + ((value - lo) / (hi - lo)) * (plotRight - plotLeft)

    return {
      lo,
      hi,
      ticks,
      decimals,
      mean,
      height: rows.length * ROW + AXIS,
      plotLeft,
      plotRight,
      x,
      measuredCount: measured.length,
      availableCount: measured.length + (rows.length - measured.length),
      totalAvailable: experiments.filter((e) =>
        e.metrics.some((m) => m.name === metricName && m.value !== null),
      ).length,
    }
  }, [rows, experiments, metricName])

  if (!plot) return null

  const hidden = plot.totalAvailable - plot.measuredCount

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <div>
          <h3 className="text-sm font-medium text-ink-100">{titleCase(metricName)}</h3>
          <p className="text-[11px] text-ink-400">
            {direction === 'minimize' ? 'Lower is better' : 'Higher is better'} · dashed line is the
            mean of {plot.measuredCount} recorded run{plot.measuredCount === 1 ? '' : 's'}
            {hidden > 0 ? `, ${hidden} further run${hidden === 1 ? '' : 's'} not shown` : ''} ·
            colour is the model family
          </p>
        </div>
      </div>

      <div className="overflow-x-auto px-5 pb-4 pt-3">
        <svg
          viewBox={`0 0 ${VIEW_W} ${plot.height}`}
          style={{ minWidth: VIEW_W_MIN }}
          className="h-auto w-full"
          role="img"
          aria-label={`${titleCase(metricName)} for ${plot.measuredCount} runs`}
        >
          {plot.ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={plot.x(tick)}
                x2={plot.x(tick)}
                y1={8}
                y2={rows.length * ROW}
                stroke="var(--color-ink-850)"
                strokeWidth={1}
              />
              <text
                x={plot.x(tick)}
                y={rows.length * ROW + 18}
                textAnchor="middle"
                className="fill-ink-500 font-mono"
                fontSize={11}
              >
                {tick.toFixed(plot.decimals)}
              </text>
            </g>
          ))}

          <line
            x1={plot.x(plot.mean)}
            x2={plot.x(plot.mean)}
            y1={8}
            y2={rows.length * ROW}
            stroke="var(--color-ink-600)"
            strokeWidth={1}
            strokeDasharray="3 4"
          />

          {rows.map((row, index) => {
            const cy = index * ROW + ROW / 2 + 4
            const aboveMean = row.value !== null && row.value >= plot.mean
            return (
              <g key={row.id}>
                <text x={plot.plotLeft - 14} y={cy + 4} textAnchor="end" className="fill-ink-300" fontSize={12}>
                  {row.label}
                </text>

                {row.value === null ? (
                  <text x={plot.plotLeft + 4} y={cy + 4} className="fill-ink-500" fontSize={11}>
                    not recorded
                  </text>
                ) : (
                  <>
                    {/* The connector carries the family colour at low weight, so the
                        dot stays the thing being compared. */}
                    <line
                      x1={plot.plotLeft}
                      x2={plot.x(row.value)}
                      y1={cy}
                      y2={cy}
                      stroke={row.color}
                      strokeWidth={1}
                      opacity={0.22}
                    />
                    <circle
                      cx={plot.x(row.value)}
                      cy={cy}
                      r={aboveMean ? 4.5 : 3.5}
                      fill={row.color}
                      stroke="var(--color-ink-900)"
                      strokeWidth={1.5}
                    />
                    <text
                      x={plot.x(row.value) + 10}
                      y={cy + 4}
                      className={aboveMean ? 'fill-ink-100 font-mono' : 'fill-ink-400 font-mono'}
                      fontSize={11}
                    >
                      {row.display}
                    </text>
                  </>
                )}
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}