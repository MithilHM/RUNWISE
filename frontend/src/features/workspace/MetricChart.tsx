import { useMemo } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { MetricValue } from '../../api/types'
import { formatMetric, titleCase } from '../../lib/format'

interface MetricChartProps {
  experiments: { id: string; name: string; metrics: MetricValue[] }[]
  metricName: string
  limit?: number
}

/**
 * Metric comparison across runs.
 *
 * A run with no recorded value for the selected metric is drawn as a gap, not as a
 * zero-height bar. A blank column means unrecorded; rendering it as 0 would make an
 * unmeasured run look like the worst possible result.
 */
export function MetricChart({ experiments, metricName, limit = 12 }: MetricChartProps) {
  const direction = useMemo(() => {
    const recorded = experiments.find((e) => e.metrics.some((m) => m.name === metricName))
    return recorded?.metrics.find((m) => m.name === metricName)?.direction ?? 'maximize'
  }, [experiments, metricName])

  const data = useMemo(
    () =>
      experiments
        .slice(0, limit)
        .map((experiment) => {
          const metric = experiment.metrics.find((m) => m.name === metricName)
          return {
            id: experiment.id,
            name: experiment.name.length > 18 ? `${experiment.name.slice(0, 17)}…` : experiment.name,
            fullName: experiment.name,
            value: metric?.value ?? null,
            display: formatMetric({ value: metric?.value ?? null, direction: metric?.direction ?? direction }),
          }
        }),
    [experiments, metricName, limit, direction],
  )

  if (!data.length) return null

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h3 className="text-sm font-medium text-ink-100">{titleCase(metricName)}</h3>
          <p className="text-[11px] text-ink-500">
            {direction === 'minimize' ? 'Lower is better' : 'Higher is better'} · gaps are unrecorded values
          </p>
        </div>
      </div>
      <div className="h-64 px-2 py-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke="#262e3b" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: '#7c8798', fontSize: 10 }}
              interval={0}
              angle={-25}
              textAnchor="end"
              height={56}
            />
            <YAxis tick={{ fill: '#7c8798', fontSize: 10 }} width={48} domain={['auto', 'auto']} />
            <Tooltip
              cursor={{ fill: '#161b24' }}
              contentStyle={{
                background: '#11151c',
                border: '1px solid #3a4453',
                borderRadius: 6,
                fontSize: 12,
                color: '#e4e9f0',
              }}
              formatter={(_value, _name, item) => [(item as { payload: { display: string } }).payload.display, metricName]}
              labelFormatter={(_label, payload) =>
                (payload?.[0] as { payload: { fullName: string; id: string } } | undefined)?.payload.fullName ?? ''
              }
            />
            <Legend wrapperStyle={{ fontSize: 11, color: '#a7b1c0' }} />
            <Bar
              dataKey="value"
              name={titleCase(metricName)}
              fill="#2cb5a8"
              radius={[3, 3, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
