/** Formatting helpers. Every "unknown" renders as a word, never as a zero. */

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(digits)
}

export function formatMetric(metric: { value: number | null; direction: string }): string {
  if (metric.value === null || metric.value === undefined) return 'not recorded'
  return `${formatNumber(metric.value)}${metric.direction === 'minimize' ? ' ↓' : ' ↑'}`
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '—'
  if (seconds < 60) return `${formatNumber(seconds, 0)}s`
  if (seconds < 3600) return `${formatNumber(seconds / 60, 1)}m`
  return `${formatNumber(seconds / 3600, 2)}h`
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatJson(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value !== 'object') return String(value)
  return JSON.stringify(value, null, 2)
}

export function titleCase(value: string): string {
  return value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

/** Unrecorded is not zero. Used wherever a metric, cost or step count is absent. */
export function orUnrecorded(value: number | null | undefined, suffix = ''): string {
  return value === null || value === undefined ? `not recorded${suffix}` : `${formatNumber(value)}${suffix}`
}
