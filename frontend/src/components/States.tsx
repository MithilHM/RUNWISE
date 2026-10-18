import type { ReactNode } from 'react'
import { ApiError } from '../api/client'

/** Loading, error and empty states. Every screen uses these, never an ad-hoc spinner. */

export function LoadingState({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 px-4 py-8 text-sm text-ink-400" role="status">
      <span className="h-3 w-3 animate-pulse rounded-full bg-signal-500" />
      {label}…
    </div>
  )
}

export function ErrorState({
  error,
  onRetry,
  context,
}: {
  error: unknown
  onRetry?: () => void
  context?: string
}) {
  const apiError = error instanceof ApiError ? error : null
  const message =
    apiError?.details[0]?.message ??
    (error instanceof Error ? error.message : 'Something went wrong.')
  const code = apiError?.details[0]?.code ?? 'unknown_error'

  return (
    <div className="panel border-danger-400/40 p-4" role="alert">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="label text-danger-400">{context ?? 'Request failed'}</p>
          <p className="mt-1 text-sm text-ink-100">{message}</p>
          <p className="mt-1 font-mono text-[11px] text-ink-500">
            {code}
            {apiError?.status ? ` · HTTP ${apiError.status}` : ''}
          </p>
        </div>
        {onRetry && (
          <button type="button" className="btn" onClick={onRetry}>
            Retry
          </button>
        )}
      </div>
      {apiError && apiError.details.length > 1 && (
        <ul className="mt-3 space-y-1 border-t border-ink-700 pt-3 text-xs text-ink-400">
          {apiError.details.slice(1).map((detail, index) => (
            <li key={`${detail.code}-${index}`}>
              <span className="font-mono text-ink-500">{detail.code}</span> {detail.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function EmptyState({
  title,
  hint,
  action,
}: {
  title: string
  hint?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
      <p className="text-sm font-medium text-ink-300">{title}</p>
      {hint && <div className="max-w-md text-xs leading-relaxed text-ink-400">{hint}</div>}
      {action}
    </div>
  )
}

/**
 * Persistent reminder that answers are produced without a language model when no
 * key is configured. PRD 15 forbids implying a model was consulted.
 */
export function DemoModeBadge({ llmConfigured }: { llmConfigured: boolean | null }) {
  if (llmConfigured === null) return null
  if (llmConfigured) {
    return (
      <span className="rounded border border-ink-600 px-2 py-0.5 font-mono text-[10px] text-ink-400">
        llm: configured
      </span>
    )
  }
  return (
    <span
      className="rounded border border-warn-400/40 bg-warn-400/10 px-2 py-0.5 font-mono text-[10px] text-warn-400"
      title="No RUNWISE_LLM_API_KEY is set. Answers are generated deterministically and report llm_used=false."
    >
      llm: off · deterministic
    </span>
  )
}
