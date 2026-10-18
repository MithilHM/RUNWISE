import { useMemo, useState } from 'react'
import { formatNumber } from '../../lib/format'
import { DECISION_META, STAGE_LABELS, isMinimized } from './runtimeConstants'
import type { LayaReasoningEntry, RuntimeState } from './runtimeModel'

/**
 * Laya's reasoning feed.
 *
 * A trial emits six frames, so an unfiltered feed is mostly stage bookkeeping and
 * buries the one thing a researcher wants: why this candidate, now. The default view
 * therefore shows the decision frames, with an "all stages" toggle for anyone
 * following the loop mechanics.
 *
 * Cited graph nodes are clickable because they are the claim's provenance. A citation
 * that cannot be opened is just an assertion, so expanding one shows the identifier in
 * full and offers to copy it for a direct lookup in FalkorDB.
 */

type FeedFilter = 'decisions' | 'all'

interface LayaReasoningFeedProps {
  state: RuntimeState
  onSelectStep?: (step: number) => void
}

export function LayaReasoningFeed({ state, onSelectStep }: LayaReasoningFeedProps) {
  const [filter, setFilter] = useState<FeedFilter>('decisions')
  const [inspected, setInspected] = useState<Record<string, boolean>>({})
  const [copied, setCopied] = useState<Record<string, boolean>>({})

  const entries = useMemo(
    () => (filter === 'decisions' ? state.feed.filter((entry) => entry.decision !== null) : state.feed),
    [filter, state.feed],
  )

  const decisionCount = useMemo(() => state.feed.filter((entry) => entry.decision !== null).length, [state.feed])

  const copy = async (id: string) => {
    try {
      await navigator.clipboard.writeText(id)
      setCopied((previous) => ({ ...previous, [id]: true }))
      setTimeout(() => setCopied((previous) => ({ ...previous, [id]: false })), 1600)
    } catch {
      // Clipboard access is blocked in some contexts. The id is already on screen and
      // selectable, so a failed copy needs no error state.
    }
  }

  return (
    <section className="panel flex min-h-[420px] flex-col" aria-label="Laya reasoning feed">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Laya reasoning</h2>
          <p className="mt-0.5 text-[11px] text-ink-500">
            {decisionCount} decision(s) taken · newest first
          </p>
        </div>
        <div className="flex items-center gap-1" role="group" aria-label="Feed filter">
          {(['decisions', 'all'] as FeedFilter[]).map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setFilter(option)}
              aria-pressed={filter === option}
              className={`rounded px-2 py-1 text-[11px] transition-colors ${
                filter === option ? 'bg-ink-800 text-ink-100' : 'text-ink-500 hover:bg-ink-850 hover:text-ink-300'
              }`}
            >
              {option === 'decisions' ? 'Decisions' : 'All stages'}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {entries.length === 0 ? (
          <p className="px-5 py-8 text-center text-xs text-ink-500">
            No decision recorded yet. Laya commits once the graph query and candidate pool are ready.
          </p>
        ) : (
          entries.map((entry) => (
            <FeedCard
              key={entry.id}
              entry={entry}
              metric={state.objective.target_metric}
              /* Derived, not stored. An `active` flag written at insertion time can
                 never be cleared, so every past card would pulse forever. */
              live={state.phase === 'running' && entry.step === state.step && entry.stage === state.stage}
              inspected={inspected}
              copied={copied}
              onToggle={(id) => setInspected((previous) => ({ ...previous, [id]: !previous[id] }))}
              onCopy={copy}
              onSelectStep={onSelectStep}
            />
          ))
        )}
      </div>
    </section>
  )
}

/**
 * Distinct cited identifiers, order preserved.
 *
 * The evaluator legitimately cites the same subgraph node more than once in one
 * decision. Rendering it twice is noise, and keying the list by the raw value
 * produced duplicate React keys, which React reports as a real defect.
 */
function dedupe(paths: string[]): string[] {
  return [...new Set(paths)]
}

function FeedCard({
  entry,
  metric,
  live,
  inspected,
  copied,
  onToggle,
  onCopy,
  onSelectStep,
}: {
  entry: LayaReasoningEntry
  metric: string
  /** True only for the single frame the run is sitting on right now. */
  live: boolean
  inspected: Record<string, boolean>
  copied: Record<string, boolean>
  onToggle: (id: string) => void
  onCopy: (id: string) => void
  onSelectStep?: (step: number) => void
}) {
  const decision = entry.decision
  const meta = decision ? DECISION_META[decision.decision_type] : null
  const postureStyle = decision
    ? decision.decision_type === 'avoid_failure_subgraph'
      ? 'border-danger-400/40 bg-danger-400/10 text-danger-400'
      : decision.decision_type === 'explore_boundary'
        ? 'border-cat-3/40 bg-cat-3/10 text-cat-3'
        : 'border-good-400/40 bg-good-400/10 text-good-400'
    : 'border-ink-700 bg-ink-850 text-ink-400'

  return (
    <article className="border-b border-ink-800 px-5 py-3.5 last:border-b-0">
      <header className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onSelectStep?.(entry.step)}
          disabled={!onSelectStep}
          className="font-mono text-[11px] text-signal-400 disabled:cursor-default"
          title={onSelectStep ? `Focus trial ${entry.step} in the execution tree` : undefined}
        >
          step {entry.step}
        </button>
        <span className="text-[11px] text-ink-500">{STAGE_LABELS[entry.stage]}</span>
        {live && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn-400" title="In flight now" />}
        <span className="ml-auto font-mono text-[10px] text-ink-600">
          {new Date(entry.at).toLocaleTimeString(undefined, { hour12: false })}
        </span>
      </header>

      {decision && meta && (
        <>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <span className={`chip ${postureStyle}`} title={meta.blurb}>
              {meta.posture}
            </span>
            <span className="text-[11px] text-ink-400">{meta.label}</span>
            <span className="ml-auto flex items-center gap-1.5" title={`Laya's stated confidence in this pick`}>
              <span className="h-1 w-16 overflow-hidden rounded-full bg-ink-700">
                <span
                  className="block h-full bg-signal-500"
                  style={{ width: `${Math.round(decision.confidence * 100)}%` }}
                />
              </span>
              <span className="font-mono text-[10px] text-ink-400" data-numeric>
                {Math.round(decision.confidence * 100)}%
              </span>
            </span>
          </div>

          <div className="mt-2 border-l-2 border-ink-700 pl-3">
            <p className="label">Hypothesis</p>
            <p className="mt-0.5 text-xs leading-relaxed text-ink-200">{decision.hypothesis_text}</p>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px]">
            <span className="text-ink-500">
              candidate{' '}
              <span className="mono-id">{decision.selected_candidate_id}</span>
            </span>
            {entry.trial && (
              <span className="text-ink-500">
                outcome{' '}
                <span className="font-mono text-ink-200">
                  {entry.trial.status.toLowerCase()}
                  {entry.trial.status === 'SUCCESS' &&
                    ` · ${formatNumber(entry.trial.metrics[metric] ?? null, 4)}${isMinimized(metric) ? ' ↓' : ' ↑'}`}
                </span>
              </span>
            )}
            {entry.bestScore !== null && (
              <span className="text-ink-500">
                best so far{' '}
                <span className="font-mono text-ink-200" data-numeric>
                  {formatNumber(entry.bestScore, 4)}
                </span>
              </span>
            )}
          </div>

          <div className="mt-2">
            <p className="label">FalkorDB evidence nodes</p>
            {decision.cited_graph_paths.length === 0 ? (
              <p className="mt-0.5 text-[11px] text-ink-500">
                No graph evidence cited. This decision rests on the pool generator alone.
              </p>
            ) : (
              <div className="mt-1 flex flex-wrap gap-1.5">
                {dedupe(decision.cited_graph_paths).map((path) => {
                  const open = inspected[path]
                  return (
                    <span key={path} className="inline-flex flex-col">
                      <button
                        type="button"
                        onClick={() => onToggle(path)}
                        aria-expanded={open}
                        aria-controls={`evidence-detail-${path}`}
                        className={`mono-id rounded border px-1.5 py-0.5 text-left transition-colors ${
                          open
                            ? 'border-signal-500 bg-signal-500/10 text-ink-100'
                            : 'border-ink-700 bg-ink-850 hover:border-ink-500'
                        }`}
                        title={open ? 'Hide the full identifier' : 'Show the full identifier and a copy action'}
                      >
                        {path.length > 26 ? `${path.slice(0, 25)}…` : path}
                      </button>
                      {open && (
                        <span
                          id={`evidence-detail-${path}`}
                          className="mt-1 inline-flex items-center gap-2 rounded border border-ink-750 bg-ink-900 px-1.5 py-1"
                        >
                          <span className="font-mono text-[10px] break-all text-ink-200">{path}</span>
                          <button
                            type="button"
                            onClick={() => void onCopy(path)}
                            className="shrink-0 text-[10px] text-signal-400 hover:text-signal-300"
                          >
                            {copied[path] ? 'copied' : 'copy'}
                          </button>
                        </span>
                      )}
                    </span>
                  )
                })}
              </div>
            )}
          </div>
        </>
      )}

      {!decision && entry.message && <p className="mt-1.5 text-[11px] leading-relaxed text-ink-400">{entry.message}</p>}
    </article>
  )
}