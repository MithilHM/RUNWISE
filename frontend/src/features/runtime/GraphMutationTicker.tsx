import { useMemo } from 'react'
import { DECISION_META } from './runtimeConstants'
import type { RuntimeState } from './runtimeModel'

/**
 * Graph mutation ticker.
 *
 * The claim this panel exists to support is persistence: a trial is not a row in a
 * table, it is a node and a set of edges that outlive the run. `graph_delta` arrives
 * as two anonymous counters, so the backend's documented mutation order is re-expanded
 * into the names it actually wrote (`Experiment`, `EVALUATED_ON`, `DERIVED_FROM`,
 * `FAILED_UNDER`, `Metric`, `PRODUCES_METRIC`) in `expandMutations`.
 *
 * Any count the server reported that the documented set cannot account for is shown
 * as an explicit residual row. Dropping it would let the panel imply that nothing
 * else was written, which is precisely the claim it must not make.
 */

interface GraphMutationTickerProps {
  state: RuntimeState
}

export function GraphMutationTicker({ state }: GraphMutationTickerProps) {
  const latest = state.mutations[0]
  const breakdown = useMemo(() => {
    const counts = new Map<string, number>()
    for (const record of state.mutations) {
      if (record.residual) continue
      counts.set(record.label, (counts.get(record.label) ?? 0) + 1)
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1])
  }, [state.mutations])

  const lastStep = state.mutations.length > 0 ? state.mutations[0].step : null

  return (
    <section className="panel flex min-h-[300px] flex-col" aria-label="FalkorDB graph mutation ticker">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Graph mutations</h2>
          <p className="mt-0.5 text-[11px] text-ink-500">
            Writes committed to FalkorDB by this run
            {state.phase === 'running' && <span className="ml-1.5 text-warn-400">live</span>}
          </p>
        </div>
        <span className="font-mono text-[10px] text-ink-500" title="Server-reported run id">
          {state.runId || 'no run yet'}
        </span>
      </div>

      <div className="grid grid-cols-3 border-b border-ink-800">
        <Counter label="nodes added" value={state.totals.nodes_added} accent="text-signal-400" />
        <Counter label="edges added" value={state.totals.edges_added} accent="text-cat-3" />
        <Counter
          label="this step"
          value={latest ? state.mutations.filter((record) => record.step === lastStep).length : 0}
          accent="text-warn-400"
        />
      </div>

      {breakdown.length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 border-b border-ink-800 px-5 py-2 text-[10px] text-ink-500">
          {breakdown.map(([label, count]) => (
            <span key={label} className="font-mono">
              {label}
              <span className="text-ink-200"> ×{count}</span>
            </span>
          ))}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto">
        {state.mutations.length === 0 ? (
          <p className="px-5 py-8 text-center text-xs text-ink-500">
            No writes yet. The graph is mutated at the end of each step, once the worker has returned a verdict.
          </p>
        ) : (
          <ul>
            {state.mutations.map((record, index) => (
              <li
                key={record.id}
                className={`flex items-baseline gap-2 px-5 py-1.5 font-mono text-[11px] ${
                  index === 0 ? 'border-l-2 border-signal-500 bg-ink-850' : 'border-l-2 border-transparent'
                }`}
              >
                <span className="w-8 shrink-0 text-ink-600" data-numeric>
                  s{record.step}
                </span>
                <span className={record.residual ? 'text-warn-400' : 'text-good-400'}>+1</span>
                <span className="shrink-0 text-ink-200">{record.label}</span>
                <span className="text-ink-600">{record.kind}</span>
                <span className="min-w-0 flex-1 truncate text-ink-500" title={record.detail}>
                  {record.detail}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}

function Counter({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <div className="border-r border-ink-800 px-4 py-2.5 last:border-r-0">
      <p className="label">{label}</p>
      <p className={`font-mono text-lg leading-tight ${accent}`} data-numeric>
        {value}
      </p>
    </div>
  )
}

/**
 * A compact statement of what the run did with the regions it knew about. It belongs
 * beside the tree rather than inside it: the tree shows which trials ran, this says
 * how the loop spent its budget avoiding the ones the graph had already ruled out.
 */
export function FailureRegionSummary({ state }: { state: RuntimeState }) {
  const stats = useMemo(() => {
    let avoided = 0
    let exploited = 0
    let explored = 0
    let failed = 0
    for (const node of state.nodes) {
      if (node.status === 'FAILED' || node.status === 'PRUNED') failed += 1
      if (node.decisionType === 'avoid_failure_subgraph') avoided += 1
      else if (node.decisionType === 'exploit_promising_regime') exploited += 1
      else if (node.decisionType === 'explore_boundary') explored += 1
    }
    return { avoided, exploited, explored, failed }
  }, [state.nodes])

  if (state.nodes.length === 0) return null

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 px-5 py-2 text-[11px]">
      {(
        [
          ['avoid_failure_subgraph', stats.avoided],
          ['exploit_promising_regime', stats.exploited],
          ['explore_boundary', stats.explored],
        ] as const
      ).map(([decision, count]) => (
        <span key={decision} className="text-ink-500" title={DECISION_META[decision].blurb}>
          {DECISION_META[decision].posture}{' '}
          <span className="font-mono text-ink-100" data-numeric>
            {count}
          </span>
        </span>
      ))}
      <span className="text-ink-500" title="Trials the worker failed or pruned; each is recorded as a FAILED_UNDER edge.">
        failed{' '}
        <span className="font-mono text-danger-400" data-numeric>
          {stats.failed}
        </span>
      </span>
    </div>
  )
}