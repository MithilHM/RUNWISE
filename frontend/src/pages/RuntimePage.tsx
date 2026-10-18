import { useMemo, useState } from 'react'
import { formatNumber } from '../lib/format'
import type { RuntimeObjective } from '../api/runtimeApi'
import { AblationScoreboard } from '../features/runtime/AblationScoreboard'
import { FailureRegionSummary, GraphMutationTicker } from '../features/runtime/GraphMutationTicker'
import { LayaReasoningFeed } from '../features/runtime/LayaReasoningFeed'
import { LiveExecutionTreeView } from '../features/runtime/LiveExecutionTreeView'
import { ObjectiveControlBar } from '../features/runtime/ObjectiveControlBar'
import { isMinimized } from '../features/runtime/runtimeConstants'
import { failureCount, failedEdgeCount, meanTrialSeconds, settledCount } from '../features/runtime/runtimeModel'
import { phaseLabel, useRuntimeSession } from '../features/runtime/useRuntimeSession'

/**
 * Autonomous runtime.
 *
 * A run is three simultaneous views of the same event stream: the tree says what ran,
 * the reasoning feed says why, and the ticker says what the run wrote down. They are
 * driven by one reducer (`useRuntimeSession`) so they cannot drift apart mid-run, and
 * the ablation below them answers whether any of it was worth the budget.
 *
 * Layout is a two-column instrument panel — the tree takes the wide column because it
 * is the only view that grows with the budget, and the two narrative panels stack in
 * the narrow one where they can be read without horizontal scrolling.
 */
export function RuntimePage() {
  const session = useRuntimeSession()
  const { state } = session
  const [selectedStep, setSelectedStep] = useState<number | null>(null)

  const settled = settledCount(state.nodes)
  const failures = failureCount(state.nodes)
  const edges = failedEdgeCount(state.nodes)
  const avoided = useMemo(
    () => state.nodes.filter((node) => node.decisionType === 'avoid_failure_subgraph').length,
    [state.nodes],
  )
  const meanSeconds = useMemo(() => meanTrialSeconds(state.nodes), [state.nodes])

  const start = (objective: RuntimeObjective, simulate: boolean) => {
    setSelectedStep(null)
    void session.start(objective, simulate)
  }

  const statusTone =
    state.phase === 'completed'
      ? 'border-good-400/40 bg-good-400/10 text-good-400'
      : state.phase === 'running' || state.phase === 'starting'
        ? 'border-warn-400/40 bg-warn-400/10 text-warn-400'
        : state.phase === 'error'
          ? 'border-danger-400/40 bg-danger-400/10 text-danger-400'
          : 'border-ink-700 bg-ink-850 text-ink-400'

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-lg font-medium text-ink-100">Autonomous runtime</h1>
          <p className="text-xs text-ink-500">
            Closed-loop search: query the graph, propose a typed candidate, let Laya commit with a stated
            hypothesis, train, write the result back.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {state.runId && <span className="mono-id">{state.runId}</span>}
          <span className={`chip ${statusTone}`}>
            {state.phase === 'running' && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn-400" />}
            {phaseLabel(state.phase)}
          </span>
          <span className="chip border-ink-700 bg-ink-850 text-ink-300" title="Trials whose worker has returned a verdict">
            {settled}/{state.objective.trial_budget} trials
          </span>
          <span
            className="chip border-ink-700 bg-ink-850 text-ink-300"
            title={`Best ${state.objective.target_metric} recorded so far in this run`}
          >
            best {state.bestScore === null ? 'not recorded' : formatNumber(state.bestScore, 4)}
            {state.bestScore !== null && (isMinimized(state.objective.target_metric) ? ' ↓' : ' ↑')}
          </span>
          {state.phase !== 'idle' && (
            <button type="button" className="btn btn-ghost" onClick={session.reset}>
              Clear
            </button>
          )}
        </div>
      </header>

      {state.error && (
        <div className="panel border-warn-400/40 px-4 py-2.5 text-[11px] text-warn-400" role="status">
          {state.error}
        </div>
      )}

      {state.phase === 'stopping' && (
        <div className="panel px-4 py-2.5 text-[11px] text-ink-400" role="status">
          {/* Text only: the reattach action lives on the control bar, which is where the
              rest of the run controls live. Two identical buttons is one too many. */}
          Detached from the run. The backend loop keeps going, so the trials below are the ones seen up to this
          point and nothing further will arrive until the stream is reattached.
        </div>
      )}

      <ObjectiveControlBar
        onStart={start}
        onStop={session.stop}
        onResume={session.resume}
        phase={state.phase}
        simulated={state.simulated}
        settled={settled}
        failed={state.failed}
        lastRun={state.runId ? state.objective : null}
      />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
        <div className="space-y-4">
          <LiveExecutionTreeView state={state} selectedStep={selectedStep} onSelect={setSelectedStep} />
          {state.nodes.length > 0 && (
            <div className="panel">
              <div className="flex flex-wrap items-center gap-3 px-5 py-1.5">
                <p className="label">How the budget was spent</p>
                {failures > 0 && (
                  <span className="text-[11px] text-ink-500">
                    {failures} trial(s) failed or were pruned
                    {edges > 0 && `, of which ${edges} wrote a FAILED_UNDER edge`} for the next run to avoid
                  </span>
                )}
              </div>
              <FailureRegionSummary state={state} />
            </div>
          )}
        </div>

        <div className="space-y-4">
          <LayaReasoningFeed state={state} onSelectStep={setSelectedStep} />
          <GraphMutationTicker state={state} />
        </div>
      </div>

      <AblationScoreboard
        result={session.ablation}
        metric={state.objective.target_metric}
        budget={state.objective.trial_budget}
        pending={session.ablationPending}
        error={session.ablationError}
        simulated={session.ablationSimulated}
        meanTrialSeconds={meanSeconds}
        avoidedRegions={avoided}
        failedRegions={failures}
        failedEdgeCount={edges}
        onRun={(budget) => void session.runAblation(budget)}
      />
    </div>
  )
}