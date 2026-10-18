import { useState } from 'react'
import type { RuntimeObjective } from '../../api/runtimeApi'
import {
  DATASET_OPTIONS,
  DEFAULT_OBJECTIVE,
  MODEL_FAMILY_OPTIONS,
  TARGET_METRIC_OPTIONS,
  isMinimized,
} from './runtimeConstants'
import type { RuntimePhase } from './runtimeModel'

/**
 * Objective control bar.
 *
 * The researcher states a goal once and then watches the loop act on it, so the bar
 * locks while a run is live rather than letting a dataset be swapped underneath a
 * stream that is still reporting trials for the previous one.
 *
 * `val_log_loss` is offered as a target because it is the only properly calibrated
 * metric of the three, and its direction is stated next to it: a researcher who
 * selects it must not have to remember that lower is better. The other two say
 * "maximize" for the same reason.
 */

const BUDGET_MIN = 5
const BUDGET_MAX = 20

interface ObjectiveControlBarProps {
  /** `simulate` is true when the operator ticked demo mode. */
  onStart: (objective: RuntimeObjective, simulate: boolean) => void
  onStop: () => void
  phase: RuntimePhase
  /** True when the live stream is in fact a local replay, however it was chosen. */
  simulated: boolean
  /** Trials whose worker has actually returned a verdict. */
  settled: number
  /** The objective actually launched, not whatever the form currently holds. */
  lastRun: RuntimeObjective | null
}

export function ObjectiveControlBar({
  onStart,
  onStop,
  phase,
  simulated,
  settled,
  lastRun,
}: ObjectiveControlBarProps) {
  const [dataset, setDataset] = useState<string>(DEFAULT_OBJECTIVE.dataset_name)
  const [modelFamily, setModelFamily] = useState<string>(DEFAULT_OBJECTIVE.model_family)
  const [targetMetric, setTargetMetric] = useState<string>(DEFAULT_OBJECTIVE.target_metric)
  const [budget, setBudget] = useState<number>(DEFAULT_OBJECTIVE.trial_budget)
  const [demoMode, setDemoMode] = useState(false)

  const live = phase === 'starting' || phase === 'running' || phase === 'stopping'
  const selectedMetric = TARGET_METRIC_OPTIONS.find((option) => option.value === targetMetric)

  const launch = () => {
    onStart(
      {
        dataset_name: dataset,
        target_metric: targetMetric,
        model_family: modelFamily,
        trial_budget: budget,
      },
      demoMode,
    )
  }

  return (
    <section className="panel" aria-label="Autonomous runtime objective">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Objective</h2>
          <p className="mt-0.5 text-[11px] text-ink-500">
            One sentence of intent. Laya turns it into a typed search and a hypothesis per trial.
          </p>
        </div>
        {simulated && (
          <span
            className="chip border-warn-400/40 bg-warn-400/10 text-warn-400"
            title="Events are replayed locally because the backend runtime is not reachable."
          >
            simulated stream
          </span>
        )}
      </div>

      <div className="grid gap-4 px-5 py-4 lg:grid-cols-[1.4fr_1fr_1.4fr_1.1fr]">
        <label className="flex flex-col gap-1.5">
          <span className="label">Dataset</span>
          <select
            className="input"
            value={dataset}
            onChange={(event) => setDataset(event.target.value)}
            disabled={live}
          >
            {DATASET_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="label">Model family</span>
          <select
            className="input"
            value={modelFamily}
            onChange={(event) => setModelFamily(event.target.value)}
            disabled={live}
          >
            {MODEL_FAMILY_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="label">Target metric</span>
          <select
            className="input"
            value={targetMetric}
            onChange={(event) => setTargetMetric(event.target.value)}
            disabled={live}
          >
            {TARGET_METRIC_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label} — {isMinimized(option.value) ? 'minimize' : 'maximize'}
              </option>
            ))}
          </select>
          <span className="text-[11px] leading-snug text-ink-500">{selectedMetric?.note}</span>
        </label>

        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between">
            <label className="label" htmlFor="trial-budget">
              Trial budget
            </label>
            <span className="font-mono text-sm text-ink-100" data-numeric>
              {budget}
            </span>
          </div>
          <input
            id="trial-budget"
            type="range"
            min={BUDGET_MIN}
            max={BUDGET_MAX}
            step={1}
            value={budget}
            onChange={(event) => setBudget(Number(event.target.value))}
            disabled={live}
            className="mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-700 accent-signal-400 disabled:cursor-not-allowed disabled:opacity-40"
          />
          <div className="flex justify-between font-mono text-[10px] text-ink-500">
            <span>{BUDGET_MIN}</span>
            <span>{BUDGET_MAX}</span>
          </div>
          <span className="text-[11px] leading-snug text-ink-500">
            {live
              ? `${settled} of ${budget} trial(s) settled.`
              : 'One trial is one sandboxed training run.'}
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 border-t border-ink-750 px-5 py-3">
        {live ? (
          <button type="button" className="btn btn-danger" onClick={onStop}>
            Stop listening
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={launch}>
            Start autonomous runtime
          </button>
        )}

        <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-400">
          <input
            type="checkbox"
            checked={demoMode}
            onChange={(event) => setDemoMode(event.target.checked)}
            disabled={live}
            className="h-3.5 w-3.5 accent-signal-400 disabled:opacity-40"
          />
          Demo mode
          <span className="text-ink-500">
            {demoMode
              ? 'replays a scripted run locally'
              : 'auto-falls back to a local replay if the runtime is unreachable'}
          </span>
        </label>

        {/* Reports what actually ran. Built from live form state it would restate the
            claim every time a select was touched after the run finished. */}
        {lastRun && phase !== 'idle' && (
          <span className="ml-auto font-mono text-[11px] text-ink-500">
            last run: {lastRun.trial_budget} trial budget · {lastRun.model_family} ·{' '}
            {TARGET_METRIC_OPTIONS.find((option) => option.value === lastRun.target_metric)?.label ??
              lastRun.target_metric}{' '}
            · {DATASET_OPTIONS.find((option) => option.value === lastRun.dataset_name)?.label ?? lastRun.dataset_name}
          </span>
        )}
      </div>
    </section>
  )
}