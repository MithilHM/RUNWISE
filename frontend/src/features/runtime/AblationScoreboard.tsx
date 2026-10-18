import { useMemo, useState } from 'react'
import { formatNumber } from '../../lib/format'
import { EmptyState, ErrorState, LoadingState } from '../../components/States'
import type { AblationCurvePoint, AblationResult } from '../../api/runtimeApi'
import { ABLATION_MAX_BUDGET, ABLATION_MIN_BUDGET, isMinimized } from './runtimeConstants'
import { trialsToReach } from './runtimeModel'

/**
 * Ablation scoreboard.
 *
 * Answers one question with evidence: what did graph-guided search buy, measured
 * against spending the same budget without it. Two rules shape the whole panel.
 *
 * First, only the arms the backend measured are drawn. The endpoint compares Graph+Laya
 * against uniform random search and nothing else, so the unconstrained column states
 * that it was not run rather than carrying a plausible invented curve — an unsourced
 * third line would read as evidence and would be indistinguishable from the two real
 * ones in the legend.
 *
 * Second, the summary figures name their own provenance in a tooltip. "Trials to
 * target" is measured against the leading arm's best value, not against some
 * externally chosen bar, and the reader can hover to find out which.
 *
 * The chart is hand-drawn SVG rather than a charting library, matching `MetricChart`:
 * the whole thing is two polylines on a shared scale, and a library would add weight
 * without adding a capability.
 */

interface AblationScoreboardProps {
  result: AblationResult | null
  metric: string
  budget: number
  pending: boolean
  error: unknown
  simulated: boolean
  /** Mean wall-clock seconds per trial from this run, or null when unrecorded. */
  meanTrialSeconds: number | null
  /** Trials the loop steered away from a recorded failure region. */
  avoidedRegions: number
  /** Trials that failed and became FAILED_UNDER edges. */
  failedRegions: number
  onRun: (budget: number) => void
}

export function AblationScoreboard({
  result,
  metric,
  budget,
  pending,
  error,
  simulated,
  meanTrialSeconds,
  avoidedRegions,
  failedRegions,
  onRun,
}: AblationScoreboardProps) {
  const [budgetOverride, setBudgetOverride] = useState<number | null>(null)
  // Derived, not synchronised in an effect: a new run's budget becomes the default
  // the moment it starts, and an explicit override still wins until it is cleared.
  const selectedBudget = budgetOverride ?? budget

  return (
    <section className="panel" aria-label="Ablation scoreboard">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Ablation · graph-guided vs uninformed search</h2>
          <p className="mt-0.5 text-[11px] text-ink-500">
            Same objective, same search space, same trial budget. The only difference is whether the graph was
            consulted before each pick.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <label className="flex items-center gap-2 text-[11px] text-ink-500">
            budget
            <input
              type="number"
              min={ABLATION_MIN_BUDGET}
              max={ABLATION_MAX_BUDGET}
              value={selectedBudget}
              onChange={(event) => setBudgetOverride(Number(event.target.value))}
              className="input w-20 py-1 text-right font-mono text-xs"
            />
          </label>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => onRun(selectedBudget)}
            disabled={pending}
          >
            {pending ? 'Running…' : 'Run ablation'}
          </button>
        </div>
      </div>

      {error ? (
        <div className="p-4">
          <ErrorState error={error} context="The ablation could not be computed" onRetry={() => onRun(selectedBudget)} />
        </div>
      ) : pending ? (
        <LoadingState label={`Running ${selectedBudget} trials per arm`} />
      ) : !result ? (
        <EmptyState
          title="No ablation run for this objective yet"
          hint="The comparison runs the leading arm and a uniform random arm over an identical budget. It needs a run to draw its curves from."
        />
      ) : (
        <div className="space-y-4 px-5 py-4">
          {simulated && (
            <p className="rounded border border-warn-400/40 bg-warn-400/10 px-3 py-1.5 text-[11px] text-warn-400">
              Both arms were replayed locally against this run's recorded scores. No optimizer was executed.
            </p>
          )}

          <ConvergenceChart result={result} metric={metric} />

          <div className="grid gap-3 md:grid-cols-3">
            <ArmCard
              title="RUNWISE"
              subtitle="Graph + Laya"
              best={result.laya_best}
              curve={result.laya_curve}
              target={result.laya_best}
              minimize={isMinimized(metric)}
              accent="signal"
              note="Multi-hop regime queries and typed failure edges steer every pick."
            />
            <ArmCard
              title="Random search"
              subtitle="uniform sampling"
              best={result.random_best}
              curve={result.random_curve}
              target={result.laya_best}
              minimize={isMinimized(metric)}
              accent="cat2"
              note="Same lattice, same budget, no memory of what already failed."
            />
            <ArmCard
              title="Unconstrained optimizer"
              subtitle="no graph guidance"
              best={result.unconstrained_best ?? null}
              curve={result.unconstrained_curve}
              target={result.laya_best}
              minimize={isMinimized(metric)}
              accent="ink"
              note="The endpoint does not run a third arm."
              unmeasured={
                result.unconstrained_curve
                  ? undefined
                  : 'Not measured. Only Graph+Laya and uniform random search are executed by POST /api/runtime/{run_id}/ablation. This column stays empty until a server supplies a measured curve.'
              }
            />
          </div>

          <SummaryCards
            result={result}
            metric={metric}
            meanTrialSeconds={meanTrialSeconds}
            avoidedRegions={avoidedRegions}
            failedRegions={failedRegions}
          />

          <p className="border-t border-ink-800 pt-3 text-[11px] leading-relaxed text-ink-400">{result.summary}</p>
        </div>
      )}
    </section>
  )
}

type Accent = 'signal' | 'cat2' | 'ink'

const ACCENT_STROKE: Record<Accent, string> = {
  signal: 'var(--color-signal-400)',
  cat2: 'var(--color-cat-2)',
  ink: 'var(--color-ink-500)',
}

const ACCENT_TEXT: Record<Accent, string> = {
  signal: 'text-signal-400',
  cat2: 'text-cat-2',
  ink: 'text-ink-400',
}

function ArmCard({
  title,
  subtitle,
  best,
  curve,
  target,
  minimize,
  accent,
  note,
  unmeasured,
}: {
  title: string
  subtitle: string
  best: number | null
  curve: AblationCurvePoint[] | undefined
  target: number
  /** Whether a lower curve is the better one. Passing it explicitly keeps the
   *  "reached the target" judgement from inverting on a minimised metric. */
  minimize: boolean
  accent: Accent
  note: string
  unmeasured?: string
}) {
  const reached = trialsToReach(curve, target, minimize)
  return (
    <div className="panel-raised px-4 py-3">
      <span className="mb-2 block h-0.5 w-8 rounded" style={{ background: ACCENT_STROKE[accent] }} />
      <div className="flex items-baseline justify-between gap-2">
        <div>
          <p className="text-xs font-medium text-ink-100">{title}</p>
          <p className="text-[11px] text-ink-500">{subtitle}</p>
        </div>
        <span className="font-mono text-lg text-ink-100" data-numeric>
          {best === null || best === undefined ? '—' : formatNumber(best, 4)}
        </span>
      </div>

      {unmeasured ? (
        <p className="mt-2 border-t border-ink-750 pt-2 text-[11px] leading-relaxed text-ink-500">{unmeasured}</p>
      ) : (
        <>
          <p className="mt-2 text-[11px] text-ink-500">
            reached the leading arm's best value at trial{' '}
            <span className={`font-mono ${ACCENT_TEXT[accent]}`} data-numeric>
              {reached === null ? 'never' : reached}
            </span>
            <span className="text-ink-600"> of {curve?.length ?? 0}</span>
          </p>
          <p className="mt-1 border-t border-ink-750 pt-2 text-[11px] leading-relaxed text-ink-500">{note}</p>
        </>
      )}
    </div>
  )
}

function SummaryCards({
  result,
  metric,
  meanTrialSeconds,
  avoidedRegions,
  failedRegions,
}: {
  result: AblationResult
  metric: string
  meanTrialSeconds: number | null
  avoidedRegions: number
  failedRegions: number
}) {
  const minimize = isMinimized(metric)
  const runwiseReached = trialsToReach(result.laya_curve, result.laya_best, minimize)
  const randomReached = trialsToReach(result.random_curve, result.laya_best, minimize)
  const trialsSaved = randomReached !== null && runwiseReached !== null ? Math.max(0, randomReached - runwiseReached) : null
  const computeSeconds = trialsSaved !== null && meanTrialSeconds !== null ? trialsSaved * meanTrialSeconds : null
  const improvement = result.improvement_percentage

  /* An ablation that only ever flatters the product is not an ablation. When random
     search reaches the target first — which happens on a short budget — the figures
     say so, and the card is not tinted as a win. */
  const runwiseWon = randomReached === null || (runwiseReached !== null && runwiseReached < randomReached)

  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <SummaryCard
        label="Trials to target metric"
        value={randomReached === null ? 'never reached' : `${randomReached} for random`}
        sub={`RUNWISE reached ${result.laya_best.toFixed(4)} at trial ${runwiseReached ?? '—'}${
          minimize ? ' (lower is better)' : ''
        }`}
        tone={runwiseWon ? 'good' : 'neutral'}
        hint={`Target is the best ${metric} the Graph+Laya arm reached. The random arm is scored on how many trials it needed to get there, which is the number a search budget is actually spent against. "Never reached" means uniform sampling did not clear that value within the budget.`}
      />
      <SummaryCard
        label="Compute time saved"
        value={
          trialsSaved === null
            ? meanTrialSeconds === null
              ? 'not recorded'
              : 'not measured'
            : computeSeconds !== null
              ? computeSeconds > 0
                ? formatDurationShort(computeSeconds)
                : 'no trials saved'
              : 'not recorded'
        }
        sub={
          trialsSaved === null
            ? 'the random arm never reached the target, so there is no saving to price'
            : meanTrialSeconds === null
              ? 'per-trial duration was not recorded by the worker'
              : `at ${meanTrialSeconds.toFixed(2)}s mean per trial, ${trialsSaved} trial(s) ahead`
        }
        tone={computeSeconds !== null && computeSeconds > 0 ? 'good' : 'neutral'}
        hint="Trials the random arm needed beyond the point RUNWISE hit its target, priced at this run's own mean trial duration from the recorded train_time_sec values. Zero means the random arm matched the target no later than RUNWISE did."
      />
      <SummaryCard
        label="Failure regions avoided"
        value={`${avoidedRegions} avoided · ${failedRegions} hit`}
        sub={`${improvement >= 0 ? '+' : ''}${improvement.toFixed(2)}% relative metric gain over random search`}
        tone={avoidedRegions > 0 ? 'good' : 'neutral'}
        hint="'Avoided' counts trials where Laya's decision type was avoid_failure_subgraph, meaning it cited a recorded FAILED_UNDER region and stayed out of it. 'Hit' counts trials the worker actually failed or pruned, each of which wrote its own FAILED_UNDER edge for the next run to avoid."
      />
    </div>
  )
}

function SummaryCard({
  label,
  value,
  sub,
  tone,
  hint,
}: {
  label: string
  value: string
  sub: string
  tone: 'good' | 'warn' | 'neutral'
  hint: string
}) {
  const toneClass = tone === 'good' ? 'text-good-400' : tone === 'warn' ? 'text-warn-400' : 'text-ink-100'
  return (
    <div className="panel-raised px-4 py-3" title={hint}>
      <p className="label">{label}</p>
      <p className={`mt-1 font-mono text-base leading-tight ${toneClass}`} data-numeric>
        {value}
      </p>
      <p className="mt-1 text-[11px] leading-snug text-ink-500">{sub}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Convergence chart
// ---------------------------------------------------------------------------

const CHART_H = 250
const PAD = { top: 16, right: 18, bottom: 30, left: 56 }
const PLOT_W = 900

/** Round tick values. A 1/2/2.5/5/10 ladder keeps labels at readable magnitudes. */
function axisTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [min]
  const rawStep = (max - min) / count
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const normalized = rawStep / magnitude
  const step = (normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10) * magnitude
  const first = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let value = first; value <= max + step * 0.001; value += step) ticks.push(Number(value.toFixed(6)))
  return ticks
}

function ConvergenceChart({ result, metric }: { result: AblationResult; metric: string }) {
  /** Every arm the server actually measured, including the optional third. */
  const arms = useMemo(
    () =>
      [
        { curve: result.laya_curve, color: 'var(--color-signal-400)', label: 'RUNWISE · Graph + Laya', best: result.laya_best },
        { curve: result.random_curve, color: 'var(--color-cat-2)', label: 'Uniform random search', best: result.random_best },
        ...(result.unconstrained_curve
          ? [
              {
                curve: result.unconstrained_curve,
                color: 'var(--color-ink-400)',
                label: 'Unconstrained optimizer',
                best: result.unconstrained_best ?? null,
              },
            ]
          : []),
      ] satisfies Array<{ curve: AblationCurvePoint[]; color: string; label: string; best: number | null }>,
    [result],
  )

  /**
   * A shared padded domain for every arm. Fixing the scale across the curves is the
   * whole point of the chart: separate auto-scaled axes would make a random walk and
   * a converging search look identical.
   */
  const scale = useMemo(() => {
    const values = arms.flatMap((arm) => arm.curve.map((point) => point.best_score))
    const rawMin = values.length ? Math.min(...values) : 0
    const rawMax = values.length ? Math.max(...values) : 1
    const padding = (rawMax - rawMin) * 0.12 || Math.abs(rawMax) * 0.05 || 0.01
    return { min: rawMin - padding, max: rawMax + padding }
  }, [arms])

  const trials = Math.max(...arms.map((arm) => arm.curve.length), 1)
  const yTicks = axisTicks(scale.min, scale.max)
  const x = (trial: number) => PAD.left + ((trial - 1) / Math.max(trials - 1, 1)) * (PLOT_W - PAD.left - PAD.right)
  const y = (value: number) => {
    const span = scale.max - scale.min || 1
    return PAD.top + (1 - (value - scale.min) / span) * (CHART_H - PAD.top - PAD.bottom)
  }

  return (
    <figure>
      <svg
        viewBox={`0 0 ${PLOT_W} ${CHART_H}`}
        className="w-full"
        style={{ height: CHART_H }}
        role="img"
        aria-label={`Best-so-far ${metric} by trial number. RUNWISE ends at ${result.laya_best}, uniform random search at ${result.random_best}.`}
      >
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              y1={y(tick)}
              x2={PLOT_W - PAD.right}
              y2={y(tick)}
              stroke="var(--color-ink-800)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 8}
              y={y(tick) + 3}
              textAnchor="end"
              className="fill-ink-500 font-mono"
              style={{ fontSize: 10 }}
            >
              {formatNumber(tick, 3)}
            </text>
          </g>
        ))}

        {/* x axis */}
        <line
          x1={PAD.left}
          y1={CHART_H - PAD.bottom}
          x2={PLOT_W - PAD.right}
          y2={CHART_H - PAD.bottom}
          stroke="var(--color-ink-700)"
          strokeWidth={1}
        />
        {Array.from({ length: trials }, (_, index) => index + 1)
          .filter((trial) => trial === 1 || trial === trials || trial % Math.ceil(trials / 6) === 0)
          .map((trial) => (
            <text
              key={trial}
              x={x(trial)}
              y={CHART_H - PAD.bottom + 14}
              textAnchor="middle"
              className="fill-ink-500 font-mono"
              style={{ fontSize: 10 }}
            >
              {trial}
            </text>
          ))}
        <text
          x={(PLOT_W + PAD.left) / 2}
          y={CHART_H - 4}
          textAnchor="middle"
          className="fill-ink-600"
          style={{ fontSize: 9, letterSpacing: '0.08em' }}
        >
          TRIAL NUMBER
        </text>

        {arms.map((arm) => (
          <Line key={arm.label} curve={arm.curve} x={x} y={y} stroke={arm.color} />
        ))}
      </svg>

      <figcaption className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-500">
        {arms.map((arm) => (
          <LegendKey key={arm.label} color={arm.color} label={arm.label} value={arm.best} metric={metric} />
        ))}
        <span className="ml-auto">
          Best-so-far {metric}, not per-trial score.{' '}
          {isMinimized(metric)
            ? 'A curve that stops falling is the run converging.'
            : 'A curve that stops rising is the run converging.'}
        </span>
      </figcaption>
    </figure>
  )
}

function Line({
  curve,
  x,
  y,
  stroke,
}: {
  curve: AblationCurvePoint[]
  x: (trial: number) => number
  y: (value: number) => number
  stroke: string
}) {
  if (curve.length === 0) return null
  const path = curve.map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.trial)},${y(point.best_score)}`).join(' ')
  const last = curve[curve.length - 1]
  return (
    <g>
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.8} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(last.trial)} cy={y(last.best_score)} r={3} fill={stroke} />
    </g>
  )
}

function LegendKey({
  color,
  label,
  value,
  metric,
}: {
  color: string
  label: string
  value: number | null
  metric: string
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-0.5 w-4 rounded" style={{ background: color }} />
      {label}
      <span className="font-mono text-ink-200" data-numeric>
        {value === null || value === undefined ? '—' : `${formatNumber(value, 4)} ${isMinimized(metric) ? '↓' : '↑'}`}
      </span>
    </span>
  )
}

function formatDurationShort(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`
  return `${(seconds / 3600).toFixed(2)}h`
}