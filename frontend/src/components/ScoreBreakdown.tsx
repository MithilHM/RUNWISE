import type { ScoreComponent } from '../api/types'
import { formatNumber } from '../lib/format'

/**
 * A heuristic score shown as its parts.
 *
 * The total is a weighted sum of documented factors, not a probability, so the UI
 * never renders it as a percentage chance of success: the components and their
 * weights are always visible, letting a researcher disagree with one term
 * (HANDOFF.md UI rule 2, ranking_rules.md section 3).
 */
export function ScoreBreakdown({
  score,
  scoreKind,
  components,
}: {
  score: number
  scoreKind: string
  components: ScoreComponent[]
}) {
  if (!components?.length) {
    return (
      <p className="text-xs text-ink-400">
        Heuristic score {formatNumber(score)} ({scoreKind}). No component breakdown was returned.
      </p>
    )
  }

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label">Heuristic score</span>
        <span className="font-mono text-sm text-ink-100" title={`${scoreKind}: not a probability`}>
          {formatNumber(score)}
        </span>
      </div>
      <div className="mt-1 h-1 w-full overflow-hidden rounded bg-ink-800" aria-hidden>
        <div className="h-full bg-signal-500" style={{ width: `${Math.round(Math.min(Math.max(score, 0), 1) * 100)}%` }} />
      </div>
      <dl className="mt-3 space-y-1.5">
        {components.map((component) => (
          <div key={component.name} className="grid grid-cols-[1fr_auto] gap-x-3 text-xs">
            <dt className="text-ink-300">
              {component.name}
              <span className="ml-1.5 font-mono text-[10px] text-ink-600">w {formatNumber(component.weight, 2)}</span>
              <div className="text-ink-500">{component.rationale}</div>
            </dt>
            <dd className="text-right font-mono text-ink-300">
              {formatNumber(component.value, 2)}
              <span className="ml-1 text-ink-600">→</span>
              <span className="ml-1 text-ink-100">{formatNumber(component.weighted_value, 3)}</span>
            </dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-[11px] text-ink-600">
        A documented weighted heuristic ({scoreKind}), not a calibrated probability.
      </p>
    </div>
  )
}
