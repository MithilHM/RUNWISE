import type { ComparabilityAssessment, ExcludedEvidence, ResearchAnswer } from '../../api/types'
import { ComparabilityBadge, EvidenceStatusChip, IdChip, SeverityDot } from '../../components/Chips'
import { EmptyState } from '../../components/States'
import { formatNumber } from '../../lib/format'
import { AssessmentView } from '../workspace/ExperimentDrawer'

/**
 * The evidence behind an answer.
 *
 * Every non-comparable verdict is listed, including runs outside the cohort, so an
 * exclusion can be explained instead of appearing as a silent gap (FR-8). Withheld
 * records are shown as withheld: they exist, they are not trusted right now, and the
 * difference matters.
 */
export function AnswerEvidence({ answer }: { answer: ResearchAnswer }) {
  const claims = answer.claims ?? []
  const comparisons = answer.comparisons ?? []

  return (
    <div className="space-y-4">
      <section className="panel">
        <div className="panel-header">
          <h3 className="text-sm font-medium text-ink-100">Claims</h3>
          <span className="text-[11px] text-ink-500">{claims.length}</span>
        </div>
        <div className="space-y-3 px-4 py-3">
          {claims.length === 0 ? (
            <EmptyState title="This answer made no claims." />
          ) : (
            claims.map((claim) => (
              <div key={claim.id} className="border-l-2 border-ink-700 pl-3">
                <div className="flex flex-wrap items-center gap-2">
                  <EvidenceStatusChip status={claim.evidence_status} />
                  <span className="font-mono text-[10px] text-ink-500">{claim.id}</span>
                </div>
                <p className="mt-1 text-sm text-ink-200">{claim.text}</p>
                {claim.detail && <p className="mt-0.5 text-xs text-ink-500">{claim.detail}</p>}
                {claim.support.length > 0 && (
                  <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-ink-500">
                    <span>support</span>
                    {claim.support.map((ref, index) => (
                      <span key={`${claim.id}-${index}`} className="flex items-center gap-1">
                        {ref.experiment_id && <IdChip id={ref.experiment_id} />}
                        {ref.metric_name && (
                          <span className="font-mono text-ink-400">
                            {ref.metric_name}={ref.metric_value === null ? '—' : formatNumber(ref.metric_value)}
                          </span>
                        )}
                        <EvidenceStatusChip status={ref.evidence_status} />
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h3 className="text-sm font-medium text-ink-100">Comparability verdicts</h3>
            <p className="text-[11px] text-ink-500">
              Every run that is not cleanly comparable, with the reason it was excluded
            </p>
          </div>
        </div>
        <div className="divide-y divide-ink-800">
          {comparisons.length === 0 ? (
            <EmptyState title="No run was excluded from this comparison." />
          ) : (
            comparisons.map((assessment, index) => (
              <ComparisonRow
                key={`${assessment.left_experiment_id}-${assessment.right_experiment_id}-${index}`}
                assessment={assessment}
              />
            ))
          )}
        </div>
      </section>

      {answer.excluded_evidence.length > 0 && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h3 className="text-sm font-medium text-ink-100">Withheld evidence</h3>
              <p className="text-[11px] text-ink-500">
                These records exist and were not used as support. Nothing was deleted.
              </p>
            </div>
          </div>
          <ul className="divide-y divide-ink-800">
            {answer.excluded_evidence.map((item) => (
              <ExcludedRow key={item.experiment_id} item={item} />
            ))}
          </ul>
        </section>
      )}

      <section className="grid gap-3 sm:grid-cols-2">
        <CaveatList title="Uncertainties" items={answer.uncertainties} tone="warn" />
        <CaveatList title="Missing information" items={answer.missing_information} tone="neutral" />
      </section>

      {answer.grounding && (
        <section className="panel px-4 py-3 text-xs">
          <h3 className="label">Grounding</h3>
          <p className="mt-1 text-ink-300">
            {answer.grounding.cited_experiment_ids.length} citation(s) matched against{' '}
            {answer.grounding.available_experiment_count} retrieved record(s) · coverage{' '}
            {formatNumber(answer.grounding.citation_coverage, 2)}
          </p>
          {answer.grounding.unknown_citations.length > 0 && (
            <p className="mt-1 text-danger-400">
              Unmatched citations: {answer.grounding.unknown_citations.join(', ')}
            </p>
          )}
        </section>
      )}
    </div>
  )
}

function ComparisonRow({ assessment }: { assessment: ComparabilityAssessment }) {
  return (
    <details className="px-4 py-2.5">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2">
        <ComparabilityBadge status={assessment.status} />
        <IdChip id={assessment.right_experiment_id} />
        <span className="font-mono text-[11px] text-ink-500">heuristic {assessment.score.toFixed(2)}</span>
        <span className="flex-1 text-[11px] text-ink-500">
          {assessment.reasons.length} reason{assessment.reasons.length === 1 ? '' : 's'}
        </span>
      </summary>
      <div className="mt-2">
        <AssessmentView assessment={assessment} />
      </div>
    </details>
  )
}

function ExcludedRow({ item }: { item: ExcludedEvidence }) {
  return (
    <li className="flex flex-wrap items-center gap-2 px-4 py-2 text-xs">
      <IdChip id={item.experiment_id} />
      <EvidenceStatusChip status={item.status} />
      <span className="text-ink-300">{item.reason}</span>
      <span className="text-ink-500">{item.detail}</span>
    </li>
  )
}

function CaveatList({
  title,
  items,
  tone,
}: {
  title: string
  items: string[]
  tone: 'warn' | 'neutral'
}) {
  return (
    <div
      className={`panel p-3 ${
        items.length ? (tone === 'warn' ? 'border-warn-400/30' : '') : ''
      }`}
    >
      <h3 className={`label ${items.length && tone === 'warn' ? 'text-warn-400' : ''}`}>
        {title} ({items.length})
      </h3>
      {items.length === 0 ? (
        <p className="mt-1 text-xs text-ink-500">
          None reported. If an answer looks this clean, treat it with suspicion.
        </p>
      ) : (
        <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-ink-300">
          {items.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      )}
    </div>
  )
}

export function SeverityLegend() {
  return (
    <div className="flex flex-wrap items-center gap-3 text-[11px] text-ink-500">
      {(['blocker', 'major', 'minor', 'info'] as const).map((severity) => (
        <span key={severity} className="flex items-center gap-1.5">
          <SeverityDot severity={severity} />
          {severity}
        </span>
      ))}
    </div>
  )
}
