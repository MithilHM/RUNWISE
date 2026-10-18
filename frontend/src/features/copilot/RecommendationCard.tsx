import { useState } from 'react'
import { useRecommendationDecision } from '../../api/hooks'
import type { Recommendation } from '../../api/types'
import { EvidenceStatusChip, IdChip } from '../../components/Chips'
import { ScoreBreakdown } from '../../components/ScoreBreakdown'
import { formatJson, orUnrecorded, titleCase } from '../../lib/format'

/**
 * A proposal card (FR-5, FR-8, FR-9).
 *
 * Three things are deliberately always visible: the epistemic grade (a proposal is
 * `inferred`, never an observation), the score as a weighted breakdown rather than a
 * probability, and the caveats. An answer with no caveats should look suspicious.
 */
export function RecommendationCard({ recommendation }: { recommendation: Recommendation }) {
  const decision = useRecommendationDecision()
  const [note, setNote] = useState('')
  const [showJson, setShowJson] = useState(false)

  const decided = recommendation.status !== 'proposed'

  const decide = (value: 'approved' | 'rejected') => {
    void decision.mutateAsync({
      id: recommendation.id,
      decision: value,
      decided_by: 'researcher',
      note: note || undefined,
    })
  }

  return (
    <article className="panel">
      <div className="panel-header">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium text-ink-100">{titleCase(recommendation.candidate_kind)}</span>
            <EvidenceStatusChip status={recommendation.evidence_status} />
            {decided && (
              <span className="rounded border border-ink-600 px-1.5 py-0.5 font-mono text-[10px] text-ink-300">
                {recommendation.status}
              </span>
            )}
          </div>
          <p className="mono-id mt-0.5">{recommendation.id}</p>
        </div>
      </div>

      <div className="space-y-4 px-4 py-4">
        <p className="text-sm text-ink-100">{recommendation.hypothesis}</p>
        {recommendation.rationale && <p className="text-xs text-ink-400">{recommendation.rationale}</p>}

        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="Baseline runs">
            <IdList ids={recommendation.baseline_experiment_ids} />
          </Field>
          <Field label="Supporting runs">
            <IdList ids={recommendation.supporting_experiment_ids} />
          </Field>
          <Field label="Independent variables">
            {recommendation.independent_variables.length ? (
              <ul className="space-y-0.5">
                {recommendation.independent_variables.map((variable) => (
                  <li key={variable} className="font-mono text-xs text-ink-200">
                    {variable}
                    {recommendation.independent_variable_values[variable] !== undefined && (
                      <span className="ml-2 text-signal-400">
                        → {JSON.stringify(recommendation.independent_variable_values[variable])}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <span className="text-xs text-ink-500">none recorded</span>
            )}
          </Field>
          <Field label="Controlled variables">
            {recommendation.controlled_variables.length ? (
              <ul className="space-y-0.5 text-xs text-ink-300">
                {recommendation.controlled_variables.map((variable) => (
                  <li key={variable}>{variable}</li>
                ))}
              </ul>
            ) : (
              <span className="text-xs text-ink-500">none recorded</span>
            )}
          </Field>
          <Field label="Metrics">
            {recommendation.metrics.length ? (
              <span className="text-xs text-ink-300">{recommendation.metrics.join(', ')}</span>
            ) : (
              <span className="text-xs text-ink-500">none recorded</span>
            )}
          </Field>
          <Field label="Estimated compute">
            <span className="font-mono text-xs text-ink-200">
              {orUnrecorded(recommendation.estimated_gpu_hours, ' GPU-h')}
            </span>
            {recommendation.estimate_source && (
              <div className="text-[11px] text-ink-500">source: {recommendation.estimate_source}</div>
            )}
          </Field>
        </div>

        {recommendation.validation_criteria.length > 0 && (
          <div>
            <h4 className="label">Validation & stopping criteria</h4>
            <ul className="mt-1.5 space-y-1 text-xs text-ink-300">
              {recommendation.validation_criteria.map((criterion, index) => (
                <li key={`${criterion.kind}-${index}`} className="flex items-start gap-2">
                  <span className="rounded bg-ink-800 px-1.5 py-0.5 font-mono text-[10px] text-ink-400">
                    {criterion.kind}
                  </span>
                  <span>
                    {criterion.description}
                    {criterion.is_suggested_default && (
                      <span className="ml-2 text-[10px] text-ink-500">suggested by RUNWISE, not chosen by you</span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {recommendation.uncertainties.length > 0 && (
          <div className="rounded border border-warn-400/30 bg-warn-400/5 p-3">
            <h4 className="label text-warn-400">Uncertainties</h4>
            <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-ink-300">
              {recommendation.uncertainties.map((item, index) => (
                <li key={index}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        <ScoreBreakdown
          score={recommendation.score}
          scoreKind={recommendation.score_kind}
          components={recommendation.score_components}
        />

        <div>
          <button type="button" className="text-[11px] text-ink-400 underline" onClick={() => setShowJson((v) => !v)}>
            {showJson ? 'Hide' : 'Export'} proposed configuration as JSON (FR-9)
          </button>
          {showJson && (
            <pre className="mt-2 max-h-56 overflow-auto rounded border border-ink-700 bg-ink-850 p-3 font-mono text-[11px] text-ink-300">
              {formatJson({
                objective: recommendation.objective,
                configuration: recommendation.configuration,
                independent_variables: recommendation.independent_variables,
                independent_variable_values: recommendation.independent_variable_values,
                controlled_variables: recommendation.controlled_variables,
                metrics: recommendation.metrics,
                estimated_gpu_hours: recommendation.estimated_gpu_hours,
              })}
            </pre>
          )}
        </div>

        {!decided && (
          <div className="flex flex-wrap items-center gap-2 border-t border-ink-700 pt-3">
            <input
              className="input flex-1"
              placeholder="Note for the record (optional)"
              value={note}
              onChange={(event) => setNote(event.target.value)}
            />
            <button
              type="button"
              className="btn btn-primary"
              disabled={decision.isPending}
              onClick={() => decide('approved')}
            >
              Approve
            </button>
            <button
              type="button"
              className="btn btn-danger"
              disabled={decision.isPending}
              onClick={() => decide('rejected')}
            >
              Reject
            </button>
            {decision.isError && (
              <span className="text-[11px] text-danger-400">
                {decision.error instanceof Error ? decision.error.message : 'Decision failed'}
              </span>
            )}
          </div>
        )}
      </div>
    </article>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="label">{label}</h4>
      <div className="mt-1">{children}</div>
    </div>
  )
}

function IdList({ ids }: { ids: string[] }) {
  if (!ids.length) return <span className="text-xs text-ink-500">none recorded</span>
  return (
    <span className="flex flex-wrap gap-1">
      {ids.map((id) => (
        <IdChip key={id} id={id} />
      ))}
    </span>
  )
}
