import { useState } from 'react'
import { useCreateCorrection } from '../../api/hooks'
import type { CorrectionImpactResponse } from '../../api/types'
import { IdChip } from '../../components/Chips'
import { ErrorState } from '../../components/States'
import { formatTimestamp } from '../../lib/format'

/**
 * Correction and impact (FR-6).
 *
 * A correction never deletes anything. It marks the source unreliable, flags the
 * runs that depend on it, and reports the three tiers separately: directly affected,
 * downstream, and explicitly unaffected. Showing the unaffected tier is the point —
 * "everything is now suspect" is not an impact analysis.
 */
export function ImpactPanel({
  projectId,
  impact,
  onImpact,
}: {
  projectId: string
  impact: CorrectionImpactResponse | null
  onImpact: (impact: CorrectionImpactResponse) => void
}) {
  const [sourceId, setSourceId] = useState('')
  const [sourceType, setSourceType] = useState('dataset_version')
  const [description, setDescription] = useState('')
  const correction = useCreateCorrection(projectId)

  const submit = async () => {
    if (!sourceId.trim() || !description.trim()) return
    const result = await correction.mutateAsync({
      source_id: sourceId.trim(),
      source_type: sourceType,
      description: description.trim(),
    })
    onImpact(result)
  }

  return (
    <div className="space-y-4">
      <div className="panel">
        <div className="panel-header">
          <div>
            <h2 className="text-sm font-medium text-ink-100">Report unreliable evidence</h2>
            <p className="text-[11px] text-ink-500">
              Marks the source, traverses real dependencies, and opens a revalidation queue. Nothing is deleted.
            </p>
          </div>
        </div>
        <div className="grid gap-3 px-4 py-3 sm:grid-cols-2">
          <label className="flex flex-col gap-1">
            <span className="label">Source node id</span>
            <input
              className="input"
              placeholder="tabular_fraud_v1"
              value={sourceId}
              onChange={(event) => setSourceId(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="label">Source type</span>
            <select className="input" value={sourceType} onChange={(event) => setSourceType(event.target.value)}>
              <option value="dataset_version">dataset version</option>
              <option value="experiment">experiment</option>
              <option value="metric">metric</option>
              <option value="claim">claim</option>
              <option value="evidence">evidence</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 sm:col-span-2">
            <span className="label">What is wrong, and what is the evidence?</span>
            <textarea
              className="input h-20 resize-y"
              placeholder="Train/validation leakage found in the v1 split."
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>
          <div className="sm:col-span-2">
            <button
              type="button"
              className="btn btn-danger"
              disabled={correction.isPending || !sourceId.trim() || !description.trim()}
              onClick={() => void submit()}
            >
              {correction.isPending ? 'Analysing impact…' : 'Record correction and analyse impact'}
            </button>
          </div>
          {correction.isError && (
            <div className="sm:col-span-2">
              <ErrorState error={correction.error} context="Correction failed" />
            </div>
          )}
        </div>
      </div>

      {impact && (
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3 className="text-sm font-medium text-ink-100">Impact report</h3>
              <p className="mono-id">{impact.correction_id}</p>
            </div>
            <span className="text-[11px] text-ink-500">{formatTimestamp(impact.generated_at)}</span>
          </div>

          <div className="px-4 py-3">
            <p className="text-sm text-ink-200">{impact.summary}</p>
            <p className="mt-1 text-xs text-ink-500">
              source <IdChip id={impact.source_id} /> ({impact.source_type ?? 'unknown type'})
            </p>

            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <Tier
                title="Directly affected"
                tone="danger"
                ids={impact.directly_affected_experiment_ids}
                hint="Read or wrote the corrected node itself."
              />
              <Tier
                title="Downstream"
                tone="warn"
                ids={impact.indirectly_affected_experiment_ids}
                hint="Reached through lineage, not through the defect itself."
              />
              <Tier
                title="Unaffected"
                tone="good"
                ids={impact.unaffected_experiment_ids}
                hint="Still usable as a baseline."
              />
            </div>

            {impact.potentially_affected_claim_ids.length > 0 && (
              <div className="mt-3">
                <h4 className="label">Conclusions to revalidate</h4>
                <div className="mt-1 flex flex-wrap gap-1">
                  {impact.potentially_affected_claim_ids.map((id) => (
                    <IdChip key={id} id={id} />
                  ))}
                </div>
              </div>
            )}

            {impact.paths.length > 0 && (
              <div className="mt-4">
                <h4 className="label">Dependency paths ({impact.paths.length})</h4>
                <ul className="mt-1.5 space-y-1.5">
                  {impact.paths.map((path, index) => (
                    <li key={index} className="flex flex-wrap items-center gap-1 text-[11px]">
                      {path.nodes.map((node, nodeIndex) => (
                        <span key={`${node}-${nodeIndex}`} className="flex items-center gap-1">
                          {nodeIndex > 0 && (
                            <span className="text-ink-500">--{path.relationship_types[nodeIndex - 1]}--&gt;</span>
                          )}
                          <IdChip id={node} />
                        </span>
                      ))}
                      {path.reason && <span className="ml-2 text-ink-500">{path.reason}</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function Tier({
  title,
  ids,
  hint,
  tone,
}: {
  title: string
  ids: string[]
  hint: string
  tone: 'danger' | 'warn' | 'good'
}) {
  const border =
    tone === 'danger' ? 'border-danger-400/40' : tone === 'warn' ? 'border-warn-400/40' : 'border-good-400/30'
  return (
    <div className={`rounded border ${border} bg-ink-850 p-3`}>
      <h4 className="label">
        {title} ({ids.length})
      </h4>
      <p className="mt-0.5 text-[11px] text-ink-500">{hint}</p>
      <div className="mt-2 flex flex-wrap gap-1">
        {ids.length ? (
          ids.map((id) => <IdChip key={id} id={id} />)
        ) : (
          <span className="text-[11px] text-ink-500">none</span>
        )}
      </div>
    </div>
  )
}
