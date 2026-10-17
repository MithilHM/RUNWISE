import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useComparability, useExperiment, useLineage, useRelated } from '../../api/hooks'
import type { ComparabilityAssessment, ExperimentDetail } from '../../api/types'
import { ComparabilityBadge, IdChip, QualityBadge, SeverityDot } from '../../components/Chips'
import { EmptyState, ErrorState, LoadingState } from '../../components/States'
import { formatDuration, formatJson, formatMetric, formatTimestamp } from '../../lib/format'

/** Detail drawer: configuration, metrics, provenance, comparability, lineage. */
export function ExperimentDrawer({ experimentId, projectId }: { experimentId: string; projectId: string }) {
  const { data, isLoading, error, refetch } = useExperiment(experimentId, projectId)
  const related = useRelated(experimentId, projectId)
  const [peer, setPeer] = useState<string | null>(null)
  const comparability = useComparability(experimentId, { peer: peer ?? undefined, projectId })

  if (isLoading) {
    return (
      <div className="panel">
        <LoadingState label="Loading run" />
      </div>
    )
  }
  if (error) {
    return (
      <div className="panel">
        <ErrorState error={error} onRetry={() => void refetch()} context="Could not load run" />
      </div>
    )
  }
  if (!data) return null

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-medium text-ink-100">{data.name}</h2>
            <QualityBadge status={data.quality_status} />
          </div>
          <p className="mono-id mt-0.5">{data.id}</p>
        </div>
        <Link to={`/lineage?experiment=${encodeURIComponent(data.id)}`} className="btn">
          Open in lineage screen
        </Link>
      </div>

      <DrawerBody experiment={data} projectId={projectId}>
        <section>
          <h3 className="label">Run</h3>
          <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <Row label="Family" value={data.model_family} />
            <Row label="Status" value={data.status} />
            <Row label="Dataset" value={data.dataset_name ?? null} />
            <Row label="Dataset version" value={data.dataset_version} />
            <Row label="Version node" value={data.dataset_version_id} mono />
            <Row label="Random seed" value={data.random_seed === null ? null : String(data.random_seed)} mono />
            <Row label="GPU hours" value={data.compute_cost_gpu_hours} />
            <Row label="Duration" value={formatDuration(data.duration_seconds)} />
            <Row label="Recorded" value={formatTimestamp(data.created_at)} />
            <Row label="Source" value={data.source_reference} mono />
          </dl>
          {data.description && <p className="mt-2 text-xs text-ink-400">{data.description}</p>}
        </section>

        <section>
          <h3 className="label">Metrics</h3>
          {data.metrics.length ? (
            <table className="mt-2 w-full text-xs">
              <thead>
                <tr className="text-left text-ink-500">
                  <th className="py-1 font-normal">name</th>
                  <th className="py-1 font-normal">value</th>
                  <th className="py-1 font-normal">direction</th>
                  <th className="py-1 font-normal">split</th>
                  <th className="py-1 font-normal">protocol</th>
                </tr>
              </thead>
              <tbody>
                {data.metrics.map((metric) => (
                  <tr key={metric.name} className="border-t border-ink-800">
                    <td className="py-1 text-ink-300">{metric.name}</td>
                    <td className="py-1 font-mono text-ink-100">
                      {metric.value === null ? (
                        <span className="text-ink-600">not recorded</span>
                      ) : (
                        formatMetric(metric)
                      )}
                    </td>
                    <td className="py-1 text-ink-400">{metric.direction}</td>
                    <td className="py-1 text-ink-400">{metric.evaluation_split ?? '—'}</td>
                    <td className="py-1 text-ink-400">{metric.evaluation_protocol || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="mt-1 text-xs text-ink-500">This run recorded no metrics.</p>
          )}
        </section>

        <section>
          <h3 className="label">Configuration</h3>
          <pre className="mt-2 max-h-56 overflow-auto rounded border border-ink-700 bg-ink-850 p-3 font-mono text-[11px] text-ink-300">
            {formatJson(data.configuration)}
          </pre>
        </section>

        <section>
          <h3 className="label">Comparability (FR-4)</h3>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <select className="input w-64 py-1" value={peer ?? ''} onChange={(event) => setPeer(event.target.value || null)}>
              <option value="">compare against best available peer</option>
              {(related.data?.same_dataset_version ?? []).map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
            {comparability.isFetching && <span className="text-[11px] text-ink-500">assessing…</span>}
          </div>
          {comparability.isError ? (
            <div className="mt-2">
              <ErrorState error={comparability.error} context="Comparability unavailable" />
            </div>
          ) : comparability.data ? (
            <AssessmentView assessment={comparability.data} />
          ) : (
            <p className="mt-2 text-xs text-ink-500">Pick a peer to see why two runs may or may not be compared.</p>
          )}
        </section>

        <section>
          <h3 className="label">Related runs</h3>
          {related.isLoading ? (
            <LoadingState label="Loading related runs" />
          ) : related.data ? (
            <div className="mt-2 space-y-2">
              {Object.entries(related.data).map(([bucket, ids]) => (
                <div key={bucket} className="flex items-start gap-2 text-xs">
                  <span className="w-44 shrink-0 text-ink-500">{bucket.replace(/_/g, ' ')}</span>
                  <span className="flex flex-wrap gap-1">
                    {ids.length ? (
                      ids.map((id) => <IdChip key={id} id={id} />)
                    ) : (
                      <span className="text-ink-600">none</span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </section>
      </DrawerBody>
    </div>
  )
}

function DrawerBody({ experiment, projectId, children }: { experiment: ExperimentDetail; projectId: string; children: React.ReactNode }) {
  const lineage = useLineage(experiment.id, projectId)
  return (
    <div className="space-y-5 px-4 py-4">
      {children}
      <section>
        <h3 className="label">Lineage</h3>
        {lineage.isLoading ? (
          <LoadingState label="Loading lineage" />
        ) : lineage.data ? (
          <div className="mt-2 space-y-1 text-xs">
            <div className="flex gap-2">
              <span className="w-24 shrink-0 text-ink-500">parents</span>
              <span className="flex flex-wrap gap-1">
                {experiment.parent_experiment_ids.length ? (
                  experiment.parent_experiment_ids.map((id) => <IdChip key={id} id={id} />)
                ) : (
                  <span className="text-ink-600">none recorded</span>
                )}
              </span>
            </div>
            <div className="flex gap-2">
              <span className="w-24 shrink-0 text-ink-500">children</span>
              <span className="flex flex-wrap gap-1">
                {experiment.child_experiment_ids.length ? (
                  experiment.child_experiment_ids.map((id) => <IdChip key={id} id={id} />)
                ) : (
                  <span className="text-ink-600">none recorded</span>
                )}
              </span>
            </div>
            <p className="text-[11px] text-ink-600">
              {lineage.data.nodes.length} node(s), {lineage.data.edges.length} lineage edge(s) within{' '}
              {lineage.data.nodes.reduce((deepest, node) => Math.max(deepest, Math.abs(node.depth)), 0)} hop(s)
            </p>
          </div>
        ) : null}
      </section>
    </div>
  )
}

export function AssessmentView({ assessment }: { assessment: ComparabilityAssessment }) {
  return (
    <div className="mt-2 rounded border border-ink-700 bg-ink-850 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <ComparabilityBadge status={assessment.status} />
        <span className="font-mono text-[11px] text-ink-500">
          heuristic {assessment.score.toFixed(2)} ({assessment.score_kind})
        </span>
      </div>
      {assessment.explanation && <p className="mt-2 text-xs text-ink-300">{assessment.explanation}</p>}
      {assessment.reasons.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {assessment.reasons.map((reason, index) => (
            <li key={`${reason.code}-${index}`} className="flex items-start gap-2 text-xs">
              <span className="mt-1.5">
                <SeverityDot severity={reason.severity} />
              </span>
              <span>
                <span className="text-ink-200">{reason.message}</span>
                <span className="ml-2 font-mono text-[10px] text-ink-600">
                  {reason.code} · {reason.dimension} · {reason.severity}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
      {!assessment.reasons.length && (
        <EmptyState title="No differences found on any recorded dimension." />
      )}
    </div>
  )
}

function Row({ label, value, mono }: { label: string; value: string | number | null | undefined; mono?: boolean }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-ink-500">{label}</dt>
      <dd className={`text-right text-ink-200 ${mono ? 'font-mono' : ''}`}>
        {value === null || value === undefined || value === '' ? (
          <span className="text-ink-600">not recorded</span>
        ) : mono ? (
          <span className="mono-id">{value}</span>
        ) : (
          String(value)
        )}
      </dd>
    </div>
  )
}
