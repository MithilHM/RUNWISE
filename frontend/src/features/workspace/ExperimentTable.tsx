import type { ExperimentSummary } from '../../api/types'
import { IdChip, QualityBadge } from '../../components/Chips'
import { formatDuration, formatMetric, formatTimestamp, orUnrecorded } from '../../lib/format'

/** The experiment history table. Rows under an open correction say so. */
export function ExperimentTable({
  experiments,
  selectedId,
  onSelect,
}: {
  experiments: ExperimentSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  return (
    <div className="panel overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-ink-700 text-left">
              {['Run', 'Family', 'Dataset version', 'Status', 'Validation accuracy', 'Cost (GPU-h)', 'Duration', 'Recorded'].map(
                (heading) => (
                  <th key={heading} className="label px-3 py-2 font-semibold">
                    {heading}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody>
            {experiments.map((experiment) => {
              const accuracy = experiment.metrics.find((m) => m.name === 'validation_accuracy')
              const selected = experiment.id === selectedId
              return (
                <tr
                  key={experiment.id}
                  onClick={() => onSelect(experiment.id)}
                  className={`cursor-pointer border-b border-ink-800 transition ${
                    selected ? 'bg-ink-800' : 'hover:bg-ink-850'
                  }`}
                >
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <IdChip id={experiment.id} />
                      <QualityBadge status={experiment.quality_status} />
                    </div>
                    <div className="text-[11px] text-ink-500">{experiment.name}</div>
                  </td>
                  <td className="table-cell">{experiment.model_family}</td>
                  <td className="table-cell">
                    {experiment.dataset_version_id ? (
                      <IdChip id={experiment.dataset_version_id} />
                    ) : (
                      <span className="text-ink-600">not recorded</span>
                    )}
                  </td>
                  <td className="table-cell">{experiment.status}</td>
                  <td className="table-cell font-mono">
                    {accuracy ? (
                      formatMetric(accuracy)
                    ) : (
                      <span className="text-ink-600">not recorded</span>
                    )}
                  </td>
                  <td className="table-cell font-mono">
                    {experiment.compute_cost_gpu_hours === null ? (
                      <span className="text-ink-600">not recorded</span>
                    ) : (
                      orUnrecorded(experiment.compute_cost_gpu_hours)
                    )}
                  </td>
                  <td className="table-cell font-mono">{formatDuration(experiment.duration_seconds)}</td>
                  <td className="table-cell text-[11px] text-ink-500">
                    {formatTimestamp(experiment.created_at)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
