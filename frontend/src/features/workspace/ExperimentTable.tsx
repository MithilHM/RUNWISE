import type { ExperimentSummary } from '../../api/types'
import { IdChip, QualityBadge } from '../../components/Chips'
import { formatDuration, formatMetric, formatTimestamp, orUnrecorded } from '../../lib/format'

const STATUS_TONE: Record<string, string> = {
  completed: 'text-ink-300',
  failed: 'text-danger-400',
  running: 'text-warn-400',
  cancelled: 'text-ink-500',
}

/**
 * The experiment history table.
 *
 * Numeric columns are right-aligned and monospaced so they compare by position
 * rather than by shape, which is the only way to read a column of scores. Rows
 * under an open correction say so.
 */
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
            <tr className="border-b border-ink-700">
              <th className="table-head whitespace-nowrap">Run</th>
              <th className="table-head whitespace-nowrap">Family</th>
              <th className="table-head whitespace-nowrap">Dataset version</th>
              <th className="table-head whitespace-nowrap">Status</th>
              <th className="table-head whitespace-nowrap text-right">Accuracy</th>
              <th className="table-head whitespace-nowrap text-right">
                Cost
                <span className="ml-1 font-normal text-ink-500">GPU-h</span>
              </th>
              <th className="table-head whitespace-nowrap text-right">Duration</th>
              <th className="table-head whitespace-nowrap">Recorded</th>
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
                  aria-selected={selected}
                  className={`cursor-pointer border-b border-ink-850 transition-colors duration-100 last:border-0 ${
                    selected ? 'bg-ink-800' : 'hover:bg-ink-850'
                  }`}
                >
                  <td className="table-cell">
                    <div className="flex items-center gap-2">
                      <IdChip id={experiment.id} />
                      <QualityBadge status={experiment.quality_status} />
                    </div>
                    <div className="truncate text-[11px] text-ink-500">{experiment.name}</div>
                  </td>
                  <td className="table-cell whitespace-nowrap text-ink-300">
                    {experiment.model_family}
                  </td>
                  <td className="table-cell">
                    {experiment.dataset_version_id ? (
                      <IdChip id={experiment.dataset_version_id} />
                    ) : (
                      <span className="text-ink-500">not recorded</span>
                    )}
                  </td>
                  <td
                    className={`table-cell whitespace-nowrap ${
                      STATUS_TONE[experiment.status] ?? 'text-ink-300'
                    }`}
                  >
                    {experiment.status}
                  </td>
                  <td className="table-num">
                    {accuracy ? (
                      formatMetric(accuracy)
                    ) : (
                      <span className="text-ink-500">not recorded</span>
                    )}
                  </td>
                  <td className="table-num">
                    {experiment.compute_cost_gpu_hours === null ? (
                      <span className="text-ink-500">not recorded</span>
                    ) : (
                      orUnrecorded(experiment.compute_cost_gpu_hours)
                    )}
                  </td>
                  <td className="table-num">{formatDuration(experiment.duration_seconds)}</td>
                  <td className="table-cell whitespace-nowrap text-[11px] text-ink-500">
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