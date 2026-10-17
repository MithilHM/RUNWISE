import { Suspense, lazy, useMemo, useState } from 'react'
import { useExperiments } from '../api/hooks'
import { useProject } from '../lib/projectContext'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { ExperimentDrawer } from '../features/workspace/ExperimentDrawer'
import { FilterBar, type FilterState } from '../features/workspace/FilterBar'
import { ImportPanel } from '../features/workspace/ImportPanel'
import { ExperimentTable } from '../features/workspace/ExperimentTable'

// The charting library is only needed once there is something to plot.
const MetricChart = lazy(() =>
  import('../features/workspace/MetricChart').then((m) => ({ default: m.MetricChart })),
)

/**
 * Screen A: experiment workspace.
 *
 * The table is driven by the filtered query, the chart by the same result set, and
 * the drawer by the selected run. Nothing here holds a second copy of the data.
 */
export function WorkspacePage() {
  const { projectId, project } = useProject()
  const [filters, setFilters] = useState<FilterState>({ q: '' })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [metricName, setMetricName] = useState('validation_accuracy')

  const queryFilters = useMemo(
    () => ({
      model_family: filters.model_family,
      dataset_id: filters.dataset_id,
      dataset_version_id: filters.dataset_version_id,
      metric: filters.metric,
      evaluation_split: filters.evaluation_split,
      status: filters.status,
      quality_status: filters.quality_status,
      config_field: filters.config_field,
      config_value: filters.config_value,
      created_after: filters.created_after,
      created_before: filters.created_before,
      limit: 500,
    }),
    [filters],
  )

  const experiments = useExperiments(projectId, queryFilters)

  const visible = useMemo(() => {
    const rows = experiments.data ?? []
    const needle = filters.q.trim().toLowerCase()
    if (!needle) return rows
    return rows.filter(
      (row) => row.name.toLowerCase().includes(needle) || row.id.toLowerCase().includes(needle),
    )
  }, [experiments.data, filters.q])

  const families = useMemo(
    () =>
      Array.from(
        new Set(
          (experiments.data ?? [])
            .map((row) => row.model_family)
            .filter((family): family is string => Boolean(family)),
        ),
      ).sort(),
    [experiments.data],
  )
  const datasetVersions = useMemo(
    () => Array.from(new Set((experiments.data ?? []).map((row) => row.dataset_version_id).filter(Boolean) as string[])).sort(),
    [experiments.data],
  )
  const metricNames = useMemo(
    () => Array.from(new Set(visible.flatMap((row) => row.metrics.map((m) => m.name)))).sort(),
    [visible],
  )

  if (!projectId) {
    return (
      <EmptyState
        title="No project selected"
        hint="Create a project through POST /api/v1/projects, then import a CSV or JSON history to get started."
      />
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-lg font-medium text-ink-100">{project?.name ?? projectId}</h1>
          <p className="text-xs text-ink-500">
            {visible.length} run{visible.length === 1 ? '' : 's'} shown
            {project ? ` · ${project.experiment_count} stored` : ''}
          </p>
        </div>
      </div>

      <ImportPanel projectId={projectId} />
      <FilterBar value={filters} onChange={setFilters} families={families} datasetVersions={datasetVersions} />

      {experiments.isLoading ? (
        <div className="panel">
          <LoadingState label="Loading experiment history" />
        </div>
      ) : experiments.isError ? (
        <ErrorState error={experiments.error} onRetry={() => void experiments.refetch()} context="Could not load history" />
      ) : visible.length === 0 ? (
        <div className="panel">
          <EmptyState
            title="No run matches these filters"
            hint="Filters run inside the graph query. Clear one to widen the search, or import more history."
          />
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_480px]">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <span className="label">Chart metric</span>
              <select className="input w-56 py-1" value={metricName} onChange={(event) => setMetricName(event.target.value)}>
                {metricNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <Suspense fallback={<LoadingState label="Loading chart" />}>
              <MetricChart experiments={visible} metricName={metricName} />
            </Suspense>
            <ExperimentTable experiments={visible} selectedId={selectedId} onSelect={setSelectedId} />
          </div>

          <div className="xl:sticky xl:top-4 xl:self-start">
            {selectedId ? (
              <ExperimentDrawer experimentId={selectedId} projectId={projectId} />
            ) : (
              <div className="panel">
                <EmptyState
                  title="Select a run"
                  hint="The drawer shows its configuration, metrics, provenance, comparability assessment and lineage."
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
