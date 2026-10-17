import type { ExperimentFilters } from '../../api/types'

export interface FilterState extends ExperimentFilters {
  q: string
}

const EMPTY: FilterState = { q: '' }

/** FR-2 filters. Every field here is enforced by the graph query, not by the table. */
export function FilterBar({
  value,
  onChange,
  families,
  datasetVersions,
}: {
  value: FilterState
  onChange: (next: FilterState) => void
  families: string[]
  datasetVersions: string[]
}) {
  const set = (patch: Partial<FilterState>) => onChange({ ...value, ...patch })
  const isFiltered = JSON.stringify({ ...value, q: '' }) !== JSON.stringify(EMPTY)

  return (
    <div className="panel px-3 py-2">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="label">Search</span>
          <input
            className="input w-52"
            placeholder="name or id"
            value={value.q}
            onChange={(event) => set({ q: event.target.value })}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Model family</span>
          <select className="input w-40" value={value.model_family ?? ''} onChange={(e) => set({ model_family: e.target.value || undefined })}>
            <option value="">any</option>
            {families.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Dataset version</span>
          <select
            className="input w-44"
            value={value.dataset_version_id ?? ''}
            onChange={(e) => set({ dataset_version_id: e.target.value || undefined })}
          >
            <option value="">any</option>
            {datasetVersions.map((version) => (
              <option key={version} value={version}>
                {version}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Status</span>
          <select className="input w-32" value={value.status ?? ''} onChange={(e) => set({ status: e.target.value || undefined })}>
            <option value="">any</option>
            {['completed', 'failed', 'running', 'cancelled'].map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Review state</span>
          <select
            className="input w-36"
            value={value.quality_status ?? ''}
            onChange={(e) => set({ quality_status: e.target.value || undefined })}
          >
            <option value="">any</option>
            <option value="valid">valid</option>
            <option value="needs_review">needs_review</option>
            <option value="disputed">disputed</option>
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Metric</span>
          <input
            className="input w-40"
            placeholder="validation_accuracy"
            value={value.metric ?? ''}
            onChange={(e) => set({ metric: e.target.value || undefined })}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Config key</span>
          <input
            className="input w-36"
            placeholder="max_depth"
            value={value.config_field ?? ''}
            onChange={(e) => set({ config_field: e.target.value || undefined })}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="label">Config value</span>
          <input
            className="input w-32"
            placeholder="6"
            value={value.config_value ?? ''}
            onChange={(e) => set({ config_value: e.target.value || undefined })}
          />
        </label>

        {isFiltered && (
          <button type="button" className="btn" onClick={() => onChange({ q: '' })}>
            Clear
          </button>
        )}
      </div>
    </div>
  )
}
