import { useState } from 'react'
import type { ExperimentFilters } from '../../api/types'

export interface FilterState extends ExperimentFilters {
  q: string
}

const EMPTY: FilterState = { q: '' }

const FIELD = 'flex min-w-0 flex-col gap-1.5'

/**
 * FR-2 filters. Every field here is enforced by the graph query, not by the table.
 *
 * The three fields researchers reach for are always visible. Configuration
 * probing is a different task from browsing, so it lives behind a disclosure
 * instead of sitting in the primary row competing with everything else.
 */
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
  const [showAdvanced, setShowAdvanced] = useState(false)
  const set = (patch: Partial<FilterState>) => onChange({ ...value, ...patch })
  const isFiltered = JSON.stringify({ ...value, q: '' }) !== JSON.stringify(EMPTY)
  const hasConfigFilter = Boolean(value.config_field || value.config_value)

  return (
    <div className="panel px-4 py-3.5">
      <div className="flex flex-wrap items-end gap-x-3 gap-y-3">
        <label className={`${FIELD} w-56`}>
          <span className="label">Search</span>
          <input
            className="input"
            placeholder="name or id"
            value={value.q}
            onChange={(event) => set({ q: event.target.value })}
          />
        </label>

        <label className={`${FIELD} w-44`}>
          <span className="label">Model family</span>
          <select
            className="input"
            value={value.model_family ?? ''}
            onChange={(e) => set({ model_family: e.target.value || undefined })}
          >
            <option value="">any</option>
            {families.map((family) => (
              <option key={family} value={family}>
                {family}
              </option>
            ))}
          </select>
        </label>

        <label className={`${FIELD} w-48`}>
          <span className="label">Dataset version</span>
          <select
            className="input"
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

        <label className={`${FIELD} w-36`}>
          <span className="label">Status</span>
          <select
            className="input"
            value={value.status ?? ''}
            onChange={(e) => set({ status: e.target.value || undefined })}
          >
            <option value="">any</option>
            {['completed', 'failed', 'running', 'cancelled'].map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </label>

        <label className={`${FIELD} w-40`}>
          <span className="label">Review state</span>
          <select
            className="input"
            value={value.quality_status ?? ''}
            onChange={(e) => set({ quality_status: e.target.value || undefined })}
          >
            <option value="">any</option>
            <option value="valid">valid</option>
            <option value="needs_review">needs_review</option>
            <option value="disputed">disputed</option>
          </select>
        </label>

        <button
          type="button"
          className="btn-ghost btn h-[38px]"
          aria-expanded={showAdvanced}
          onClick={() => setShowAdvanced((open) => !open)}
        >
          {showAdvanced ? 'Fewer filters' : 'Configuration filters'}
          {hasConfigFilter && (
            <span className="ml-1 size-1.5 rounded-full bg-signal-400" aria-hidden="true" />
          )}
        </button>

        {isFiltered && (
          <button type="button" className="btn h-[38px]" onClick={() => onChange({ q: '' })}>
            Clear all
          </button>
        )}
      </div>

      {showAdvanced && (
        <div className="mt-3.5 flex flex-wrap items-end gap-x-3 gap-y-3 border-t border-ink-800 pt-3.5">
          <label className={`${FIELD} w-48`}>
            <span className="label">Metric recorded</span>
            <input
              className="input"
              placeholder="validation_accuracy"
              value={value.metric ?? ''}
              onChange={(e) => set({ metric: e.target.value || undefined })}
            />
          </label>

          <label className={`${FIELD} w-40`}>
            <span className="label">Config key</span>
            <input
              className="input"
              placeholder="max_depth"
              value={value.config_field ?? ''}
              onChange={(e) => set({ config_field: e.target.value || undefined })}
            />
          </label>

          <label className={`${FIELD} w-32`}>
            <span className="label">Config value</span>
            <input
              className="input"
              placeholder="6"
              value={value.config_value ?? ''}
              onChange={(e) => set({ config_value: e.target.value || undefined })}
            />
          </label>

          <p className="ml-1 max-w-sm text-[11px] leading-relaxed text-ink-500">
            Configuration is matched after the graph query, because the stored document
            cannot be probed by a dynamic key on this FalkorDB build.
          </p>
        </div>
      )}
    </div>
  )
}