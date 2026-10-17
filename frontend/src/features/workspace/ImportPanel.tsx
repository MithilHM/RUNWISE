import { useRef, useState } from 'react'
import { useImportFile } from '../../api/hooks'
import type { ImportSummary } from '../../api/types'
import { ErrorState } from '../../components/States'

/** FR-1 import. The validation summary is the point of the panel, not a toast. */
export function ImportPanel({ projectId }: { projectId: string }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [summary, setSummary] = useState<ImportSummary | null>(null)
  const mutation = useImportFile(projectId)

  const onFile = async (file: File | undefined) => {
    if (!file) return
    setSummary(null)
    try {
      setSummary(await mutation.mutateAsync(file))
    } catch {
      /* rendered from mutation.error below */
    }
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Import experiment history</h2>
          <p className="text-[11px] text-ink-500">CSV or JSON. Unknown fields are preserved, never invented.</p>
        </div>
        <label className="btn cursor-pointer">
          {mutation.isPending ? 'Importing…' : 'Choose file'}
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.json"
            className="hidden"
            onChange={(event) => void onFile(event.target.files?.[0])}
          />
        </label>
      </div>

      {mutation.isError && (
        <div className="p-3">
          <ErrorState error={mutation.error} context="Import failed" />
        </div>
      )}

      {summary && (
        <div className="px-4 py-3 text-xs">
          <div className="flex flex-wrap gap-4">
            <span className="text-good-400">
              {summary.imported} imported
            </span>
            <span className="text-ink-300">{summary.skipped} skipped</span>
            <span className={summary.failed ? 'text-danger-400' : 'text-ink-300'}>
              {summary.failed} failed
            </span>
          </div>
          {summary.errors.length > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-ink-400">
                {summary.errors.length} validation error{summary.errors.length === 1 ? '' : 's'}
              </summary>
              <ul className="mt-2 max-h-48 space-y-1 overflow-auto border-t border-ink-800 pt-2">
                {summary.errors.map((item, index) => (
                  <li key={`${item.row}-${index}`} className="text-ink-400">
                    {item.row !== null && <span className="font-mono text-ink-600">row {item.row}</span>}
                    {item.experiment_id && <span className="mono-id ml-2">{item.experiment_id}</span>}
                    <span className="ml-2">{item.reason}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </div>
  )
}
