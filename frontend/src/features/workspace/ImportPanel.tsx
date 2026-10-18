import { useRef, useState } from 'react'
import { useImportFile } from '../../api/hooks'
import type { ImportSummary } from '../../api/types'
import { ErrorState } from '../../components/States'

/**
 * FR-1 import.
 *
 * Importing is occasional, so it stays a control rather than a panel occupying
 * the top of the screen. The validation summary is the point of the feature, not
 * a toast, so it expands in place and stays until the next import.
 */
export function ImportPanel({ projectId }: { projectId: string }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [summary, setSummary] = useState<ImportSummary | null>(null)
  const [expanded, setExpanded] = useState(false)
  const mutation = useImportFile(projectId)

  const onFile = async (file: File | undefined) => {
    if (!file) return
    setSummary(null)
    setExpanded(true)
    try {
      setSummary(await mutation.mutateAsync(file))
    } catch {
      /* rendered from mutation.error below */
    }
    if (inputRef.current) inputRef.current.value = ''
  }

  const showDetail = expanded && (summary !== null || mutation.isError)

  return (
    <div className="panel overflow-hidden">
      <div className="flex items-center gap-3 px-4 py-3">
        <div className="min-w-0">
          <h2 className="text-sm font-medium text-ink-100">Import experiment history</h2>
          <p className="truncate text-[11px] text-ink-500">
            CSV or JSON. Unknown fields are preserved, never invented.
          </p>
        </div>
        <label className="btn ml-auto shrink-0 cursor-pointer">
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

      {showDetail && (
        <div className="border-t border-ink-750 px-4 py-3 text-xs">
          {mutation.isError && <ErrorState error={mutation.error} context="Import failed" />}

          {summary && (
            <>
              <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
                <span className="text-good-400">{summary.imported} imported</span>
                <span className="text-ink-300">{summary.skipped} skipped</span>
                <span className={summary.failed ? 'text-danger-400' : 'text-ink-300'}>
                  {summary.failed} failed
                </span>
              </div>
              {summary.errors.length > 0 && (
                <details className="mt-2.5">
                  <summary className="cursor-pointer text-ink-400">
                    {summary.errors.length} validation error{summary.errors.length === 1 ? '' : 's'}
                  </summary>
                  <ul className="mt-2 max-h-48 space-y-1 overflow-auto border-t border-ink-850 pt-2">
                    {summary.errors.map((item, index) => (
                      <li key={`${item.row}-${index}`} className="text-ink-400">
                        {item.row !== null && (
                          <span className="font-mono text-ink-500">row {item.row}</span>
                        )}
                        {item.experiment_id && <span className="mono-id ml-2">{item.experiment_id}</span>}
                        <span className="ml-2">{item.reason}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}