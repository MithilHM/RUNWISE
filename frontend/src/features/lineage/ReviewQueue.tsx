import { useState } from 'react'
import { useResolveReviewItem, useReviewQueue } from '../../api/hooks'
import { IdChip } from '../../components/Chips'
import { EmptyState, ErrorState, LoadingState } from '../../components/States'
import { formatTimestamp } from '../../lib/format'

/** Revalidation queue (FR-6 #7, #9). Resolution is recorded, never erased. */
export function ReviewQueue({ projectId }: { projectId: string }) {
  const queue = useReviewQueue(projectId)
  const resolve = useResolveReviewItem()
  const [notes, setNotes] = useState<Record<string, string>>({})

  return (
    <div className="panel">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Revalidation queue</h2>
          <p className="text-[11px] text-ink-500">Runs an open correction reached. A human closes each one.</p>
        </div>
        <button type="button" className="btn" onClick={() => void queue.refetch()}>
          Refresh
        </button>
      </div>

      {queue.isLoading ? (
        <LoadingState label="Loading queue" />
      ) : queue.isError ? (
        <div className="p-3">
          <ErrorState error={queue.error} onRetry={() => void queue.refetch()} context="Could not load the queue" />
        </div>
      ) : !queue.data?.length ? (
        <EmptyState title="Nothing awaiting review" hint="No open correction currently reaches a run in this project." />
      ) : (
        <ul className="divide-y divide-ink-800">
          {queue.data.map((item) => (
            <li key={item.id} className="px-4 py-3">
              <div className="flex flex-wrap items-center gap-2">
                <IdChip id={item.experiment_id} />
                <span className="text-xs text-ink-300">{item.reason}</span>
                {item.correction_id && <IdChip id={item.correction_id} title="correction" />}
              </div>
              <p className="mt-1 text-[11px] text-ink-500">
                opened {formatTimestamp(item.created_at)}
                {item.resolved_at && ` · resolved ${formatTimestamp(item.resolved_at)}`}
                {item.resolved_by && ` by ${item.resolved_by}`}
              </p>
              {item.resolution_notes && <p className="mt-1 text-[11px] text-ink-400">{item.resolution_notes}</p>}

              {item.status === 'open' && (
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <input
                    className="input flex-1"
                    placeholder="Resolution note (what did you check?)"
                    value={notes[item.id] ?? ''}
                    onChange={(event) => setNotes((prev) => ({ ...prev, [item.id]: event.target.value }))}
                  />
                  <button
                    type="button"
                    className="btn"
                    disabled={resolve.isPending}
                    onClick={() =>
                      void resolve.mutateAsync({ id: item.id, status: 'resolved', notes: notes[item.id] })
                    }
                  >
                    Mark verified
                  </button>
                  <button
                    type="button"
                    className="btn"
                    disabled={resolve.isPending}
                    onClick={() =>
                      void resolve.mutateAsync({ id: item.id, status: 'dismissed', notes: notes[item.id] })
                    }
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
