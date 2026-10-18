import React, { useState, useEffect, useRef } from 'react'
import type { LogEntry } from '../../api/types'


interface LogStreamProps {
  runId: string | null
  mode?: string
}

const LEVEL_COLORS: Record<string, string> = {
  INFO: 'text-slate-300',
  WARN: 'text-amber-400 font-semibold',
  ERROR: 'text-rose-400 font-semibold',
}

const STAGE_BADGES: Record<string, string> = {
  ingest: 'bg-sky-950 text-sky-400 border-sky-800',
  propose: 'bg-purple-950 text-purple-400 border-purple-800',
  decide: 'bg-amber-950 text-amber-400 border-amber-800',
  execute: 'bg-cyan-950 text-cyan-400 border-cyan-800',
  mutate: 'bg-emerald-950 text-emerald-400 border-emerald-800',
  done: 'bg-slate-800 text-slate-300 border-slate-700',
}

export const LogStream: React.FC<LogStreamProps> = ({ runId, mode }) => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!runId) {
      setLogs([])
      return
    }

    let active = true

    // 1. Fetch initial historical logs
    fetch(`/api/v1/runs/${runId}/logs?after=0`)
      .then((res) => (res.ok ? res.json() : { entries: [] }))
      .then((data) => {
        if (!active) return
        if (data.entries) {
          setLogs(data.entries.sort((a: LogEntry, b: LogEntry) => a.seq - b.seq))
        }
      })
      .catch((err) => console.error('Failed to load initial logs', err))

    // 2. Open SSE stream
    const eventSource = new EventSource(`/api/v1/runs/${runId}/events`)

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data)
        if (parsed.type === 'log' && parsed.payload) {
          const entry: LogEntry = parsed.payload
          setLogs((prev) => {
            if (prev.some((l) => l.seq === entry.seq)) return prev
            const next = [...prev, entry]
            return next.sort((a, b) => a.seq - b.seq)
          })
        }
      } catch (err) {
        // Heartbeat or malformed
      }
    }

    eventSource.onerror = () => {
      // Reconnect handled automatically by EventSource
    }

    return () => {
      active = false
      eventSource.close()
    }
  }, [runId])

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs flex flex-col h-[200px]">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Live Run Logs (F6)
          </span>
          {runId && (
            <span className="font-mono text-[10px] bg-slate-800 text-cyan-400 px-1.5 py-0.5 rounded">
              {runId} ({mode || 'autonomous'})
            </span>
          )}
        </div>
        <label className="flex items-center gap-1.5 text-[10px] text-slate-400 cursor-pointer">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="rounded bg-slate-800 border-slate-700 text-cyan-600 focus:ring-0"
          />
          <span>Auto-scroll</span>
        </label>
      </div>

      <div ref={containerRef} className="flex-1 overflow-y-auto space-y-1 font-mono text-[11px] pr-1">
        {logs.length === 0 ? (
          <div className="text-slate-500 italic py-4 text-center">
            {runId ? 'Awaiting harness log entries...' : 'Start a run to stream live logs.'}
          </div>
        ) : (
          logs.map((log) => {
            const timeStr = log.ts ? log.ts.slice(11, 19) : '--:--:--'
            const badgeClass = STAGE_BADGES[log.stage] || 'bg-slate-800 text-slate-400 border-slate-700'
            const levelClass = LEVEL_COLORS[log.level] || 'text-slate-300'

            return (
              <div key={log.seq} className="flex items-start gap-2 hover:bg-slate-800/40 px-1 py-0.5 rounded">
                <span className="text-slate-500 select-none w-14">{timeStr}</span>
                <span className={`px-1.5 py-0.2 rounded text-[9px] uppercase border font-semibold ${badgeClass}`}>
                  {log.stage}
                </span>
                <span className={`flex-1 break-all ${levelClass}`}>{log.message}</span>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
