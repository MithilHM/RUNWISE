import React, { useState } from 'react'
import type { Proposal, WhyRecord } from '../../api/types'


interface WhyPanelProps {
  whyRecord: WhyRecord | null
  pendingProposal: Proposal | null
  isAwaitingReview: boolean
  onDecide?: (action: 'approve' | 'reject', overrides?: Record<string, any>, note?: string) => void
  onHighlightChip: (chipId: string) => void
}

export const WhyPanel: React.FC<WhyPanelProps> = ({
  whyRecord,
  pendingProposal,
  isAwaitingReview,
  onDecide,
  onHighlightChip,
}) => {
  const [rejectNote, setRejectNote] = useState('')
  const [showRejectInput, setShowRejectInput] = useState(false)

  const decision = whyRecord?.decision

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs flex flex-col h-full space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Why This Trial (F7)
        </span>
        {decision && (
          <span
            className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase font-semibold ${
              decision.actor === 'researcher'
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                : 'bg-purple-950 text-purple-300 border border-purple-800'
            }`}
          >
            {decision.actor}
          </span>
        )}
      </div>

      {/* Human Review Gate: Pending Proposal */}
      {isAwaitingReview && pendingProposal && (
        <div className="bg-amber-950/40 border border-amber-800/80 rounded p-2.5 space-y-2">
          <div className="flex items-center justify-between text-amber-300 font-semibold text-[11px]">
            <span>⚠️ Awaiting Review (Step {pendingProposal.step})</span>
            <span className="text-[10px] uppercase font-mono">{pendingProposal.model_id}</span>
          </div>
          <div className="text-slate-300 text-[11px] leading-relaxed">
            {pendingProposal.why}
          </div>

          <div className="flex items-center gap-2 pt-1">
            <button
              onClick={() => onDecide && onDecide('approve')}
              className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-1 rounded transition text-xs"
            >
              ✓ Approve
            </button>
            <button
              onClick={() => setShowRejectInput(!showRejectInput)}
              className="bg-rose-900/60 hover:bg-rose-800 text-rose-200 border border-rose-800 font-semibold px-3 py-1 rounded transition text-xs"
            >
              ✕ Reject
            </button>
          </div>

          {showRejectInput && (
            <div className="space-y-1.5 pt-1">
              <input
                type="text"
                placeholder="Reason for rejection (guides next candidate)..."
                value={rejectNote}
                onChange={(e) => setRejectNote(e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs"
              />
              <button
                onClick={() => {
                  if (onDecide) onDecide('reject', undefined, rejectNote)
                  setShowRejectInput(false)
                }}
                className="w-full bg-rose-600 hover:bg-rose-500 text-white font-semibold py-0.5 rounded text-[11px]"
              >
                Confirm Reject
              </button>
            </div>
          )}
        </div>
      )}

      {/* Decision Prose & Hypothesis */}
      {decision ? (
        <div className="space-y-2.5 overflow-y-auto max-h-[220px] pr-1">
          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Strategy</div>
            <div className="text-slate-200 font-medium font-mono text-[11px]">{decision.strategy}</div>
          </div>

          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Justification</div>
            <div className="text-slate-300 leading-relaxed text-[11px]">{decision.why}</div>
          </div>

          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">Hypothesis</div>
            <div className="text-slate-400 italic text-[11px]">"{decision.hypothesis}"</div>
          </div>

          <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1">
            <span>Confidence:</span>
            <span className="font-mono text-cyan-400 font-semibold">
              {(decision.confidence * 100).toFixed(0)}%
            </span>
          </div>

          {/* Evidence Chips */}
          <div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">
              Evidence Chips (Click to highlight node)
            </div>
            <div className="flex flex-wrap gap-1.5">
              {whyRecord?.evidence.map((item) => (
                <button
                  key={item.id}
                  onClick={() => onHighlightChip(item.id)}
                  title={item.detail}
                  className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-cyan-300 hover:text-cyan-200 text-[10px] font-mono px-2 py-0.5 rounded transition inline-flex items-center gap-1"
                >
                  <span className="text-slate-500">[{item.kind}]</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="text-slate-500 italic text-[11px]">
          No trial selected. Click any Trial or Decision node in the dependency graph to inspect its prose justification and citations.
        </div>
      )}
    </div>
  )
}
