import React, { useMemo } from 'react'
import type { GraphSnapshot } from '../../api/types'


interface LeaderboardProps {
  snapshot: GraphSnapshot
  metricName: string
  onSelectTrial?: (trialId: string) => void
}

export const Leaderboard: React.FC<LeaderboardProps> = ({
  snapshot,
  metricName,
  onSelectTrial,
}) => {
  const isLoss = metricName.toLowerCase().includes('loss')

  // Group trials by model and find best for each
  const rankedModels = useMemo(() => {
    const trials = snapshot.nodes
      .filter((n) => n.type === 'Trial' && n.props?.metric_value !== undefined && n.props?.metric_value !== null)
      .map((n) => ({
        id: n.id,
        model_id: n.props.model_id || 'unknown',
        metric_value: Number(n.props.metric_value),
        step: n.props.step,
        status: n.props.status,
      }))

    const byModel: Record<string, typeof trials[0]> = {}
    for (const t of trials) {
      if (!byModel[t.model_id]) {
        byModel[t.model_id] = t
      } else {
        const currentBest = byModel[t.model_id].metric_value
        const isBetter = isLoss ? t.metric_value < currentBest : t.metric_value > currentBest
        if (isBetter) {
          byModel[t.model_id] = t
        }
      }
    }

    const list = Object.values(byModel)
    list.sort((a, b) => (isLoss ? a.metric_value - b.metric_value : b.metric_value - a.metric_value))
    return list
  }, [snapshot, isLoss])

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Leaderboard (Ranked by {metricName} — {isLoss ? 'Minimizing' : 'Maximizing'})
        </span>
        <span className="text-[10px] text-slate-500 font-mono">
          {rankedModels.length} model(s) evaluated
        </span>
      </div>

      <div className="flex items-center gap-3 overflow-x-auto pb-1">
        {rankedModels.length === 0 ? (
          <span className="text-slate-500 italic text-[11px]">No evaluated trials yet.</span>
        ) : (
          rankedModels.map((item, idx) => (
            <div
              key={item.model_id}
              onClick={() => onSelectTrial && onSelectTrial(item.id)}
              className="flex items-center gap-2 bg-slate-950 border border-slate-800 hover:border-cyan-600/50 px-3 py-1.5 rounded cursor-pointer transition"
            >
              <span className="font-mono text-cyan-400 font-bold">#{idx + 1}</span>
              <span className="font-semibold text-slate-200 uppercase text-[11px]">
                {item.model_id}
              </span>
              <span className="font-mono text-emerald-400 font-bold bg-emerald-950/60 border border-emerald-900 px-1.5 py-0.2 rounded text-[11px]">
                {item.metric_value.toFixed(4)}
              </span>
              <span className="text-[10px] text-slate-500 font-mono">s{item.step}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
