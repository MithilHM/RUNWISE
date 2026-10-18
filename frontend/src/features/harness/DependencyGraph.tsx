import React, { useMemo, useRef, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { GraphNode, GraphSnapshot } from '../../api/types'


interface DependencyGraphProps {
  snapshot: GraphSnapshot
  highlight: string[]
  onNodeClick: (node: GraphNode) => void
  bestMetric?: number | null
}

const LAYER_X: Record<string, number> = {
  Dataset: 80,
  Run: 220,
  Model: 380,
  Param: 540,
  Setting: 680,
  Trial: 820,
  Decision: 960,
}

const TYPE_COLORS: Record<string, string> = {
  Dataset: '#38bdf8', // sky-400
  Run: '#818cf8',     // indigo-400
  Model: '#c084fc',   // purple-400
  Param: '#f472b6',   // pink-400
  Setting: '#fbbf24', // amber-400
  Trial: '#34d399',   // emerald-400
  Decision: '#a78bfa',// violet-400
}

export const DependencyGraph: React.FC<DependencyGraphProps> = ({
  snapshot,
  highlight,
  onNodeClick,
  bestMetric,
}) => {
  const fgRef = useRef<any>(null)

  const graphData = useMemo(() => {
    const nodes = snapshot.nodes.map((n, idx) => {
      const targetX = LAYER_X[n.type] ?? 500
      return {
        ...n,
        id: n.id,
        name: n.label || n.id,
        val: n.type === 'Trial' ? 8 : 6,
        x: targetX,
        y: 100 + (idx % 12) * 45,
      }
    })

    const edges = snapshot.edges.map((e) => ({
      source: e.source,
      target: e.target,
      type: e.type,
    }))

    return { nodes, links: edges }
  }, [snapshot])

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-120)
    }
  }, [graphData])

  const highlightSet = useMemo(() => new Set(highlight), [highlight])

  return (
    <div className="relative w-full h-[460px] bg-slate-950 border border-slate-800 rounded-lg overflow-hidden">
      <div className="absolute top-2 left-3 z-10 flex items-center gap-2 text-[10px] text-slate-400 bg-slate-900/80 backdrop-blur px-2 py-1 rounded border border-slate-800 pointer-events-none">
        <span className="font-semibold text-slate-300">Layers (L→R):</span>
        <span>Dataset → Run → Model → Setting → Trial → Decision</span>
      </div>

      {highlight.length > 0 && (
        <div className="absolute top-2 right-3 z-10 text-[10px] bg-cyan-950/80 border border-cyan-800 text-cyan-300 px-2 py-1 rounded">
          Highlight: {highlight.length} active node(s)
        </div>
      )}

      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeLabel={(node: any) => `${node.type}: ${node.name}`}
        nodeColor={(node: any) => {
          const isHighlighted = highlightSet.size === 0 || highlightSet.has(node.id)
          const baseColor = TYPE_COLORS[node.type] || '#94a3b8'
          return isHighlighted ? baseColor : '#334155'
        }}
        nodeCanvasObject={(node: any, ctx, globalScale) => {
          const isHighlighted = highlightSet.size === 0 || highlightSet.has(node.id)
          const radius = node.type === 'Trial' ? 7 : 5
          const alpha = isHighlighted ? 1.0 : 0.2

          ctx.save()
          ctx.globalAlpha = alpha

          // Circle
          ctx.beginPath()
          ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false)
          ctx.fillStyle = TYPE_COLORS[node.type] || '#94a3b8'
          if (node.type === 'Trial' && node.props?.metric_value && bestMetric) {
            // Shade relative to best
            const ratio = Math.min(node.props.metric_value / bestMetric, 1.0)
            ctx.fillStyle = ratio > 0.95 ? '#10b981' : '#f59e0b'
          }
          ctx.fill()

          if (isHighlighted && highlightSet.size > 0) {
            ctx.lineWidth = 2
            ctx.strokeStyle = '#38bdf8'
            ctx.stroke()
          }

          // Label
          if (globalScale > 0.7 || isHighlighted) {
            const label = node.name.length > 14 ? node.name.slice(0, 12) + '..' : node.name
            ctx.font = `${10 / globalScale}px Sans-Serif`
            ctx.fillStyle = isHighlighted ? '#f8fafc' : '#64748b'
            ctx.fillText(label, node.x + radius + 3, node.y + 3)
          }

          ctx.restore()
        }}
        linkColor={() => '#334155'}
        linkWidth={1.2}
        linkDirectionalArrowLength={3.5}
        linkDirectionalArrowRelPos={0.9}
        onNodeClick={(node: any) => {
          onNodeClick({
            id: node.id,
            label: node.name,
            type: node.type,
            props: node.props || {},
          })
        }}
        width={680}
        height={460}
      />
    </div>
  )
}
