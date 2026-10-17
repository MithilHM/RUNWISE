import { useMemo, useRef } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import type { ForceGraphMethods } from 'react-force-graph-2d'
import type { GraphEdgeData, GraphNodeData } from '../../api/types'
import { useContainerSize } from '../../lib/useContainerSize'

/**
 * Lineage and impact graph (Screen C).
 *
 * Everything drawn here comes from `GET /projects/{id}/graph`, which reads real
 * nodes and relationships out of FalkorDB. There is no fallback demo graph: a
 * project with no stored lineage renders an empty canvas and says so.
 *
 * When an impact report is supplied, its paths are highlighted, so the picture a
 * researcher sees is the one the traversal actually produced.
 */

const LABEL_COLORS: Record<string, string> = {
  Project: '#4fd1c5',
  Experiment: '#e4e9f0',
  Dataset: '#f0b429',
  DatasetVersion: '#f0b429',
  Configuration: '#a7b1c0',
  Metric: '#5fd18b',
  Claim: '#c084fc',
  Evidence: '#94a3b8',
  Correction: '#f2666e',
  ReviewItem: '#fb923c',
  Recommendation: '#60a5fa',
}

interface LineageGraphProps {
  nodes: GraphNodeData[]
  edges: GraphEdgeData[]
  highlightNodeIds?: string[]
  highlightEdgeKeys?: string[]
  onNodeClick?: (node: GraphNodeData) => void
  focusNodeId?: string | null
}

interface RenderNode extends GraphNodeData {
  x?: number
  y?: number
  highlighted?: boolean
  dimmed?: boolean
}

/** react-force-graph rewrites `source`/`target` into node objects once it resolves links. */
interface RenderLink extends Omit<GraphEdgeData, 'source' | 'target'> {
  source: string | RenderNode
  target: string | RenderNode
  highlighted?: boolean
  dimmed?: boolean
}

export function LineageGraph({
  nodes,
  edges,
  highlightNodeIds = [],
  highlightEdgeKeys = [],
  onNodeClick,
  focusNodeId,
}: LineageGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<ForceGraphMethods<RenderNode, RenderLink> | undefined>(undefined)
  const size = useContainerSize(containerRef)

  const highlighted = useMemo(() => new Set(highlightNodeIds), [highlightNodeIds])
  const highlightedEdges = useMemo(() => new Set(highlightEdgeKeys), [highlightEdgeKeys])
  const hasHighlight = highlighted.size > 0

  const graphData = useMemo(() => {
    const renderNodes: RenderNode[] = nodes.map((node) => ({
      ...node,
      highlighted: highlighted.has(node.id),
      dimmed: hasHighlight && !highlighted.has(node.id),
    }))
    const renderLinks: RenderLink[] = edges.map((edge) => {
      const key = `${edge.source}->${edge.target}:${edge.relationship_type}`
      return {
        ...edge,
        highlighted: highlightedEdges.has(key),
        dimmed: hasHighlight && !highlightedEdges.has(key),
      }
    })
    return { nodes: renderNodes, links: renderLinks }
  }, [nodes, edges, highlighted, highlightedEdges, hasHighlight])

  if (!nodes.length) {
    return (
      <div className="flex h-[560px] items-center justify-center text-sm text-ink-500">
        This project has no stored lineage to draw.
      </div>
    )
  }

  return (
    <div ref={containerRef} className="h-[560px] w-full">
      {size.width > 0 && (
        <ForceGraph2D<RenderNode, RenderLink>
          ref={graphRef}
          width={size.width}
          height={560}
          graphData={graphData}
          backgroundColor="#0a0c10"
          nodeRelSize={4}
          d3VelocityDecay={0.28}
          linkColor={(link) =>
            link.highlighted ? '#f2666e' : link.dimmed ? 'rgba(58,68,83,0.25)' : 'rgba(124,135,152,0.55)'
          }
          linkWidth={(link) => (link.highlighted ? 2 : 1)}
          linkDirectionalArrowLength={(link) => (link.highlighted ? 5 : 3)}
          linkDirectionalArrowRelPos={0.9}
          onNodeClick={(node) => onNodeClick?.(node)}
          onEngineStop={() => {
            if (focusNodeId) {
              const target = graphData.nodes.find((node) => node.id === focusNodeId)
              if (target && typeof target.x === 'number') {
                graphRef.current?.centerAt(target.x, target.y, 400)
              }
            }
          }}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.label
            const x = node.x ?? 0
            const y = node.y ?? 0
            const color = LABEL_COLORS[label] ?? '#a7b1c0'
            const alpha = node.dimmed ? 0.22 : 1

            ctx.globalAlpha = alpha
            ctx.beginPath()
            ctx.arc(x, y, node.highlighted ? 5.5 : 3.5, 0, 2 * Math.PI, false)
            ctx.fillStyle = color
            ctx.fill()
            if (node.highlighted) {
              ctx.strokeStyle = '#f2666e'
              ctx.lineWidth = 1.5 / globalScale
              ctx.stroke()
            }

            const name = String(node.properties?.name ?? node.id)
            const fontSize = Math.max(10 / globalScale, 2.2)
            ctx.font = `${fontSize}px Inter, sans-serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'top'
            ctx.fillStyle = node.dimmed ? 'rgba(167,177,192,0.25)' : '#e4e9f0'
            const shown = name.length > 22 ? `${name.slice(0, 21)}…` : name
            ctx.fillText(shown, x, y + 6)
            ctx.globalAlpha = 1
          }}
          nodePointerAreaPaint={(node, color, ctx) => {
            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(node.x ?? 0, node.y ?? 0, 8, 0, 2 * Math.PI, false)
            ctx.fill()
          }}
        />
      )}
    </div>
  )
}

export function GraphLegend({ labels }: { labels: string[] }) {
  return (
    <div className="flex flex-wrap items-center gap-3 px-4 py-2 text-[11px] text-ink-500">
      {labels.map((label) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: LABEL_COLORS[label] ?? '#a7b1c0' }} />
          {label}
        </span>
      ))}
    </div>
  )
}
