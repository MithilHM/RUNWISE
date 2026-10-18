import { useCallback, useMemo, useRef, useState } from 'react'
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
 *
 * Labelling is gated on zoom and on node significance. A project graph runs to
 * hundreds of nodes; drawing every caption at once produces an unreadable mat of
 * overlapping text, so structural nodes stay named, everything else reveals its
 * name as the researcher closes in, and hover always reveals it.
 */

const LABEL_COLORS: Record<string, string> = {
  Project: '#4fd1c5',
  Experiment: '#e4e9f0',
  Dataset: '#f0b429',
  DatasetVersion: '#f0b429',
  Configuration: '#a7b1c0',
  Metric: '#5fd18b',
  Claim: '#c4a2f7',
  Evidence: '#94a3b8',
  Correction: '#f2666e',
  ReviewItem: '#fb923c',
  Recommendation: '#93a4f7',
}

/** Node types that name the shape of the project, so they stay captioned. */
const STRUCTURAL = new Set(['Project', 'Correction'])

const NODE_RADIUS: Record<string, number> = {
  Project: 7,
  DatasetVersion: 5,
  Dataset: 5,
  Correction: 5,
  ReviewItem: 4.5,
  Experiment: 4,
  Claim: 4,
  Configuration: 3,
  Metric: 3,
  Evidence: 3,
}

const HEIGHT = 620
const LABEL_ZOOM_THRESHOLD = 1.2

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
  const [hovered, setHovered] = useState<RenderNode | null>(null)

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

  const paintNode = useCallback(
    (node: RenderNode, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const x = node.x ?? 0
      const y = node.y ?? 0
      const color = LABEL_COLORS[node.label] ?? '#a7b1c0'
      const alpha = node.dimmed ? 0.25 : 1
      const base = NODE_RADIUS[node.label] ?? 3.5
      const radius = node.highlighted ? base + 2 : base

      ctx.globalAlpha = alpha
      ctx.beginPath()
      ctx.arc(x, y, radius, 0, 2 * Math.PI, false)
      ctx.fillStyle = color
      ctx.fill()

      if (node.highlighted) {
        ctx.strokeStyle = '#f2666e'
        ctx.lineWidth = 1.6 / globalScale
        ctx.stroke()
      }

      const isHovered = hovered?.id === node.id
      const forced = node.highlighted || isHovered
      const shouldLabel =
        !node.dimmed &&
        (forced || STRUCTURAL.has(node.label) || globalScale >= LABEL_ZOOM_THRESHOLD)

      if (!shouldLabel) {
        ctx.globalAlpha = 1
        return
      }

      const name = String(node.properties?.name ?? node.id)
      const fontSize = Math.max(11 / globalScale, 2.4)
      ctx.font = `${fontSize}px Inter, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'

      /* A dark plate behind the caption keeps it readable where the graph is
         dense, and the caption sits below the node so it never covers it. */
      const shown = name.length > 20 ? `${name.slice(0, 19)}…` : name
      const metrics = ctx.measureText(shown)
      ctx.fillStyle = 'rgba(10, 12, 16, 0.72)'
      ctx.fillRect(x - metrics.width / 2 - 2, y + radius + 1, metrics.width + 4, fontSize + 2)
      ctx.fillStyle = forced ? '#ffffff' : '#ccd4e0'
      ctx.fillText(shown, x, y + radius + 2)
      ctx.globalAlpha = 1
    },
    [hovered],
  )

  if (!nodes.length) {
    return (
      <div className="flex items-center justify-center text-sm text-ink-500" style={{ height: HEIGHT }}>
        This project has no stored lineage to draw.
      </div>
    )
  }

  return (
    <div>
      <div ref={containerRef} className="w-full" style={{ height: HEIGHT }}>
        {size.width > 0 && (
          <ForceGraph2D<RenderNode, RenderLink>
            ref={graphRef}
            width={size.width}
            height={HEIGHT}
            graphData={graphData}
            backgroundColor="#0a0c10"
            /* Spreading the layout matters more than node size: 200 nodes in a
               dense cluster is what makes this unreadable, not their radius. */
            d3VelocityDecay={0.32}
            d3AlphaDecay={0.03}
            warmupTicks={120}
            cooldownTicks={260}
            linkColor={(link) =>
              link.highlighted ? '#f2666e' : link.dimmed ? 'rgba(58,68,83,0.2)' : 'rgba(124,135,152,0.4)'
            }
            linkWidth={(link) => (link.highlighted ? 2 : 1)}
            linkDirectionalArrowLength={(link) => (link.highlighted ? 5 : 2.5)}
            linkDirectionalArrowRelPos={0.92}
            linkDirectionalArrowColor={(link) =>
              link.highlighted ? '#f2666e' : 'rgba(124,135,152,0.35)'
            }
            nodeCanvasObject={paintNode}
            nodePointerAreaPaint={(node, color, ctx) => {
              ctx.fillStyle = color
              ctx.beginPath()
              ctx.arc(node.x ?? 0, node.y ?? 0, 9, 0, 2 * Math.PI, false)
              ctx.fill()
            }}
            onNodeHover={(node) => setHovered(node ?? null)}
            onNodeClick={(node) => onNodeClick?.(node)}
            onEngineStop={() => {
              /* Only recentre on an explicit focus. Fitting the whole graph to the
                 canvas zooms so far out that the structural names fall below the
                 readable threshold and every caption disappears. */
              if (focusNodeId) {
                const target = graphData.nodes.find((node) => node.id === focusNodeId)
                if (target && typeof target.x === 'number') {
                  graphRef.current?.centerAt(target.x, target.y, 400)
                }
              }
            }}
          />
        )}
      </div>

      <div className="border-t border-ink-800 px-4 py-2.5">
        <p className="text-[11px] text-ink-500">
          {hovered ? (
            <span className="text-ink-200">
              <span style={{ color: LABEL_COLORS[hovered.label] ?? '#a7b1c0' }}>●</span>{' '}
              {hovered.label} · {String(hovered.properties?.name ?? hovered.id)}
            </span>
          ) : (
            'Scroll to zoom in and reveal every run name. The project and any correction stay named.'
          )}
        </p>
      </div>
    </div>
  )
}

export function GraphLegend({ labels }: { labels: string[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-ink-500">
      {labels.map((label) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: LABEL_COLORS[label] ?? '#a7b1c0' }} />
          {label}
        </span>
      ))}
    </div>
  )
}