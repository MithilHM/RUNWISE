import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useGraph, useGraphSummary } from '../api/hooks'
import type { CorrectionImpactResponse, GraphNodeData } from '../api/types'
import { IdChip } from '../components/Chips'
import { useProject } from '../lib/projectContext'
import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { GraphLegend, LineageGraph } from '../features/lineage/LineageGraph'
import { ImpactPanel } from '../features/lineage/ImpactPanel'
import { ReviewQueue } from '../features/lineage/ReviewQueue'

/**
 * Screen C: lineage and impact analysis.
 *
 * The graph is the stored project graph, and the impact report highlights the nodes
 * and relationships a real traversal returned. Ask for a correction and the same
 * graph highlights what that correction reached, with the unaffected runs left
 * visible as still-usable evidence.
 */
export function LineagePage() {
  const { projectId, project } = useProject()
  const [searchParams, setSearchParams] = useSearchParams()
  const focusedId = searchParams.get('experiment')
  const [impact, setImpact] = useState<CorrectionImpactResponse | null>(null)
  const [selected, setSelected] = useState<GraphNodeData | null>(null)

  const graph = useGraph(projectId, 800)
  const summary = useGraphSummary(projectId)

  const nodes = graph.data?.nodes ?? []
  const edges = graph.data?.edges ?? []

  const highlight = useMemo(() => {
    if (!impact) return { nodeIds: [] as string[], edgeKeys: [] as string[] }
    const nodeIds = new Set<string>()
    const edgeKeys = new Set<string>()
    for (const path of impact.paths) {
      path.nodes.forEach((node, index) => {
        nodeIds.add(node)
        if (index > 0) {
          const rel = path.relationship_types[index - 1]
          edgeKeys.add(`${path.nodes[index - 1]}->${node}:${rel}`)
          // The correction node is linked to the corrected source by AFFECTS or
          // CHALLENGES; match it whichever verb the traversal used.
          edgeKeys.add(`${path.nodes[index - 1]}->${node}:AFFECTS`)
          edgeKeys.add(`${path.nodes[index - 1]}->${node}:CHALLENGES`)
        }
      })
    }
    return { nodeIds: [...nodeIds], edgeKeys: [...edgeKeys] }
  }, [impact])

  if (!projectId) {
    return <EmptyState title="No project selected" hint="Pick a project to inspect its lineage and impact." />
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-medium text-ink-100">Lineage &amp; impact</h1>
        <p className="text-xs text-ink-500">
          {project?.name ?? projectId} · {nodes.length} node(s), {edges.length} relationship(s) read from FalkorDB
          {graph.data?.truncated && ` · truncated at the ${graph.data.node_limit} node budget, so this view is partial`}
        </p>
      </div>

      {summary.data && (
        <div className="panel flex flex-wrap gap-x-6 gap-y-1 px-4 py-2 text-[11px]">
          {Object.entries(summary.data.node_counts).map(([label, count]) => (
            <span key={label} className="text-ink-400">
              <span className="text-ink-100">{count}</span> {label.toLowerCase()}
            </span>
          ))}
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div className="space-y-4">
          <div className="panel overflow-hidden">
            <div className="panel-header">
              <h2 className="text-sm font-medium text-ink-100">Dependency graph</h2>
              {focusedId && (
                <button type="button" className="btn" onClick={() => setSearchParams({})}>
                  Clear focus
                </button>
              )}
            </div>
            {graph.isLoading ? (
              <LoadingState label="Loading graph" />
            ) : graph.isError ? (
              <div className="p-3">
                <ErrorState error={graph.error} onRetry={() => void graph.refetch()} context="Could not load the graph" />
              </div>
            ) : (
              <>
                <GraphLegend labels={[...new Set(nodes.map((node) => node.label))]} />
                <LineageGraph
                  nodes={nodes}
                  edges={edges}
                  highlightNodeIds={highlight.nodeIds}
                  highlightEdgeKeys={highlight.edgeKeys}
                  focusNodeId={focusedId}
                  onNodeClick={(node) => {
                    setSelected(node)
                    if (node.label === 'Experiment') setSearchParams({ experiment: node.id })
                  }}
                />
              </>
            )}
          </div>

          {selected && (
            <div className="panel px-4 py-3">
              <h3 className="text-sm font-medium text-ink-100">
                <IdChip id={selected.id} /> <span className="text-ink-400">({selected.label})</span>
              </h3>
              <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs md:grid-cols-3">
                {Object.entries(selected.properties)
                  .filter(([, value]) => value !== null && value !== '')
                  .slice(0, 18)
                  .map(([key, value]) => (
                    <div key={key} className="flex justify-between gap-2">
                      <dt className="text-ink-500">{key}</dt>
                      <dd className="truncate font-mono text-ink-200" title={String(value)}>
                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                      </dd>
                    </div>
                  ))}
              </dl>
            </div>
          )}

          <ReviewQueue projectId={projectId} />
        </div>

        <div className="xl:sticky xl:top-4 xl:self-start">
          <ImpactPanel projectId={projectId} impact={impact} onImpact={setImpact} />
        </div>
      </div>
    </div>
  )
}
