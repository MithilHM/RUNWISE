import { useState, useMemo, useEffect } from 'react'
import {
  useExperiments,
  useGraph,
  useComparability,
  useCreateCorrection,
  useRecommendationRequest,
  useRecommendationDecision,
} from '../api/hooks'
import { useProject } from '../lib/projectContext'
import { LineageGraph } from '../features/lineage/LineageGraph'
import type { CorrectionImpactResponse, ExperimentSummary, Recommendation } from '../api/types'
import { formatNumber } from '../lib/format'
import { LoadingState } from '../components/States'

export function ShowcaseHeroPage() {
  const { projectId, project } = useProject()
  const [selectedExpId, setSelectedExpId] = useState<string>('exp_002')
  const [isSimulatedLeak, setIsSimulatedLeak] = useState(false)
  const [impactData, setImpactData] = useState<CorrectionImpactResponse | null>(null)
  const [tourStep, setTourStep] = useState<number | null>(null)
  const [copiedSpec, setCopiedSpec] = useState(false)

  // Fetch real experiments and graph from backend
  const experimentsQuery = useExperiments(projectId, { limit: 50 })
  const graphQuery = useGraph(projectId, 500)
  const createCorrection = useCreateCorrection(projectId)
  const recommendationRequest = useRecommendationRequest(projectId)
  const decideRecommendation = useRecommendationDecision()

  const allExperiments = useExperimentsQueryData(experimentsQuery.data)

  // Filter runs relevant to the core fraud dataset showcase
  const fraudRuns = useMemo(() => {
    return allExperiments.filter(
      (e) =>
        e.dataset_version_id?.includes('tabular_fraud') ||
        e.id === 'exp_001' ||
        e.id === 'exp_002' ||
        e.id === 'exp_003' ||
        e.id === 'exp_009' ||
        e.id === 'exp_control',
    )
  }, [allExperiments])

  const selectedExp = useMemo(
    () => allExperiments.find((e) => e.id === selectedExpId) ?? fraudRuns[0] ?? null,
    [allExperiments, selectedExpId, fraudRuns],
  )

  // Comparability check for selected experiment vs exp_001
  const comparabilityQuery = useComparability(
    selectedExp && selectedExp.id !== 'exp_001' ? selectedExp.id : null,
    { peer: 'exp_001', projectId: projectId ?? undefined },
  )

  // Active recommendation state
  const [activeRec, setActiveRec] = useState<Recommendation | null>(null)
  const [recLoading, setRecLoading] = useState(false)

  // Generate initial recommendation
  const fetchRecommendation = (objectiveText: string, budget = 2.0) => {
    if (!projectId) return
    setRecLoading(true)
    recommendationRequest.mutate(
      { objective: objectiveText, compute_budget_gpu_hours: budget },
      {
        onSuccess: (data) => {
          if (data.recommendations && data.recommendations.length > 0) {
            setActiveRec(data.recommendations[0])
          }
          setRecLoading(false)
        },
        onError: () => {
          setRecLoading(false)
        },
      },
    )
  }

  // Initial recommendation load
  useEffect(() => {
    if (projectId && !activeRec && !recLoading) {
      fetchRecommendation('Maximize fraud detection validation accuracy within 2 GPU hours', 2.0)
    }
  }, [projectId])

  // Toggle Data Leakage Invalidation
  const toggleLeakSimulation = async () => {
    if (!isSimulatedLeak) {
      setIsSimulatedLeak(true)
      try {
        const res = await createCorrection.mutateAsync({
          source_id: 'tabular_fraud_v1',
          source_type: 'dataset_version',
          description: 'Data leakage identified between train and validation splits in tabular_fraud v1',
        })
        setImpactData(res)
        // Refresh recommendation to show adaptive exclusion of tainted runs
        fetchRecommendation('Maximize fraud detection validation accuracy on clean baseline', 2.0)
      } catch (err) {
        console.error('Correction failed:', err)
      }
    } else {
      setIsSimulatedLeak(false)
      setImpactData(null)
      fetchRecommendation('Maximize fraud detection validation accuracy within 2 GPU hours', 2.0)
    }
  }

  // 30-Second Guided Tour
  const runGuidedTour = () => {
    setTourStep(1)
    setSelectedExpId('exp_001')
    setTimeout(() => {
      setTourStep(2)
      setSelectedExpId('exp_002')
    }, 4000)
    setTimeout(() => {
      setTourStep(3)
      if (!isSimulatedLeak) {
        void toggleLeakSimulation()
      }
    }, 9000)
    setTimeout(() => {
      setTourStep(4)
      setSelectedExpId('exp_009')
    }, 15000)
    setTimeout(() => {
      setTourStep(null)
    }, 22000)
  }

  // Calculate highlighted nodes & edges for graph visualization
  const graphHighlights = useMemo(() => {
    if (!isSimulatedLeak || !impactData) {
      return { nodeIds: [selectedExpId], edgeKeys: [] }
    }
    const nodeIds = new Set<string>()
    const edgeKeys = new Set<string>()
    for (const path of impactData.paths) {
      path.nodes.forEach((node, idx) => {
        nodeIds.add(node)
        if (idx > 0) {
          const rel = path.relationship_types[idx - 1]
          edgeKeys.add(`${path.nodes[idx - 1]}->${node}:${rel}`)
          edgeKeys.add(`${path.nodes[idx - 1]}->${node}:AFFECTS`)
          edgeKeys.add(`${path.nodes[idx - 1]}->${node}:USES_DATASET_VERSION`)
        }
      })
    }
    return { nodeIds: [...nodeIds], edgeKeys: [...edgeKeys] }
  }, [isSimulatedLeak, impactData, selectedExpId])

  // Affected IDs check
  const isExpTainted = (id: string) => {
    if (!isSimulatedLeak || !impactData) return false
    return (
      impactData.directly_affected_experiment_ids.includes(id) ||
      impactData.indirectly_affected_experiment_ids.includes(id)
    )
  }

  const handleApprove = () => {
    if (!activeRec) return
    decideRecommendation.mutate({
      id: activeRec.id,
      decision: 'approved',
      decided_by: 'lead_researcher',
      note: 'Approved during hackathon demonstration',
    })
  }

  const handleCopySpec = () => {
    if (!activeRec) return
    const spec = JSON.stringify(
      {
        target_model: activeRec.configuration,
        controlled_variables: activeRec.controlled_variables,
        compute_budget: activeRec.estimated_gpu_hours,
      },
      null,
      2,
    )
    void navigator.clipboard.writeText(spec)
    setCopiedSpec(true)
    setTimeout(() => setCopiedSpec(false), 2000)
  }

  if (experimentsQuery.isLoading || graphQuery.isLoading) {
    return <LoadingState label="Connecting to FalkorDB and initializing live showcase..." />
  }

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-xl border border-signal-500/30 bg-gradient-to-r from-ink-900 via-ink-850 to-ink-900 p-6 shadow-2xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-signal-500/20 px-2.5 py-0.5 text-xs font-semibold tracking-wide text-signal-400">
                FalkorDB Graph Memory Demo
              </span>
              <span className="text-xs text-ink-400">• Project: {project?.name ?? projectId}</span>
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-ink-100 sm:text-3xl">
              Research Decision Copilot &amp; Adaptive Lineage
            </h1>
            <p className="mt-1 max-w-2xl text-xs text-ink-400 sm:text-sm">
              Multiple model versions evaluated on a single dataset. Watch FalkorDB trace multi-hop dependencies,
              isolate tainted data, and dynamically adapt recommendations under budget.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={runGuidedTour}
              className="btn border-ink-600 bg-ink-800 text-xs hover:border-signal-400"
            >
              <span>🎬</span> 30s Guided Tour
            </button>
            <button
              type="button"
              onClick={toggleLeakSimulation}
              className={`btn text-xs font-semibold shadow-lg transition-all ${
                isSimulatedLeak
                  ? 'border-warn-400 bg-warn-400 text-ink-950 hover:bg-warn-400/90'
                  : 'btn-danger'
              }`}
            >
              {isSimulatedLeak ? '↺ Reset Dataset Invalidation' : '🚨 Simulate Data Leakage in v1'}
            </button>
          </div>
        </div>

        {/* Guided Tour Banner */}
        {tourStep && (
          <div className="mt-4 flex items-center justify-between rounded-lg border border-signal-500/40 bg-signal-500/10 px-4 py-2 text-xs text-signal-300 animate-pulse">
            <span>
              {tourStep === 1 && 'Step 1/4: Inspecting baseline model (exp_001) trained on tabular_fraud_v1...'}
              {tourStep === 2 && 'Step 2/4: Examining tuned XGBoost (exp_002) — controlled max_depth ablation...'}
              {tourStep === 3 && 'Step 3/4: Data leakage reported in v1! FalkorDB multi-hop traversal in action...'}
              {tourStep === 4 && 'Step 4/4: Adaptive Copilot switches baseline to clean exp_009 on v2!'}
            </span>
            <button type="button" onClick={() => setTourStep(null)} className="text-ink-400 hover:text-ink-100">
              ✕
            </button>
          </div>
        )}
      </div>

      {/* 3-Column Hero Grid */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Column 1: Dataset & Model Versions (4 cols) */}
        <div className="space-y-4 lg:col-span-4">
          <div className="panel p-4">
            <div className="flex items-center justify-between">
              <span className="label">Target Dataset</span>
              <span
                className={`rounded px-1.5 py-0.5 text-[10px] font-mono font-medium ${
                  isSimulatedLeak ? 'bg-danger-400/20 text-danger-400' : 'bg-good-400/20 text-good-400'
                }`}
              >
                {isSimulatedLeak ? 'v1 TAINTED BY LEAKAGE' : 'v1 & v2 ACTIVE'}
              </span>
            </div>
            <h2 className="mt-1 text-base font-semibold text-ink-100">tabular_fraud (Customer Records)</h2>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-ink-400">
              <span>Metric: <strong>Validation Accuracy</strong></span>
              <span>•</span>
              <span>Split: <strong>Stratified 5-Fold</strong></span>
              <span>•</span>
              <span>Seed: <strong>42</strong></span>
            </div>
          </div>

          <div className="panel space-y-2 p-4">
            <div className="flex items-center justify-between">
              <span className="label">Trained Model Progression</span>
              <span className="text-[11px] text-ink-500">{fraudRuns.length} models mapped</span>
            </div>

            <div className="space-y-2 pt-1">
              {fraudRuns.map((exp) => {
                const tainted = isExpTainted(exp.id)
                const isSelected = exp.id === selectedExpId
                const accMetric = exp.metrics.find((m) => m.name === 'validation_accuracy')
                const accVal = accMetric?.value != null ? `${formatNumber(accMetric.value * 100, 1)}%` : '—'

                return (
                  <div
                    key={exp.id}
                    onClick={() => setSelectedExpId(exp.id)}
                    className={`cursor-pointer rounded-lg border p-3 transition-all ${
                      isSelected
                        ? 'border-signal-500 bg-ink-800 shadow-md ring-1 ring-signal-500/40'
                        : 'border-ink-750 bg-ink-850/60 hover:border-ink-600 hover:bg-ink-800'
                    } ${tainted ? 'border-danger-400/60 bg-danger-400/5' : ''}`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold text-ink-100">{exp.id}</span>
                        {tainted && (
                          <span className="rounded bg-danger-400/20 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-danger-400">
                            TAINTED
                          </span>
                        )}
                        {!tainted && exp.dataset_version_id?.includes('v2') && (
                          <span className="rounded bg-good-400/20 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-good-400">
                            CLEAN (v2)
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-xs font-bold text-signal-400">{accVal}</span>
                    </div>

                    <div className="mt-1 flex items-center justify-between text-xs text-ink-400">
                      <span className="truncate">{exp.name}</span>
                      <span className="font-mono text-[11px] text-ink-500">
                        {exp.compute_cost_gpu_hours ? `${exp.compute_cost_gpu_hours}h` : '—'}
                      </span>
                    </div>

                    <div className="mt-1 flex items-center gap-2 text-[10px] text-ink-500 font-mono">
                      <span>version: {exp.dataset_version_id ?? 'default'}</span>
                      <span>•</span>
                      <span>family: {exp.model_family ?? 'custom'}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Selected Run Inspection Drawer */}
          {selectedExp && (
            <div className="panel p-4 text-xs">
              <div className="flex items-center justify-between border-b border-ink-800 pb-2">
                <span className="label">Inspector: {selectedExp.id}</span>
                {comparabilityQuery.data && (
                  <span
                    className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold ${
                      comparabilityQuery.data.status === 'COMPARABLE'
                        ? 'bg-good-400/20 text-good-400'
                        : comparabilityQuery.data.status === 'COMPARABLE_WITH_CAVEATS'
                        ? 'bg-warn-400/20 text-warn-400'
                        : 'bg-danger-400/20 text-danger-400'
                    }`}
                  >
                    {comparabilityQuery.data.status.replace(/_/g, ' ')}
                  </span>
                )}
              </div>

              <div className="mt-3 space-y-2 text-ink-300">
                <div className="flex justify-between">
                  <span className="text-ink-500">Model Family:</span>
                  <span className="font-medium text-ink-200">{selectedExp.model_family}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-500">Dataset Version:</span>
                  <span className="font-mono text-ink-200">{selectedExp.dataset_version_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-ink-500">Compute Cost:</span>
                  <span className="font-mono text-ink-200">{selectedExp.compute_cost_gpu_hours} GPU-hours</span>
                </div>
              </div>

              {comparabilityQuery.data?.explanation && (
                <div className="mt-3 rounded border border-ink-700 bg-ink-950 p-2 text-[11px] text-ink-400">
                  <span className="font-semibold text-ink-200">Comparability Verdict (vs exp_001): </span>
                  {comparabilityQuery.data.explanation}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Column 2: FalkorDB Graph Lineage Canvas (5 cols) */}
        <div className="space-y-4 lg:col-span-5">
          <div className="panel overflow-hidden">
            <div className="panel-header">
              <div className="flex items-center gap-2">
                <span className="label">FalkorDB Lineage Graph</span>
                <span className="rounded bg-signal-500/10 px-2 py-0.5 font-mono text-[10px] text-signal-400">
                  Live Cypher Lineage
                </span>
              </div>
              <span className="text-[11px] text-ink-500">
                {graphQuery.data?.nodes.length ?? 0} nodes / {graphQuery.data?.edges.length ?? 0} edges
              </span>
            </div>

            <div className="relative h-[480px] w-full bg-ink-950">
              <LineageGraph
                nodes={graphQuery.data?.nodes ?? []}
                edges={graphQuery.data?.edges ?? []}
                highlightNodeIds={graphHighlights.nodeIds}
                highlightEdgeKeys={graphHighlights.edgeKeys}
                focusNodeId={selectedExpId}
                onNodeClick={(node) => {
                  if (node.label === 'Experiment') {
                    setSelectedExpId(node.id)
                  }
                }}
              />

              {/* Real-time Graph Overlay Indicator */}
              <div className="pointer-events-none absolute bottom-3 left-3 rounded-lg border border-ink-700 bg-ink-900/90 p-2.5 text-[10px] backdrop-blur-sm">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-good-400"></span>
                  <span className="text-ink-300">Clean / Verified Evidence</span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-danger-400"></span>
                  <span className="text-ink-300">Tainted by Upstream Leak</span>
                </div>
                <div className="mt-1 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-signal-400"></span>
                  <span className="text-ink-300">Selected Run Ancestry</span>
                </div>
              </div>
            </div>

            <div className="border-t border-ink-800 bg-ink-900 p-3 text-xs text-ink-400">
              {isSimulatedLeak ? (
                <div className="flex items-center gap-2 text-danger-400">
                  <span>⚠️</span>
                  <span>
                    FalkorDB traversed multi-hop lineage: <strong>(tabular_fraud_v1) ➔ (exp_001) ➔ (exp_002) ➔ (exp_003)</strong>.
                    All 3 runs marked disputed.
                  </span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-ink-400">
                  <span>ℹ️</span>
                  <span>Click any experiment node to inspect its lineage and hyperparameter ablation path.</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Column 3: Copilot Recommendation & Trade-Off Matrix (3 cols) */}
        <div className="space-y-4 lg:col-span-3">
          {/* Accuracy vs Compute Efficiency Matrix */}
          <div className="panel p-4">
            <span className="label">Model Efficiency Matrix</span>
            <p className="mt-1 text-xs text-ink-400">Validation Accuracy vs. Compute Consumed</p>

            <div className="mt-3 space-y-3">
              <div>
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-ink-200">XGBoost (exp_002)</span>
                  <span className="font-bold text-signal-400">93.8% (1.2h)</span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-ink-800">
                  <div className="h-2 rounded-full bg-signal-400" style={{ width: '93.8%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-ink-200">Distilled Tree (exp_003)</span>
                  <span className="font-bold text-good-400">92.5% (0.5h)</span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-ink-800">
                  <div className="h-2 rounded-full bg-good-400" style={{ width: '92.5%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-ink-200">Baseline (exp_001)</span>
                  <span className="font-bold text-ink-400">91.2% (0.8h)</span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-ink-800">
                  <div className="h-2 rounded-full bg-ink-400" style={{ width: '91.2%' }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs">
                  <span className="font-semibold text-ink-200">Linear (exp_010)</span>
                  <span className="font-bold text-danger-400">74.0% (0.2h)</span>
                </div>
                <div className="mt-1 h-2 w-full rounded-full bg-ink-800">
                  <div className="h-2 rounded-full bg-danger-400" style={{ width: '74.0%' }}></div>
                </div>
              </div>
            </div>

            <div className="mt-3 rounded border border-ink-800 bg-ink-950 p-2 text-[11px] text-ink-400">
              💡 <strong>Key Finding:</strong> Tree-based architectures beat linear models by <strong>+19.8%</strong> with high compute efficiency.
            </div>
          </div>

          {/* Adaptive Copilot Recommendation */}
          <div className="panel p-4">
            <div className="flex items-center justify-between">
              <span className="label">RUNWISE Copilot Verdict</span>
              {recLoading && <span className="text-[10px] text-signal-400 animate-pulse">Computing...</span>}
            </div>

            <div className="mt-3 space-y-3">
              {isSimulatedLeak ? (
                <div className="rounded-lg border border-warn-400/40 bg-warn-400/10 p-2.5 text-xs text-warn-300">
                  <strong>Adaptive Recalibration:</strong> Tainted baselines (exp_001, exp_002) were automatically purged.
                  Copilot pivoted to build upon verified baseline <strong>exp_009 (v2)</strong>!
                </div>
              ) : null}

              <div className="rounded-lg border border-ink-700 bg-ink-950 p-3">
                <span className="text-[10px] font-semibold uppercase text-signal-400">Recommended Next Move</span>
                <p className="mt-1 text-xs font-medium text-ink-100">
                  {activeRec?.hypothesis ?? 'Controlled hyperparameter tuning on gradient boosted trees within budget.'}
                </p>

                <div className="mt-2 space-y-1 text-[11px] text-ink-400">
                  <div className="flex justify-between">
                    <span>Target Budget:</span>
                    <span className="font-mono text-ink-200">
                      {activeRec?.estimated_gpu_hours ? `${activeRec.estimated_gpu_hours} GPU-hours` : '1.4 GPU-hours'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Baseline Run:</span>
                    <span className="font-mono text-signal-400">
                      {activeRec?.baseline_experiment_ids?.join(', ') || (isSimulatedLeak ? 'exp_009' : 'exp_001')}
                    </span>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleApprove}
                  className="btn btn-primary w-full text-xs"
                >
                  ✓ Approve Experiment Proposal
                </button>
                <button
                  type="button"
                  onClick={handleCopySpec}
                  className="btn w-full text-xs border-ink-700"
                >
                  {copiedSpec ? '✓ Copied Config to Clipboard' : '📋 Export Training Spec'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function useExperimentsQueryData(data: ExperimentSummary[] | undefined): ExperimentSummary[] {
  return useMemo(() => data ?? [], [data])
}
