import { useCallback, useEffect, useMemo, useRef } from 'react'
import { EmptyState } from '../../components/States'
import { formatDuration, formatNumber, formatTimestamp } from '../../lib/format'
import {
  DECISION_META,
  RUNNING_NODE,
  TRIAL_STATUS_META,
  isMinimized,
} from './runtimeConstants'
import { formatHyperparameter, settledCount, type ExecutionTreeNode, type RuntimeState } from './runtimeModel'

/**
 * Live execution tree.
 *
 * Every edge drawn here is an edge the backend actually wrote, which constrains the
 * layout more than a generic graph would:
 *
 *  - `EVALUATED_ON` joins every trial to the one dataset, so it is routed as a bus
 *    down a dedicated right-hand gutter and stubbed in per row. Drawing twenty
 *    separate curves from the dataset to the trials would cross the whole canvas and
 *    bury the `DERIVED_FROM` structure that matters.
 *  - `DERIVED_FROM` joins a trial to the hypothesis of the *same* step, so the tree is
 *    a column of two-node pairs rather than a deep chain. Cross-step lineage is drawn
 *    only when the server states a parent trial; nothing is inferred to fill the gap.
 *
 * Nodes are HTML buttons positioned over an SVG edge layer, not SVG shapes. That keeps
 * them focusable, selectable by keyboard and readable by a screen reader, which a
 * hand-painted canvas would not be.
 */

/**
 * Column geometry. The total canvas width is deliberately kept under the widest panel
 * the page grid hands this view, so the dataset node and the `EVALUATED_ON` trunk stay
 * on screen. An edge the reader has to scroll sideways to find is not communicating the
 * persistence this panel exists to show.
 */
const NODE_W = { hypothesis: 192, trial: 214, dataset: 156 }
const COL_X = { hypothesis: 0, trial: 298, dataset: 556 }
const NODE_H = { hypothesis: 54, trial: 58, failure: 26, dataset: 44 }
const ROW_PITCH = 104
const TOP_PAD = 84
const FAILURE_OFFSET = 64
const BUS_X = COL_X.dataset + NODE_W.dataset + 20
const CANVAS_W = BUS_X + 20

interface LiveExecutionTreeViewProps {
  state: RuntimeState
  selectedStep: number | null
  onSelect: (step: number | null) => void
}

export function LiveExecutionTreeView({ state, selectedStep, onSelect }: LiveExecutionTreeViewProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const previousCount = useRef(state.nodes.length)
  const previousSelection = useRef<number | null>(selectedStep)

  // Follow the run as it grows. Only when the operator has not scrolled away from the
  // tail, so inspecting an old trial is not yanked out from under them by the next event.
  useEffect(() => {
    if (state.nodes.length === previousCount.current) return
    previousCount.current = state.nodes.length
    const element = scrollRef.current
    if (!element) return
    const atTail = element.scrollHeight - element.scrollTop - element.clientHeight < 80
    if (atTail) element.scrollTop = element.scrollHeight
  }, [state.nodes.length])

  // Selecting a trial from the reasoning feed has to bring its node into view *and*
  // put focus on it: the feed promises "focus trial N", and a scroll alone leaves a
  // keyboard or screen-reader user with the viewport moved and focus left behind.
  useEffect(() => {
    if (selectedStep === previousSelection.current) return
    previousSelection.current = selectedStep
    if (selectedStep === null) return
    const index = state.nodes.findIndex((node) => node.step === selectedStep)
    const element = scrollRef.current
    if (index < 0 || !element) return
    const offset = TOP_PAD + index * ROW_PITCH - element.clientHeight / 2
    element.scrollTo({ top: Math.max(0, offset), behavior: 'smooth' })
    // Scrolling the container moves the viewport only; the node itself must take focus.
    nodeRefs.current.get(selectedStep)?.focus({ preventScroll: true })
  }, [selectedStep, state.nodes])

  const selected = useMemo(
    () => state.nodes.find((node) => node.step === selectedStep) ?? null,
    [state.nodes, selectedStep],
  )

  const childrenOf = useMemo(() => {
    const map = new Map<string, ExecutionTreeNode[]>()
    for (const node of state.nodes) {
      if (!node.parentTrialId) continue
      const list = map.get(node.parentTrialId)
      if (list) list.push(node)
      else map.set(node.parentTrialId, [node])
    }
    return map
  }, [state.nodes])

  const rowIndexByTrialId = useMemo(() => {
    const map = new Map<string, number>()
    state.nodes.forEach((node, index) => {
      if (node.trialId) map.set(node.trialId, index)
    })
    return map
  }, [state.nodes])

  /** Node buttons, so selecting from the feed can move focus, not just the viewport. */
  const nodeRefs = useRef(new Map<number, HTMLButtonElement>())

  const registerNode = useCallback((step: number, element: HTMLButtonElement | null) => {
    if (element) nodeRefs.current.set(step, element)
    else nodeRefs.current.delete(step)
  }, [])

  const canvasHeight = TOP_PAD + state.nodes.length * ROW_PITCH + 24

  if (state.nodes.length === 0) {
    return (
      <div className="panel flex min-h-[420px] items-center justify-center">
        <EmptyState
          title="No trials executed yet"
          hint="Set an objective and start the runtime. Each trial appears here the moment Laya commits to it, before the worker has returned."
        />
      </div>
    )
  }

  return (
    <div className="panel overflow-hidden">
      <div className="panel-header">
        <div>
          <h2 className="text-sm font-medium text-ink-100">Execution tree</h2>
          <p className="mt-0.5 text-[11px] text-ink-500">
            {state.nodes.length} node(s) · {settledCount(state.nodes)} settled · edges are the relationships
            written to FalkorDB
          </p>
        </div>
        <TreeLegend />
      </div>

      <div ref={scrollRef} className="max-h-[560px] overflow-auto">
        <div className="relative" style={{ width: CANVAS_W, height: canvasHeight }}>
          <svg
            className="absolute inset-0"
            width={CANVAS_W}
            height={canvasHeight}
            aria-hidden="true"
            focusable="false"
          >
            <defs>
              <marker id="edge-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--color-ink-600)" />
              </marker>
              <marker
                id="edge-arrow-hi"
                viewBox="0 0 8 8"
                refX="7"
                refY="4"
                markerWidth="6"
                markerHeight="6"
                orient="auto"
              >
                <path d="M0,0 L8,4 L0,8 z" fill="var(--color-signal-500)" />
              </marker>
              <marker id="edge-arrow-danger" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L8,4 L0,8 z" fill="var(--color-danger-400)" />
              </marker>
            </defs>

            {/* EVALUATED_ON bus: one trunk, stubbed per trial. Both coordinates are given
                explicitly — an SVG line missing `x2` defaults it to 0, which draws
                a diagonal straight across the canvas. */}
            {state.nodes.length > 0 && (
              <line
                x1={BUS_X}
                y1={54}
                x2={BUS_X}
                y2={TOP_PAD + (state.nodes.length - 1) * ROW_PITCH + NODE_H.trial / 2}
                stroke="var(--color-ink-700)"
                strokeWidth={1}
              />
            )}

            {state.nodes.map((node, index) => {
              const rowY = TOP_PAD + index * ROW_PITCH
              const trialCenterY = rowY + NODE_H.trial / 2
              const isSelected = node.step === selectedStep
              const edgeColor = isSelected ? 'var(--color-signal-500)' : 'var(--color-ink-600)'
              const marker = isSelected ? 'edge-arrow-hi' : 'edge-arrow'

              return (
                <g key={node.step}>
                  {/* DERIVED_FROM: hypothesis of the same step -> its trial. */}
                  <line
                    x1={COL_X.hypothesis + NODE_W.hypothesis}
                    y1={rowY + NODE_H.hypothesis / 2}
                    x2={COL_X.trial - 6}
                    y2={trialCenterY}
                    stroke={edgeColor}
                    strokeWidth={isSelected ? 1.6 : 1}
                    markerEnd={`url(#${marker})`}
                  />
                  <text
                    x={(COL_X.hypothesis + NODE_W.hypothesis + COL_X.trial) / 2}
                    y={rowY + NODE_H.hypothesis / 2 - 7}
                    textAnchor="middle"
                    className="fill-ink-500 font-mono"
                    style={{ fontSize: 9, letterSpacing: '0.04em' }}
                  >
                    DERIVED_FROM
                  </text>

                  {/* EVALUATED_ON stub into the shared trunk. */}
                  <line
                    x1={COL_X.trial + NODE_W.trial}
                    y1={trialCenterY}
                    x2={BUS_X - 6}
                    y2={trialCenterY}
                    stroke={isSelected ? 'var(--color-signal-500)' : 'var(--color-ink-700)'}
                    strokeWidth={1}
                    markerEnd={`url(#${isSelected ? 'edge-arrow-hi' : 'edge-arrow'})`}
                  />

                  {/* FAILED_UNDER, only when the worker actually failed. */}
                  {node.status === 'FAILED' && node.failureReason && (
                    <>
                      <line
                        x1={COL_X.trial + NODE_W.trial / 2}
                        y1={rowY + NODE_H.trial}
                        x2={COL_X.trial + NODE_W.trial / 2}
                        y2={rowY + FAILURE_OFFSET - 4}
                        stroke="var(--color-danger-400)"
                        strokeWidth={1}
                        markerEnd="url(#edge-arrow-danger)"
                      />
                      <text
                        x={COL_X.trial + NODE_W.trial / 2 + 8}
                        y={rowY + FAILURE_OFFSET - 1}
                        className="fill-danger-400 font-mono"
                        style={{ fontSize: 9, letterSpacing: '0.04em' }}
                      >
                        FAILED_UNDER
                      </text>
                    </>
                  )}

                  /* Cross-step lineage. Drawn only when the server states a parent trial, and routed
                 to the row that parent actually occupies — assuming it is the row
                 above would draw a confident line to the wrong trial. */
                  {node.parentTrialId &&
                    index > 0 &&
                    (() => {
                      const parentIndex = rowIndexByTrialId.get(node.parentTrialId)
                      if (parentIndex === undefined || parentIndex >= index) return null
                      const fromY = TOP_PAD + parentIndex * ROW_PITCH + NODE_H.trial
                      return (
                        <line
                          x1={COL_X.trial + NODE_W.trial / 2}
                          y1={fromY}
                          x2={COL_X.trial + NODE_W.trial / 2}
                          y2={rowY}
                          stroke="var(--color-signal-500)"
                          strokeWidth={1}
                          strokeDasharray="3 3"
                          opacity={isSelected ? 0.9 : 0.4}
                          markerEnd="url(#edge-arrow-hi)"
                        />
                      )
                    })()}
                </g>
              )
            })}

            <text
              x={BUS_X - 8}
              y={40}
              textAnchor="end"
              className="fill-ink-500 font-mono"
              style={{ fontSize: 9, letterSpacing: '0.04em' }}
            >
              EVALUATED_ON
            </text>
          </svg>

          <DatasetNode x={COL_X.dataset} y={10} dataset={state.objective.dataset_name} family={state.objective.model_family} />

          {state.nodes.map((node, index) => {
            const rowY = TOP_PAD + index * ROW_PITCH
            return (
              <div key={node.step}>
                <HypothesisNode node={node} x={COL_X.hypothesis} y={rowY} selected={node.step === selectedStep} />
                <TrialNode
                  node={node}
                  metric={state.objective.target_metric}
                  x={COL_X.trial}
                  y={rowY}
                  selected={node.step === selectedStep}
                  registerRef={(element) => registerNode(node.step, element)}
                  onSelect={() => onSelect(node.step === selectedStep ? null : node.step)}
                />
                {node.status === 'FAILED' && node.failureReason && (
                  <FailureNode node={node} x={COL_X.trial} y={rowY + FAILURE_OFFSET} />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {selected && (
        <NodeDetail
          node={selected}
          metric={state.objective.target_metric}
          derivedTrials={childrenOf.get(selected.trialId ?? '') ?? []}
          onClose={() => onSelect(null)}
        />
      )}
    </div>
  )
}

function TreeLegend() {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-ink-500">
      <span className="flex items-center gap-1">
        <span className="h-2 w-2 animate-pulse rounded-full bg-warn-400" /> running
      </span>
      <span className="flex items-center gap-1">
        <span className="h-2 w-2 rounded-full bg-good-400" /> succeeded
      </span>
      <span className="flex items-center gap-1">
        <span className="h-2 w-2 rounded-full bg-danger-400" /> failed
      </span>
      <span className="flex items-center gap-1">
        <span className="h-2 w-2 rounded-full bg-danger-400/60" /> pruned
      </span>
    </div>
  )
}

function DatasetNode({ x, y, dataset, family }: { x: number; y: number; dataset: string; family: string }) {
  return (
    <div
      className="absolute rounded-md border border-ink-600 bg-ink-850 px-3 py-2"
      style={{ left: x, top: y, width: NODE_W.dataset, height: NODE_H.dataset }}
    >
      <p className="label">Dataset node</p>
      <p className="truncate font-mono text-[12px] text-ink-100" title={dataset}>
        {dataset}
      </p>
      <p className="text-[10px] text-ink-500">{family}</p>
    </div>
  )
}

function HypothesisNode({ node, x, y, selected }: { node: ExecutionTreeNode; x: number; y: number; selected: boolean }) {
  const meta = node.decisionType ? DECISION_META[node.decisionType] : null
  return (
    <div
      className={`absolute rounded-md border px-3 py-1.5 ${
        selected ? 'border-signal-500 bg-signal-500/10' : 'border-ink-700 bg-ink-850'
      }`}
      style={{ left: x, top: y, width: NODE_W.hypothesis, height: NODE_H.hypothesis }}
      title={`hyp_${node.candidateId}`}
    >
      <div className="flex items-center gap-1.5">
        <span
          className={`inline-flex h-4 w-5 items-center justify-center rounded font-mono text-[9px] font-semibold ${
            node.decisionType === 'avoid_failure_subgraph'
              ? 'bg-danger-400/15 text-danger-400'
              : node.decisionType === 'explore_boundary'
                ? 'bg-cat-3/15 text-cat-3'
                : 'bg-good-400/15 text-good-400'
          }`}
        >
          {meta?.token ?? '··'}
        </span>
        <span className="font-mono text-[10px] text-ink-500">step {node.step}</span>
        {node.confidence !== null && (
          <span className="ml-auto font-mono text-[10px] text-ink-400" data-numeric>
            {Math.round(node.confidence * 100)}%
          </span>
        )}
      </div>
      <p className="mt-0.5 line-clamp-2 text-[10px] leading-tight text-ink-400">
        {node.hypothesis || 'No hypothesis recorded for this step.'}
      </p>
    </div>
  )
}

function TrialNode({
  node,
  metric,
  x,
  y,
  selected,
  registerRef,
  onSelect,
}: {
  node: ExecutionTreeNode
  metric: string
  x: number
  y: number
  selected: boolean
  registerRef: (element: HTMLButtonElement | null) => void
  onSelect: () => void
}) {
  const running = node.status === 'RUNNING'
  const meta = node.status === 'RUNNING' ? null : TRIAL_STATUS_META[node.status]
  const border = running ? RUNNING_NODE : (meta?.node ?? 'border-ink-700')
  // A trial that failed or was pruned produced no target metric. That is unrecorded,
  // not zero, so the badge says so rather than printing a direction beside a dash.
  const badge = running
    ? 'running'
    : node.score === null
      ? 'not recorded'
      : `${formatNumber(node.score, 4)}${isMinimized(metric) ? ' ↓' : ' ↑'}`

  return (
    <button
      type="button"
      ref={registerRef}
      onClick={onSelect}
      aria-pressed={selected}
      aria-current={selected ? 'true' : undefined}
      className={`absolute rounded-md border px-3 py-1.5 text-left transition-colors hover:border-ink-500 ${
        selected ? 'border-signal-500 bg-signal-500/10' : border
      } ${running ? 'bg-ink-850' : 'bg-ink-900'}`}
      style={{ left: x, top: y, width: NODE_W.trial, height: NODE_H.trial }}
      title={
        node.failureReason
          ? `${node.status}: ${node.failureReason}`
          : `${node.status} · ${node.trialId ?? 'trial id not yet recorded'}`
      }
    >
      {running && (
        <span
          className="pointer-events-none absolute inset-0 animate-ping rounded-md border border-warn-400/40"
          aria-hidden="true"
        />
      )}
      <span className="relative flex items-center gap-1.5">
        <span className={`h-2 w-2 shrink-0 rounded-full ${running ? 'animate-pulse bg-warn-400' : meta?.dot}`} />
        <span className="truncate font-mono text-[10px] text-ink-200" title={node.trialId ?? undefined}>
          {node.trialId ?? `${node.candidateId || 'candidate'}…`}
        </span>
        <span
          className={`ml-auto shrink-0 whitespace-nowrap rounded px-1.5 py-px font-mono text-[10px] ${
            running ? 'bg-warn-400/15 text-warn-400' : 'bg-ink-750 text-ink-300'
          }`}
          data-numeric
          title={running ? 'The worker has not returned yet' : 'Target metric for this trial'}
        >
          {badge}
        </span>
      </span>
      <span className="relative mt-0.5 block truncate text-[10px] text-ink-500">
        {running ? 'worker has not returned' : (node.failureReason ?? meta?.label ?? node.status)}
      </span>
    </button>
  )
}

function FailureNode({ node, x, y }: { node: ExecutionTreeNode; x: number; y: number }) {
  return (
    <div
      className="absolute rounded border border-danger-400/40 bg-danger-400/10 px-2 py-1"
      style={{ left: x, top: y, width: NODE_W.trial, height: NODE_H.failure }}
      title={node.failureReason ?? 'Execution failure'}
    >
      <p className="truncate text-[10px] text-danger-400">{node.failureReason}</p>
    </div>
  )
}

/**
 * Detail for the selected node: the exact configuration that ran, and the lineage
 * around it. An absent parent is stated rather than left blank, because "no parent
 * recorded" and "root" are different claims and the backend only ever writes the former.
 */
function NodeDetail({
  node,
  metric,
  derivedTrials,
  onClose,
}: {
  node: ExecutionTreeNode
  metric: string
  /** Trials the server records as derived from this one. Usually empty. */
  derivedTrials: ExecutionTreeNode[]
  onClose: () => void
}) {
  const meta = node.status === 'RUNNING' ? null : TRIAL_STATUS_META[node.status]
  const decision = node.decisionType ? DECISION_META[node.decisionType] : null

  return (
    <div className="border-t border-ink-750 bg-ink-900 px-5 py-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h3 className="flex items-center gap-2 text-sm font-medium text-ink-100">
            <span className="mono-id">{node.trialId ?? 'trial pending'}</span>
            <span className="text-[11px] font-normal text-ink-500">
              step {node.step} · {meta?.label ?? 'running'}
            </span>
          </h3>
          {node.hypothesis && <p className="mt-1 max-w-3xl text-xs leading-relaxed text-ink-400">{node.hypothesis}</p>}
        </div>
        <button type="button" className="btn btn-ghost shrink-0 text-xs" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="mt-3 grid gap-4 md:grid-cols-3">
        <div>
          <p className="label">Hyperparameters executed</p>
          <dl className="mt-1 space-y-0.5">
            {Object.keys(node.hyperparameters).length === 0 ? (
              <p className="text-[11px] text-ink-500">Not recorded until the worker returns.</p>
            ) : (
              Object.entries(node.hyperparameters).map(([key, value]) => (
                <div key={key} className="flex items-baseline justify-between gap-3">
                  <dt className="truncate font-mono text-[11px] text-ink-500">{key}</dt>
                  <dd className="font-mono text-[11px] text-ink-100" data-numeric>
                    {formatHyperparameter(value)}
                  </dd>
                </div>
              ))
            )}
          </dl>
        </div>

        <div>
          <p className="label">Lineage</p>
          <dl className="mt-1 space-y-0.5 text-[11px]">
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">DERIVED_FROM</dt>
              <dd className="mono-id">hyp_{node.candidateId || '—'}</dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">EVALUATED_ON</dt>
              <dd className="mono-id">the run's dataset node</dd>
            </div>
            {node.status === 'FAILED' && (
              <div className="flex items-baseline justify-between gap-3">
                <dt className="text-ink-500">FAILED_UNDER</dt>
                <dd className="truncate text-danger-400" title={node.failureReason ?? ''}>
                  {node.failureReason ?? 'not recorded'}
                </dd>
              </div>
            )}
            <div className="flex items-baseline justify-between gap-3 border-t border-ink-800 pt-1">
              <dt className="text-ink-500">Parent trial</dt>
              <dd className="mono-id">{node.parentTrialId ?? 'none recorded'}</dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">Derived trials</dt>
              <dd className="mono-id">
                {derivedTrials.length ? derivedTrials.map((child) => child.trialId).join(', ') : 'none recorded'}
              </dd>
            </div>
          </dl>
        </div>

        <div>
          <p className="label">Outcome</p>
          <dl className="mt-1 space-y-0.5 text-[11px]">
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">{metric}</dt>
              <dd className="font-mono text-ink-100" data-numeric>
                {node.score === null ? 'not recorded' : formatNumber(node.score, 4) + (isMinimized(metric) ? ' ↓' : ' ↑')}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">Recorded at</dt>
              <dd className="font-mono text-ink-300" data-numeric>
                {formatTimestamp(node.at)}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">Peak memory</dt>
              <dd className="font-mono text-ink-200" data-numeric>
                {node.memoryMb === null ? 'not recorded' : `${formatNumber(node.memoryMb, 0)} MB`}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">Train time</dt>
              <dd className="font-mono text-ink-200" data-numeric>
                {node.metrics.train_time_sec === undefined || node.metrics.train_time_sec === null
                  ? 'not recorded'
                  : formatDuration(node.metrics.train_time_sec)}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">Strategy</dt>
              <dd className="text-ink-200" title={decision?.blurb}>
                {decision ? `${decision.posture} · ${decision.label}` : 'not recorded'}
              </dd>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-ink-500">Graph writes</dt>
              <dd className="font-mono text-ink-200" data-numeric>
                {node.graphDelta
                  ? `+${node.graphDelta.nodes_added} nodes / +${node.graphDelta.edges_added} edges`
                  : 'pending'}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      {node.citedPaths.length > 0 && (
        <div className="mt-3 border-t border-ink-800 pt-2">
          <p className="label">Graph evidence cited for this decision</p>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {node.citedPaths.map((path) => (
              <span key={path} className="mono-id">
                {path}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}