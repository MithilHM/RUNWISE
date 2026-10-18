/**
 * The runtime's view model.
 *
 * One reducer folds the `RuntimeStepEvent` stream into the three things the screen
 * renders — a tree of trials, a reasoning feed and a log of graph mutations — so the
 * panels can never show contradictory accounts of the same step. The reducer is pure
 * and the same function serves a real SSE stream, a replayed demo stream and a
 * rehydration from `GET /status`, which is why a reload mid-run restores an identical
 * picture.
 *
 * Nothing in here touches React.
 */

import type {
  AblationCurvePoint,
  ExecutionTreeEntry,
  GraphDelta,
  LayaDecision,
  RuntimeObjective,
  RuntimeStage,
  RuntimeStatusResponse,
  RuntimeStepEvent,
  TrialResult,
} from '../../api/runtimeApi'

export type RuntimePhase = 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'error'

/** A trial node can also be in flight, before the worker has returned a verdict. */
export type TreeNodeStatus = 'RUNNING' | 'SUCCESS' | 'FAILED' | 'PRUNED'

export interface ExecutionTreeNode {
  step: number
  /** Null until the worker reports the trial id. */
  trialId: string | null
  candidateId: string
  decisionType: LayaDecision['decision_type'] | null
  hypothesis: string
  status: TreeNodeStatus
  score: number | null
  hyperparameters: Record<string, unknown>
  metrics: Record<string, number | null>
  memoryMb: number | null
  failureReason: string | null
  /**
   * Set only when the server states it. The backend's own `DERIVED_FROM` edge points
   * from a trial to the hypothesis of the same step, so cross-step lineage is absent
   * in practice and must not be invented to fill the graph.
   */
  parentTrialId: string | null
  citedPaths: string[]
  confidence: number | null
  graphDelta: GraphDelta | null
  at: string
}

export interface LayaReasoningEntry {
  id: string
  step: number
  stage: RuntimeStage
  at: string
  message: string | null
  decision: LayaDecision | null
  trial: TrialResult | null
  bestScore: number | null
}

export interface GraphMutationRecord {
  id: string
  step: number
  at: string
  kind: 'node' | 'edge'
  /** The graph label: `Experiment`, `EVALUATED_ON`, `FAILED_UNDER`, … */
  label: string
  detail: string
  /** True for a count the backend reported but the documented mutation set cannot name. */
  residual: boolean
}

export interface RuntimeState {
  runId: string
  phase: RuntimePhase
  stage: RuntimeStage
  step: number
  bestScore: number | null
  objective: RuntimeObjective
  nodes: ExecutionTreeNode[]
  feed: LayaReasoningEntry[]
  /** Newest first, so both the feed and the ticker read index 0 as the latest write. */
  mutations: GraphMutationRecord[]
  totals: GraphDelta
  /**
   * Ids of events already folded in, as `${step_number}-${stage}`.
   *
   * The backend backfills its whole event history into every new subscriber queue
   * (`RuntimeLoop.subscribe`), so a reattach — or the reconcile read that precedes the
   * first attach — replays frames this screen has already counted. Without this,
   * pressing "Reattach" would roughly double the node and edge totals and duplicate
   * every feed row. Folding is therefore idempotent: an event already applied is a
   * no-op, and the reducer returns the identical state object.
   */
  seenEvents: string[]
  /** True when the events came from the local simulation rather than the backend. */
  simulated: boolean
  error: string | null
}

export function initialRuntimeState(objective: RuntimeObjective, runId: string): RuntimeState {
  return {
    runId,
    phase: 'idle',
    stage: 'QUERYING_GRAPH',
    step: 0,
    bestScore: null,
    objective,
    nodes: [],
    feed: [],
    mutations: [],
    totals: { nodes_added: 0, edges_added: 0 },
    seenEvents: [],
    simulated: false,
    error: null,
  }
}

/** Stable identity for an event frame, used to make folding idempotent. */
function eventId(event: RuntimeStepEvent): string {
  return `${event.step_number}-${event.stage}`
}

/** Fold one event in. An event already applied returns the same state object. */
export function applyEvent(state: RuntimeState, event: RuntimeStepEvent): RuntimeState {
  const id = eventId(event)
  if (state.seenEvents.includes(id)) return state

  const trial = event.trial_result
  const decision = event.laya_decision
  const terminal = event.stage === 'COMPLETED'

  const nodes = mergeNode(state.nodes, event, state.objective.target_metric)
  const feed: LayaReasoningEntry[] = [
    {
      id,
      step: event.step_number,
      stage: event.stage,
      at: event.timestamp,
      message: event.message,
      decision,
      trial,
      bestScore: event.current_best_metric,
    },
    ...state.feed,
  ]

  const mutations =
    event.stage === 'MUTATING_GRAPH'
      ? [...expandMutations(event, state.objective.dataset_name), ...state.mutations]
      : state.mutations

  return {
    ...state,
    phase: terminal ? 'completed' : 'running',
    stage: event.stage,
    step: event.step_number,
    bestScore: event.current_best_metric,
    nodes,
    feed,
    mutations,
    seenEvents: [...state.seenEvents, id],
    totals: {
      nodes_added: state.totals.nodes_added + (event.graph_delta?.nodes_added ?? 0),
      edges_added: state.totals.edges_added + (event.graph_delta?.edges_added ?? 0),
    },
  }
}

function blankNode(step: number, at: string): ExecutionTreeNode {
  return {
    step,
    trialId: null,
    candidateId: '',
    decisionType: null,
    hypothesis: '',
    status: 'RUNNING',
    score: null,
    hyperparameters: {},
    metrics: {},
    memoryMb: null,
    failureReason: null,
    parentTrialId: null,
    citedPaths: [],
    confidence: null,
    graphDelta: null,
    at,
  }
}

/**
 * A step arrives three times: `LAYA_EVALUATING` opens the node, `TRAINING_WORKER`
 * closes it with a verdict, `MUTATING_GRAPH` attaches the write. Later stages win
 * field by field, so a partial event can never blank out what an earlier one recorded.
 */
function mergeNode(
  nodes: ExecutionTreeNode[],
  event: RuntimeStepEvent,
  targetMetric: string,
): ExecutionTreeNode[] {
  if (!event.laya_decision && !event.trial_result) return nodes

  const index = nodes.findIndex((node) => node.step === event.step_number)
  const existing = index >= 0 ? nodes[index] : blankNode(event.step_number, event.timestamp)

  const merged: ExecutionTreeNode = {
    ...existing,
    candidateId: event.laya_decision?.selected_candidate_id ?? existing.candidateId,
    decisionType: event.laya_decision?.decision_type ?? existing.decisionType,
    hypothesis: event.laya_decision?.hypothesis_text ?? existing.hypothesis,
    citedPaths: event.laya_decision?.cited_graph_paths ?? existing.citedPaths,
    confidence: event.laya_decision?.confidence ?? existing.confidence,
    trialId: event.trial_result?.trial_id ?? existing.trialId,
    hyperparameters: event.trial_result?.hyperparameters ?? existing.hyperparameters,
    metrics: event.trial_result?.metrics ?? existing.metrics,
    memoryMb: event.trial_result?.memory_mb ?? existing.memoryMb,
    failureReason: event.trial_result?.failure_reason ?? existing.failureReason,
    parentTrialId: event.trial_result?.parent_trial_id ?? existing.parentTrialId,
    score: scoreFor(targetMetric, event.trial_result) ?? existing.score,
    status: event.trial_result?.status ?? existing.status,
    graphDelta: event.stage === 'MUTATING_GRAPH' ? event.graph_delta : existing.graphDelta,
    at: event.timestamp,
  }

  if (index >= 0) {
    const next = [...nodes]
    next[index] = merged
    return next
  }
  return [...nodes, merged].sort((a, b) => a.step - b.step)
}

/**
 * The target metric's value for a trial, resolved by metric name rather than by
 * position: `metrics` also carries loss and timing, and a positional read would pick
 * whichever landed first. A trial that failed before producing one reads as null,
 * which the tree renders as "not recorded" rather than as zero.
 */
function scoreFor(targetMetric: string, trial: TrialResult | null | undefined): number | null {
  if (!trial) return null
  const value = trial.metrics[targetMetric]
  return value === undefined || value === null || Number.isNaN(value) ? null : value
}

/**
 * Re-expand the backend's anonymous node/edge counts into the names it wrote.
 *
 * `graph_delta` arrives as `{nodes_added, edges_added}` with no labels. The issue
 * order below is the one `RuntimeGraphService.mutate_runtime_trial` uses, so the
 * ticker can name every write instead of only counting them.
 *
 * Anything the server counted but this list cannot account for is emitted as a single
 * residual row rather than dropped: a silent gap would read as "nothing else was
 * written", which is the one claim this panel is not allowed to make.
 */
export function expandMutations(event: RuntimeStepEvent, datasetName: string): GraphMutationRecord[] {
  const trial = event.trial_result
  if (!trial) return []

  const step = event.step_number
  const at = event.timestamp
  const base = `${step}-${at}`
  const records: GraphMutationRecord[] = []
  const push = (kind: 'node' | 'edge', label: string, detail: string) => {
    records.push({
      id: `${base}-${label}-${records.length}`,
      step,
      at,
      kind,
      label,
      detail,
      residual: false,
    })
  }

  push('node', 'Experiment', trial.trial_id)
  push('edge', 'EVALUATED_ON', `→ ${datasetName}`)

  const hypothesisId = `hyp_${trial.candidate_id}`
  if (event.laya_decision) {
    push('node', 'Hypothesis', hypothesisId)
    push('edge', 'DERIVED_FROM', `→ ${hypothesisId}`)
  }

  if (trial.status === 'FAILED') {
    const reason = trial.failure_reason ?? 'execution failure'
    push('node', 'FailureReason', reason)
    push('edge', 'FAILED_UNDER', `→ ${reason}`)
  }

  for (const [name, value] of Object.entries(trial.metrics)) {
    if (value === null || value === undefined) continue
    push('node', 'Metric', `${name}=${value}`)
    push('edge', 'PRODUCES_METRIC', `→ ${name}`)
  }

  const namedNodes = records.filter((record) => record.kind === 'node').length
  const namedEdges = records.filter((record) => record.kind === 'edge').length
  const nodeSlack = event.graph_delta.nodes_added - namedNodes
  const edgeSlack = event.graph_delta.edges_added - namedEdges

  // Node and edge gaps are reported separately. Collapsing them with a min() would
  // quietly drop the larger one, which is the case this row exists to surface.
  if (nodeSlack > 0) {
    records.push({
      id: `${base}-residual-node`,
      step,
      at,
      kind: 'node',
      label: 'Unnamed',
      detail: `${nodeSlack} node(s) counted by the server but not part of the documented mutation set`,
      residual: true,
    })
  }
  if (edgeSlack > 0) {
    records.push({
      id: `${base}-residual-edge`,
      step,
      at,
      kind: 'edge',
      label: 'Unnamed',
      detail: `${edgeSlack} edge(s) counted by the server but not part of the documented mutation set`,
      residual: true,
    })
  }

  return records
}

/**
 * Reconcile the stream against the server's stored status.
 *
 * Merge, never replace. The snapshot's `execution_tree` rows carry a verdict but no
 * metric map, no cited evidence and no confidence, so overwriting streamed nodes with
 * them silently deletes detail the stream already delivered — the per-trial duration
 * in particular, which is what prices the ablation's compute saving. Steps the stream
 * has never seen are added outright; every other field of a known step is left alone.
 *
 * This also runs on a timer while a run is live, as a safety net against a dropped
 * stream, so the feed is only told about it when it actually contributed a step.
 * Logging every poll would fill the feed with identical "restored N trials" lines
 * within a minute.
 */
export function reconcileFromStatus(state: RuntimeState, status: RuntimeStatusResponse): RuntimeState {
  const byStep = new Map(state.nodes.map((node) => [node.step, node]))
  const added: number[] = []

  const fromSnapshot = status.execution_tree.map((entry) => {
    const existing = byStep.get(entry.step)
    if (!existing) added.push(entry.step)
    return mergeEntry(existing, entry, status.objective)
  })

  const snapshotSteps = new Set(fromSnapshot.map((node) => node.step))
  /* Every known node the snapshot has not caught up with is carried over, not just
     the ones still holding an open trial. The backend broadcasts `TRAINING_WORKER`
     before it appends to `execution_tree`, so there is a real window in which the
     stream knows about a closed trial that the snapshot does not list yet; dropping
     it would make a row blink out and reappear on the next poll. */
  const ahead = state.nodes.filter((node) => !snapshotSteps.has(node.step))
  const merged = [...fromSnapshot, ...ahead].sort((a, b) => a.step - b.step)

  /* A `FAILED` run must not read as a success. The stream signals failure by emitting
     a `COMPLETED` stage with a failure message, so the stage alone cannot be trusted;
     the status endpoint's own status field can. */
  const failed = status.status === 'FAILED'
  const phase: RuntimePhase = failed
    ? 'error'
    : status.status === 'COMPLETED'
      ? 'completed'
      : state.phase === 'starting'
        ? 'running'
        : state.phase

  /* Mark the restored steps as already folded, so the stream's backfill of the same
     history does not re-add their deltas and roughly double the totals. */
  const seen = new Set(state.seenEvents)
  for (const step of added) seen.add(`${step}-MUTATING_GRAPH`)

  return {
    ...state,
    runId: status.run_id,
    objective: status.objective ?? state.objective,
    phase,
    step: merged.length > 0 ? merged[merged.length - 1].step : state.step,
    bestScore: status.best_score ?? state.bestScore,
    nodes: merged,
    seenEvents: [...seen],
    totals: {
      nodes_added: Math.max(status.graph_mutation_stats?.nodes_added ?? 0, state.totals.nodes_added),
      edges_added: Math.max(status.graph_mutation_stats?.edges_added ?? 0, state.totals.edges_added),
    },
    error: failed ? (state.error ?? 'The runtime loop reported a failure. See the reasoning feed for the step it stopped on.') : state.error,
    feed:
      added.length > 0
        ? [
            {
              id: `rehydrate-status-${added.length}-${merged.length}`,
              step: merged.length > 0 ? merged[merged.length - 1].step : 0,
              stage: 'MUTATING_GRAPH',
              at: new Date().toISOString(),
              message: `Restored ${added.length} trial(s) this run had already recorded before the stream attached.`,
              decision: null,
              trial: null,
              bestScore: status.best_score ?? null,
            },
            ...state.feed,
          ]
        : state.feed,
  }
}

function mergeEntry(
  existing: ExecutionTreeNode | null | undefined,
  entry: ExecutionTreeEntry,
  objective: RuntimeObjective | null,
): ExecutionTreeNode {
  const base = existing ?? nodeFromEntry(entry, objective)
  const hyperparameters = entry.hyperparameters ?? {}
  return {
    ...base,
    // The snapshot is authoritative about the verdict and the configuration.
    status: entry.status,
    score: entry.score ?? base.score,
    trialId: entry.trial_id ?? base.trialId,
    candidateId: entry.candidate_id || base.candidateId,
    decisionType: entry.decision_type ?? base.decisionType,
    hypothesis: entry.hypothesis || base.hypothesis,
    hyperparameters: Object.keys(hyperparameters).length > 0 ? hyperparameters : base.hyperparameters,
    graphDelta: entry.graph_delta ?? base.graphDelta,
    // Left as the stream set them, because only the stream carries them: metrics,
    // citedPaths, confidence, memoryMb, failureReason, parentTrialId.
  }
}

function nodeFromEntry(entry: ExecutionTreeEntry, objective: RuntimeObjective | null): ExecutionTreeNode {
  return {
    step: entry.step,
    trialId: entry.trial_id,
    candidateId: entry.candidate_id,
    decisionType: entry.decision_type,
    hypothesis: entry.hypothesis ?? '',
    status: entry.status,
    score: entry.score ?? null,
    hyperparameters: entry.hyperparameters ?? {},
    // These rows carry no metric map, so only the target metric is claimable.
    metrics: objective ? { [objective.target_metric]: entry.score ?? null } : {},
    memoryMb: null,
    failureReason: null,
    parentTrialId: null,
    citedPaths: [],
    confidence: null,
    graphDelta: entry.graph_delta ?? { nodes_added: 0, edges_added: 0 },
    at: new Date().toISOString(),
  }
}

/** Trials whose worker verdict is in. Drives the budget progress readout. */
export function settledCount(nodes: ExecutionTreeNode[]): number {
  return nodes.filter((node) => node.status !== 'RUNNING').length
}

export function failureCount(nodes: ExecutionTreeNode[]): number {
  return nodes.filter((node) => node.status === 'FAILED' || node.status === 'PRUNED').length
}

/**
 * How many trials each arm needed to first reach the target. The target is the best
 * value the leading arm achieved, so the comparison answers "what did it cost the
 * others to get here", not "who finished higher".
 *
 * Direction is explicit because it cannot be inferred from the numbers: a
 * best-so-far curve running downwards is converging on a minimised metric, and
 * comparing it with `>=` would invert every conclusion drawn from it.
 */
export function trialsToReach(
  curve: AblationCurvePoint[] | undefined,
  target: number,
  minimize = false,
): number | null {
  if (!curve?.length) return null
  const reached = curve.find((point) => (minimize ? point.best_score <= target : point.best_score >= target))
  return reached ? reached.trial : null
}

/**
 * Mean wall-clock cost of one trial, from the run's own recorded timings. Null when
 * no trial recorded a duration, so the compute-saved figure can say "not recorded"
 * rather than assume a cost per trial.
 */
export function meanTrialSeconds(nodes: ExecutionTreeNode[]): number | null {
  const durations = nodes
    .map((node) => node.metrics.train_time_sec)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0)
  if (!durations.length) return null
  return durations.reduce((total, value) => total + value, 0) / durations.length
}

/** Render a hyperparameter cell without letting an object spill raw JSON across the panel. */
export function formatHyperparameter(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}