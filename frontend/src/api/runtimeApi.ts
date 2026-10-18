/**
 * Autonomous runtime client.
 *
 * The runtime loop is the one surface in RUNWISE that is not request/response: it
 * is a Server-Sent Events stream of `RuntimeStepEvent`s. The REST helpers here reuse
 * `request` from `./client` so there is still one base URL and one `ApiError` shape,
 * but the stream reader is written out by hand because `fetch` cannot parse SSE and
 * `EventSource` cannot be pointed at a base URL that lives behind a dev proxy.
 *
 * Types mirror `backend/app/models/runtime_contracts.py` field for field. Notes that
 * change how the UI renders:
 *
 *  - `graph_delta` counts nodes and edges but does not name them. `GraphMutationTicker`
 *    re-expands it into the named mutations that `RuntimeGraphService.mutate_runtime_trial`
 *    actually issues; see `expandMutations` there.
 *  - `status`, `stage` and `decision_type` are machine enums. They are rendered as
 *    words, never passed through as colour decisions.
 *  - `score` is a metric value, not a probability, and may be `null` when a trial
 *    failed before producing one. Null renders as "not recorded", never as 0.
 *  - `cited_graph_paths` is the evidence Laya leaned on. An empty list is a valid
 *    answer meaning "no graph evidence was cited", not "unknown".
 *  - The ablation endpoint compares Graph+Laya against uniform random search only. The
 *    unconstrained third arm is optional here and is never synthesised client-side;
 *    absent means "not measured".
 */

import { BASE_URL, request } from './client'

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/** The five closed-loop phases, plus the terminal one. */
export type RuntimeStage =
  | 'QUERYING_GRAPH'
  | 'GENERATING_CANDIDATES'
  | 'LAYA_EVALUATING'
  | 'TRAINING_WORKER'
  | 'MUTATING_GRAPH'
  | 'COMPLETED'

export type TrialStatus = 'SUCCESS' | 'FAILED' | 'PRUNED'

export type LayaDecisionType =
  | 'exploit_promising_regime'
  | 'explore_boundary'
  | 'avoid_failure_subgraph'

/**
 * Task configuration for one optimisation loop. Only `dataset_name`, `target_metric`,
 * `model_family` and `trial_budget` are user-facing; the search space and timeout keep
 * their backend defaults unless a caller overrides them.
 */
export interface RuntimeObjective {
  dataset_name: string
  target_metric: string
  model_family: string
  trial_budget: number
  timeout_seconds?: number
  project_id?: string
  search_space?: Record<string, unknown[]>
}

/** A proposed hyperparameter point, before it has been trained. */
export interface CandidateConfig {
  candidate_id: string
  hyperparameters: Record<string, unknown>
  rationale: string
  source_regime_edges: string[]
}

/** Laya's pick for the current step, with the reasoning that justified it. */
export interface LayaDecision {
  selected_candidate_id: string
  decision_type: LayaDecisionType
  hypothesis_text: string
  cited_graph_paths: string[]
  confidence: number
}

/** The outcome of training one candidate. */
export interface TrialResult {
  trial_id: string
  candidate_id: string
  hyperparameters: Record<string, unknown>
  metrics: Record<string, number | null>
  status: TrialStatus
  failure_reason: string | null
  memory_mb: number
  /**
   * Not part of the published contract. The backend links each trial to a
   * hypothesis node of its own step, never across steps, so this is absent in
   * practice; the tree renders cross-step lineage only when it is supplied.
   */
  parent_trial_id?: string | null
}

/** Nodes and edges written to FalkorDB during one step. */
export interface GraphDelta {
  nodes_added: number
  edges_added: number
}

/** One SSE frame. Six per trial plus the terminal frame. */
export interface RuntimeStepEvent {
  step_number: number
  stage: RuntimeStage
  current_best_metric: number | null
  trial_result: TrialResult | null
  laya_decision: LayaDecision | null
  graph_delta: GraphDelta
  timestamp: string
  message: string | null
}

/** One point on a convergence curve. `score` is this trial, `best_score` the running max. */
export interface AblationCurvePoint {
  trial: number
  score: number
  best_score: number
}

/**
 * Convergence comparison for a completed run. The first two curves are measured by
 * the backend. The unconstrained arm is optional and stays absent unless a server
 * actually runs one, because an invented third curve would read as evidence.
 */
export interface AblationResult {
  run_id: string
  budget: number
  laya_curve: AblationCurvePoint[]
  random_curve: AblationCurvePoint[]
  laya_best: number
  random_best: number
  improvement_percentage: number
  summary: string
  unconstrained_curve?: AblationCurvePoint[]
  unconstrained_best?: number
}

/** Snapshot of a run, used to rehydrate the page after a reload mid-run. */
export interface RuntimeStatusResponse {
  run_id: string
  status: string
  objective: RuntimeObjective
  best_score: number | null
  best_hyperparameters: Record<string, unknown> | null
  completed_trials: TrialResult[]
  execution_tree: ExecutionTreeEntry[]
  graph_mutation_stats: GraphDelta
}

/** One row of `execution_tree` as the backend appends it. */
export interface ExecutionTreeEntry {
  step: number
  trial_id: string
  candidate_id: string
  decision_type: LayaDecisionType
  hypothesis: string
  score: number | null
  status: TrialStatus
  hyperparameters: Record<string, unknown>
  graph_delta: GraphDelta
}

export interface StartRunResponse {
  run_id: string
  status: string
  dataset_name: string
  target_metric: string
  trial_budget: number
}

// ---------------------------------------------------------------------------
// Transport
// ---------------------------------------------------------------------------

export interface RuntimeStreamHandle {
  /** Detach the listener. Idempotent. */
  close(): void
}

function search(params: Record<string, string | number | boolean | undefined | null>): string {
  const parts = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    parts.set(key, String(value))
  }
  const encoded = parts.toString()
  return encoded ? `?${encoded}` : ''
}

/**
 * Subscribe to a run's step events.
 *
 * One detail of the backend stream is load-bearing: its frames are unnamed
 * (`data: {...}`), so they arrive as default `message` events with no `event:`
 * type to filter on.
 *
 * The server also emits `: keep-alive` comment frames every 15s while a trial
 * trains. Those are handled below the `EventSource` API â€” the SSE parser discards
 * comment lines without dispatching an event â€” so they never reach `onmessage`.
 * The empty-payload guard below is belt-and-braces for a proxy that rewrites a
 * comment into an empty `data:` field, which would otherwise make `JSON.parse('')`
 * throw on a perfectly healthy stream.
 *
 * A transient drop is left to `EventSource`'s own reconnection rather than being
 * closed out. The server replays the run's full history to each new subscriber and
 * the caller folds idempotently, so a reconnect recovers silently and without
 * double-counting; the caller's status poll covers the case where it does not.
 * Only a genuinely closed socket is surfaced, and a `COMPLETED` frame closes the
 * source itself so the loop does not outlive the run.
 */
export function streamRuntimeEvents(
  runId: string,
  onEvent: (event: RuntimeStepEvent) => void,
  onError?: (error: Error) => void,
): RuntimeStreamHandle {
  const source = new EventSource(`${BASE_URL}/api/runtime/${runId}/stream`)
  let closed = false

  const close = () => {
    if (closed) return
    closed = true
    source.close()
  }

  source.onmessage = (message) => {
    if (!message.data) return
    try {
      const event = JSON.parse(message.data) as RuntimeStepEvent
      onEvent(event)
      if (event.stage === 'COMPLETED') close()
    } catch (cause) {
      onError?.(new Error(`Malformed runtime event: ${(cause as Error).message}`))
    }
  }

  source.onerror = () => {
    if (closed) return
    /* CLOSED means the browser gave up; CONNECTING means it is reconnecting on its
       own, which is silent recovery rather than a failure worth reporting. */
    if (source.readyState === EventSource.CLOSED) {
      close()
      onError?.(new Error(`Lost the event stream for run ${runId}. The run may still be alive.`))
    }
  }

  return { close }
}

export const runtimeApi = {
  /** Launch the closed-loop optimiser in the background and get back its run id. */
  startAutonomousRun: (objective: RuntimeObjective) =>
    request<StartRunResponse>('/api/runtime/start', {
      method: 'POST',
      body: JSON.stringify(objective),
    }),

  streamRuntimeEvents,

  /** Backfill the tree after a reload, or reconcile a stream against the server. */
  getRuntimeStatus: (runId: string) => request<RuntimeStatusResponse>(`/api/runtime/${runId}/status`),

  /**
   * Compare Graph+Laya against uniform random search over the same budget. The
   * endpoint is POST-only (the budget is a query parameter, the comparison itself is
   * not idempotent â€” it trains trials), and the budget is capped server-side at 2..30.
   */
  runAblationStudy: (runId: string, budget = 10) =>
    request<AblationResult>(`/api/runtime/${runId}/ablation${search({ budget })}`, {
      method: 'POST',
    }),
}
