import { describe, expect, it } from 'vitest'
import type { LayaDecision, RuntimeObjective, RuntimeStatusResponse, RuntimeStepEvent, TrialResult } from '../../api/runtimeApi'
import {
  applyEvent,
  expandMutations,
  initialRuntimeState,
  meanTrialSeconds,
  reconcileFromStatus,
  trialsToReach,
  type RuntimeState,
} from './runtimeModel'

/**
 * The reducer is the whole runtime screen: every panel reads from it, and it is fed
 * by a transport that replays history. These tests pin the properties that a passing
 * build cannot catch — idempotent folding, no clobbering across a step's three frames,
 * and above all that an absent measurement never becomes a zero.
 */

const OBJECTIVE: RuntimeObjective = {
  dataset_name: 'credit_card_fraud',
  target_metric: 'val_pr_auc',
  model_family: 'lightgbm',
  trial_budget: 3,
}

const DECISION: LayaDecision = {
  selected_candidate_id: 'cand_001',
  decision_type: 'explore_boundary',
  hypothesis_text: 'Probe the depth boundary.',
  cited_graph_paths: ['hyp_cand_001'],
  confidence: 0.8,
}

function frame(overrides: Partial<RuntimeStepEvent>): RuntimeStepEvent {
  return {
    step_number: 1,
    stage: 'LAYA_EVALUATING',
    current_best_metric: null,
    trial_result: null,
    laya_decision: null,
    graph_delta: { nodes_added: 0, edges_added: 0 },
    timestamp: '2026-10-18T10:00:00Z',
    message: null,
    ...overrides,
  }
}

function trial(overrides: Partial<TrialResult> = {}): TrialResult {
  return {
    trial_id: 'trial_0001',
    candidate_id: 'cand_001',
    hyperparameters: { max_depth: 4 },
    metrics: { val_pr_auc: 0.82, train_time_sec: 0.4 },
    status: 'SUCCESS',
    failure_reason: null,
    memory_mb: 1024,
    ...overrides,
  }
}

/** One full step: decision, verdict, then the graph write. */
function foldStep(state: RuntimeState, step: number, result: TrialResult = trial()): RuntimeState {
  const decision = frame({ step_number: step, stage: 'LAYA_EVALUATING', laya_decision: DECISION })
  const trained = frame({ step_number: step, stage: 'TRAINING_WORKER', trial_result: result, current_best_metric: 0.82 })
  const mutated = frame({
    step_number: step,
    stage: 'MUTATING_GRAPH',
    trial_result: result,
    laya_decision: DECISION,
    current_best_metric: 0.82,
    graph_delta: { nodes_added: 4, edges_added: 4 },
  })
  return applyEvent(applyEvent(applyEvent(state, decision), trained), mutated)
}

describe('applyEvent', () => {
  it('opens a trial node on the decision frame, before the worker returns', () => {
    const state = applyEvent(initialRuntimeState(OBJECTIVE, 'run_1'), frame({ laya_decision: DECISION }))

    expect(state.nodes).toHaveLength(1)
    expect(state.nodes[0].status).toBe('RUNNING')
    expect(state.nodes[0].trialId).toBeNull()
    expect(state.nodes[0].decisionType).toBe('explore_boundary')
    expect(state.phase).toBe('running')
  })

  it('keeps the hypothesis recorded on the later frames that omit it', () => {
    const state = foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1)

    expect(state.nodes[0].hypothesis).toBe('Probe the depth boundary.')
    expect(state.nodes[0].citedPaths).toEqual(['hyp_cand_001'])
    expect(state.nodes[0].status).toBe('SUCCESS')
    // The target metric is resolved by name, not by position in the metrics map.
    expect(state.nodes[0].score).toBe(0.82)
  })

  it('records a failed trial with no metric as unrecorded rather than zero', () => {
    const failed = trial({
      status: 'FAILED',
      failure_reason: 'Memory limit exceeded (OOM)',
      metrics: { val_pr_auc: null, train_time_sec: 0.9 },
    })
    const state = foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1, failed)
    const node = state.nodes[0]

    expect(node.status).toBe('FAILED')
    expect(node.score).toBeNull()
    expect(node.failureReason).toBe('Memory limit exceeded (OOM)')
  })

  it('is idempotent, because the server replays history to every new subscriber', () => {
    const once = foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1)
    const replayed = applyEvent(once, frame({ stage: 'MUTATING_GRAPH', trial_result: trial(), graph_delta: { nodes_added: 4, edges_added: 4 } }))

    expect(replayed).toBe(once)
    expect(replayed.totals.nodes_added).toBe(4)
    expect(replayed.nodes).toHaveLength(1)
    expect(replayed.mutations).toHaveLength(once.mutations.length)
  })

  it('accumulates graph writes once per step, not once per event', () => {
    let state = initialRuntimeState(OBJECTIVE, 'run_1')
    state = foldStep(state, 1)
    state = foldStep(state, 2)

    expect(state.totals).toEqual({ nodes_added: 8, edges_added: 8 })
    expect(state.nodes).toHaveLength(2)
  })

  it('marks the run completed on the terminal frame', () => {
    const state = applyEvent(foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1), frame({ stage: 'COMPLETED' }))

    expect(state.phase).toBe('completed')
  })
})

describe('expandMutations', () => {
  const mutate = (result: TrialResult, decision: LayaDecision | null = DECISION) =>
    frame({ stage: 'MUTATING_GRAPH', trial_result: result, laya_decision: decision, graph_delta: { nodes_added: 0, edges_added: 0 } })

  it('names the experiment node and its EVALUATED_ON edge', () => {
    const records = expandMutations(mutate(trial()), 'credit_card_fraud')

    expect(records[0]).toMatchObject({ kind: 'node', label: 'Experiment', detail: 'trial_0001' })
    expect(records[1]).toMatchObject({ kind: 'edge', label: 'EVALUATED_ON', detail: '→ credit_card_fraud' })
  })

  it('names a FAILED_UNDER edge only when the worker actually failed', () => {
    const ok = expandMutations(mutate(trial()), 'credit_card_fraud')
    const failed = expandMutations(
      mutate(trial({ status: 'FAILED', failure_reason: 'OOM', metrics: { val_pr_auc: null } })),
      'credit_card_fraud',
    )

    expect(ok.some((record) => record.label === 'FAILED_UNDER')).toBe(false)
    expect(failed.some((record) => record.label === 'FAILED_UNDER')).toBe(true)
  })

  it('reports a residual rather than implying nothing else was written', () => {
    const records = expandMutations(mutate(trial()), 'credit_card_fraud')

    // Documented set: experiment + hypothesis + two metrics = 4 nodes, 4 edges.
    const nodes = records.filter((record) => record.kind === 'node' && !record.residual)
    const edges = records.filter((record) => record.kind === 'edge' && !record.residual)
    expect(nodes).toHaveLength(4)
    expect(edges).toHaveLength(4)

    const overReported = frame({
      stage: 'MUTATING_GRAPH',
      trial_result: trial(),
      laya_decision: DECISION,
      graph_delta: { nodes_added: 9, edges_added: 4 },
    })
    const residual = expandMutations(overReported, 'credit_card_fraud').filter((record) => record.residual)
    expect(residual).toHaveLength(1)
    expect(residual[0].detail).toContain('5')
  })
})

describe('reconcileFromStatus', () => {
  const snapshot = (overrides: Partial<RuntimeStatusResponse> = {}): RuntimeStatusResponse => ({
    run_id: 'run_1',
    status: 'RUNNING',
    objective: OBJECTIVE,
    best_score: 0.82,
    best_hyperparameters: { max_depth: 4 },
    completed_trials: [],
    execution_tree: [],
    graph_mutation_stats: { nodes_added: 4, edges_added: 4 },
    ...overrides,
  })

  it('keeps stream-only detail that the snapshot cannot supply', () => {
    const streamed = foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1)
    const merged = reconcileFromStatus(streamed, snapshot())

    // The snapshot's rows carry no metric map; the durations the stream delivered
    // must survive the merge or the ablation loses its compute figure.
    expect(merged.nodes[0].metrics.train_time_sec).toBe(0.4)
    expect(merged.nodes[0].citedPaths).toEqual(['hyp_cand_001'])
  })

  it('restores trials recorded before the stream attached, without replaying their writes', () => {
    const merged = reconcileFromStatus(initialRuntimeState(OBJECTIVE, 'run_1'), snapshot({
      execution_tree: [
        {
          step: 1,
          trial_id: 'trial_0001',
          candidate_id: 'cand_001',
          decision_type: 'explore_boundary',
          hypothesis: 'Probe the depth boundary.',
          score: 0.82,
          status: 'SUCCESS',
          hyperparameters: { max_depth: 4 },
          graph_delta: { nodes_added: 4, edges_added: 4 },
        },
      ],
    }))

    expect(merged.nodes).toHaveLength(1)
    expect(merged.totals.nodes_added).toBe(4)
    expect(merged.seenEvents).toContain('1-MUTATING_GRAPH')

    // Replaying the stream's backfill of that same step must not double the totals.
    const replayed = applyEvent(merged, frame({ step_number: 1, stage: 'MUTATING_GRAPH', trial_result: trial(), graph_delta: { nodes_added: 4, edges_added: 4 } }))
    expect(replayed.totals.nodes_added).toBe(4)
  })

  it('does not drop a trial the snapshot has not caught up with yet', () => {
    const streamed = foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1)
    const merged = reconcileFromStatus(streamed, snapshot({ execution_tree: [] }))

    expect(merged.nodes.map((node) => node.trialId)).toEqual(['trial_0001'])
  })

  it('reports a failed run as an error rather than a completed one', () => {
    const merged = reconcileFromStatus(initialRuntimeState(OBJECTIVE, 'run_1'), snapshot({ status: 'FAILED' }))

    expect(merged.phase).toBe('error')
    expect(merged.error).toContain('reported a failure')
  })
})

describe('trialsToReach', () => {
  const curve = [
    { trial: 1, score: 0.5, best_score: 0.5 },
    { trial: 2, score: 0.7, best_score: 0.7 },
    { trial: 3, score: 0.6, best_score: 0.7 },
  ]

  it('reads a falling best-so-far as converged for a minimised metric', () => {
    const falling = [
      { trial: 1, score: 0.6, best_score: 0.6 },
      { trial: 2, score: 0.3, best_score: 0.3 },
      { trial: 3, score: 0.4, best_score: 0.3 },
    ]

    expect(trialsToReach(curve, 0.7)).toBe(2)
    expect(trialsToReach(falling, 0.3, true)).toBe(2)
    /* Reading the same falling curve as a maximised metric claims the run hit the
       target on its very first trial, because 0.6 clears 0.3. That inversion is
       exactly what assuming "higher is better" would put in the scoreboard. */
    expect(trialsToReach(falling, 0.3, false)).toBe(1)
  })

  it('returns null rather than zero when the target was never reached', () => {
    expect(trialsToReach(curve, 0.99)).toBeNull()
    expect(trialsToReach(undefined, 0.7)).toBeNull()
  })
})

describe('meanTrialSeconds', () => {
  it('averages only the trials that recorded a positive duration', () => {
    const state = foldStep(initialRuntimeState(OBJECTIVE, 'run_1'), 1)
    const withTiming = { ...state.nodes[0], metrics: { val_pr_auc: 0.8, train_time_sec: 0.4 } }
    const without = { ...withTiming, metrics: { val_pr_auc: 0.8, train_time_sec: null } }

    expect(meanTrialSeconds([withTiming, without])).toBe(0.4)
  })

  it('returns null when nothing recorded a duration, so nothing is priced from nothing', () => {
    expect(meanTrialSeconds([])).toBeNull()
    expect(meanTrialSeconds([{ ...trial(), metrics: {} } as never])).toBeNull()
  })
})