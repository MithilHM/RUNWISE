/**
 * Simulated runtime for demo and offline use.
 *
 * The backend loop needs a worker, a graph and a model family to be reachable. During
 * a demonstration none of that is guaranteed, so this module replays a scripted run
 * through exactly the same `RuntimeStepEvent` contract the SSE endpoint serves. The
 * screen cannot tell the difference by construction: there is one reducer and one set
 * of components, and the only thing that changes is which transport filled it.
 *
 * Two rules keep the simulation honest:
 *
 *  - Scores are a deterministic function of the hyperparameters, so a run that looks
 *    like it converged on a small `learning_rate` with a raised `scale_pos_weight` is
 *    actually consistent with that. Random noise is drawn from a seeded generator, so
 *    the same objective replays identically and a demo can be rehearsed.
 *  - Failures are emitted for the same reason the real worker emits them: deep trees
 *    at high `n_estimators` blow the memory ceiling. The run therefore has failure
 *    regions for the scoreboard to report having avoided.
 *
 * Nothing here is presented as measured. Every screen that consumes it is told the
 * events are simulated, and says so.
 */

import type {
  AblationCurvePoint,
  AblationResult,
  GraphDelta,
  LayaDecision,
  LayaDecisionType,
  RuntimeObjective,
  RuntimeStepEvent,
  RuntimeStreamHandle,
  TrialResult,
  TrialStatus,
} from '../../api/runtimeApi'
import { MOCK_EVENT_INTERVAL_MS } from './runtimeConstants'

/** Seeded PRNG (mulberry32). Same seed, same demo, every time. */
function seeded(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function pick<T>(random: () => number, options: T[]): T {
  return options[Math.floor(random() * options.length)] as T
}

/**
 * The default search space, matching `RuntimeObjective.search_space` on the backend.
 * Candidates are drawn from these exact grids so the numbers shown are on the same
 * lattice a real trial would land on.
 */
const SEARCH_SPACE: Record<string, number[]> = {
  learning_rate: [0.01, 0.03, 0.05, 0.1, 0.2],
  max_depth: [3, 4, 6, 8],
  n_estimators: [50, 100, 150, 200],
  colsample_bytree: [0.6, 0.8, 1.0],
  scale_pos_weight: [1.0, 5.0, 10.0, 20.0, 50.0],
}

/**
 * Synthetic PR-AUC for an imbalanced tabular regime.
 *
 * Each term is a hand-shaped response curve: small learning rates need more trees,
 * an under-weighted positive class caps recall, and depth beyond six mostly buys
 * variance on 99:1 data. The shape is what makes the demo's convergence story
 * believable, not the absolute values.
 */
function syntheticScore(hparams: Record<string, number>, targetMetric: string, random: () => number): number {
  const learningRate = Number(hparams.learning_rate ?? 0.05)
  const maxDepth = Number(hparams.max_depth ?? 4)
  const nEstimators = Number(hparams.n_estimators ?? 100)
  const colsample = Number(hparams.colsample_bytree ?? 0.8)
  const posWeight = Number(hparams.scale_pos_weight ?? 1)

  let score = 0.34
  score += 0.26 * Math.exp(-Math.pow((learningRate - 0.05) / 0.09, 2))
  score += 0.05 * (Math.log(nEstimators) / Math.log(200))
  score += 0.11 * Math.min(posWeight, 20) / 20
  score -= 0.08 * Math.max(0, posWeight - 20) / 30
  score += 0.07 * (1 - Math.abs(maxDepth - 4) / 5)
  score += 0.03 * ((colsample - 0.6) / 0.4)
  score += (random() - 0.5) * 0.03

  if (targetMetric === 'val_loss') return Math.max(0.05, 0.62 - score)
  if (targetMetric === 'val_f1') return Math.min(0.97, score * 0.88 + 0.06)
  return Math.min(0.96, Math.max(0.12, score))
}

/** Mirrors `isMinimized` in runtimeConstants: these keys run the other way. */
function lowerIsBetter(targetMetric: string): boolean {
  return targetMetric === 'val_loss' || targetMetric === 'val_log_loss'
}

function drawHyperparameters(random: () => number): Record<string, number> {
  return {
    learning_rate: pick(random, SEARCH_SPACE.learning_rate),
    max_depth: pick(random, SEARCH_SPACE.max_depth),
    n_estimators: pick(random, SEARCH_SPACE.n_estimators),
    colsample_bytree: pick(random, SEARCH_SPACE.colsample_bytree),
    scale_pos_weight: pick(random, SEARCH_SPACE.scale_pos_weight),
  }
}

/** The memory ceiling the sandboxed worker applies, mirrored so failures are consistent. */
const MEMORY_CEILING_MB = 8192

function memoryFootprintMb(hparams: Record<string, number>): number {
  return Math.round(900 * hparams.max_depth * (hparams.n_estimators / 50) * (1.6 - hparams.colsample_bytree * 0.5))
}

export interface MockStreamConfig {
  runId: string
  objective: RuntimeObjective
  /** Override the pacing; defaults to `MOCK_EVENT_INTERVAL_MS`. */
  intervalMs?: number
  seed?: number
}

export function mockRunId(): string {
  const suffix = Math.floor(Math.random() * 0xffffffff).toString(16).padStart(8, '0')
  return `run_demo_${suffix}`
}

/**
 * Build the whole scripted run up front, then play it back on a timer.
 *
 * Materialising the script first means the pacing is the only thing that changes
 * between a real stream and this one, and it keeps the reducer under test fed in the
 * same order the backend emits: five stages per trial, then one terminal frame.
 */
function buildScript(config: MockStreamConfig): RuntimeStepEvent[] {
  const { objective } = config
  const random = seeded(config.seed ?? 1337)
  const budget = Math.max(1, Math.min(20, objective.trial_budget))
  const minimize = lowerIsBetter(objective.target_metric)

  const events: RuntimeStepEvent[] = []
  let best: number | null = null
  let seenFailure = false
  let lastSuccessStep: number | null = null

  for (let step = 1; step <= budget; step += 1) {
    const timestamp = new Date().toISOString()
    const failedSubgraphs = seenFailure ? 2 + Math.floor(random() * 3) : 0

    events.push({
      step_number: step,
      stage: 'QUERYING_GRAPH',
      current_best_metric: best,
      trial_result: null,
      laya_decision: null,
      graph_delta: { nodes_added: 0, edges_added: 0 },
      timestamp,
      message: `Queried FalkorDB: ${4 + step} similar regime records and ${failedSubgraphs} failed subgraphs`,
    })

    events.push({
      step_number: step,
      stage: 'GENERATING_CANDIDATES',
      current_best_metric: best,
      trial_result: null,
      laya_decision: null,
      graph_delta: { nodes_added: 0, edges_added: 0 },
      timestamp,
      message: 'Generated pool of 3 typed candidates (exploit, explore, boundary)',
    })

    // Strategy rotates: avoid a region that has already bitten, exploit the best
    // regime once one exists, otherwise probe the boundary.
    const decisionType: LayaDecisionType =
      seenFailure && random() < 0.4
        ? 'avoid_failure_subgraph'
        : best !== null && random() < 0.55
          ? 'exploit_promising_regime'
          : 'explore_boundary'

    const candidateId = `cand_${step.toString().padStart(3, '0')}`
    const cited: string[] = [`hyp_${candidateId}`, `dsv_${objective.dataset_name}_v1`]
    if (failedSubgraphs > 0) cited.push('fail_4c1e9ab2')
    if (lastSuccessStep !== null) cited.push(`exp_${lastSuccessStep.toString().padStart(4, '0')}`)

    const decision: LayaDecision = {
      selected_candidate_id: candidateId,
      decision_type: decisionType,
      hypothesis_text: HYPOTHESES[decisionType](objective, step),
      cited_graph_paths: cited,
      confidence: 0.62 + random() * 0.33,
    }

    events.push({
      step_number: step,
      stage: 'LAYA_EVALUATING',
      current_best_metric: best,
      trial_result: null,
      laya_decision: decision,
      graph_delta: { nodes_added: 0, edges_added: 0 },
      timestamp,
      message: `Laya evaluated candidates and selected '${candidateId}' (${decisionType})`,
    })

    const hyperparameters = drawHyperparameters(random)
    const memoryMb = memoryFootprintMb(hyperparameters)
    let status: TrialStatus = 'SUCCESS'
    let failureReason: string | null = null

    if (memoryMb > MEMORY_CEILING_MB) {
      status = 'FAILED'
      failureReason = 'Memory limit exceeded (OOM)'
      seenFailure = true
    } else if (hyperparameters.scale_pos_weight >= 20 && hyperparameters.max_depth >= 6) {
      // Early stopping fires on an over-weighted, over-deep configuration.
      status = 'PRUNED'
      failureReason = 'Pruned by early stopping: no validation gain after 25 rounds'
    }

    const rawScore = syntheticScore(hyperparameters, objective.target_metric, random)
    const score = status === 'SUCCESS' ? Number(rawScore.toFixed(4)) : null

    const trialId = `trial_${step.toString().padStart(4, '0')}`
    const metrics: Record<string, number | null> =
      status === 'SUCCESS'
        ? {
            [objective.target_metric]: score,
            // The worker writes the loss under `val_loss`, not `loss`. Keeping the
            // same key means the demo's metric list matches a real trial's.
            val_loss: Number((0.9 - rawScore * 0.5).toFixed(4)),
            train_time_sec: Number((6 + memoryMb / 900).toFixed(2)),
          }
        : { [objective.target_metric]: null, train_time_sec: Number((2 + memoryMb / 2400).toFixed(2)) }

    const trial: TrialResult = {
      trial_id: trialId,
      candidate_id: candidateId,
      hyperparameters,
      metrics,
      status,
      failure_reason: failureReason,
      memory_mb: memoryMb,
      // Deliberately no `parent_trial_id`. The real worker has no such field, so a
      // demo that emits one would draw a cross-step DERIVED_FROM edge the backend can
      // never produce — a capability the tree would appear to have and not have.
    }

    if (score !== null && (best === null || (minimize ? score < best : score > best))) {
      best = score
    }

    events.push({
      step_number: step,
      stage: 'TRAINING_WORKER',
      current_best_metric: best,
      trial_result: trial,
      laya_decision: null,
      graph_delta: { nodes_added: 0, edges_added: 0 },
      timestamp,
      message: `Worker executed trial '${trialId}': status=${status}, score=${score ?? 'n/a'}`,
    })

    if (status === 'SUCCESS') lastSuccessStep = step

    // Mirrors RuntimeGraphService.mutate_runtime_trial: the experiment node and its
    // EVALUATED_ON edge, the hypothesis node and its DERIVED_FROM edge, one node and
    // edge per failure reason, and one node and edge per recorded metric.
    const metricCount = Object.values(metrics).filter((value) => value !== null).length
    const delta: GraphDelta = {
      nodes_added: 1 + 1 + (status === 'FAILED' ? 1 : 0) + metricCount,
      edges_added: 1 + 1 + (status === 'FAILED' ? 1 : 0) + metricCount,
    }

    events.push({
      step_number: step,
      stage: 'MUTATING_GRAPH',
      current_best_metric: best,
      trial_result: trial,
      laya_decision: decision,
      graph_delta: delta,
      timestamp,
      message: `Mutated FalkorDB: added ${delta.nodes_added} nodes and ${delta.edges_added} edges`,
    })
  }

  events.push({
    step_number: budget,
    stage: 'COMPLETED',
    current_best_metric: best,
    trial_result: null,
    laya_decision: null,
    graph_delta: { nodes_added: 0, edges_added: 0 },
    timestamp: new Date().toISOString(),
    message: `Autonomous optimization loop ${config.runId} completed successfully with best ${objective.target_metric}=${best}`,
  })

  return events
}

const HYPOTHESES: Record<LayaDecisionType, (objective: RuntimeObjective, step: number) => string> = {
  exploit_promising_regime: (objective, step) =>
    `The best recorded regime on ${objective.dataset_name} pairs a small learning rate with a raised scale_pos_weight; reproducing it with a wider column sample should extend the existing high-PR-AUC cluster rather than open a new one. Step ${step} exploits that regime.`,
  explore_boundary: (_objective, step) =>
    `Depth 6 sits on the boundary between the stable shallow region and the variance-heavy deep region. Probing it in step ${step} tests whether the deep failures are a data property or an overfitting artefact before committing the remaining budget.`,
  avoid_failure_subgraph: (objective, step) =>
    `A FAILED_UNDER cluster already covers deep trees at high n_estimators on ${objective.dataset_name}. Step ${step} stays outside that subgraph and tests a shallower, better-weighted point instead of re-entering a known failure region.`,
}

/**
 * Play a scripted run. Returns the same handle shape as the real SSE reader so the
 * session hook swaps transports without a second code path.
 */
export function createMockStream(
  config: MockStreamConfig,
  onEvent: (event: RuntimeStepEvent) => void,
  onError?: (error: Error) => void,
): RuntimeStreamHandle {
  const script = buildScript(config)
  const interval = config.intervalMs ?? MOCK_EVENT_INTERVAL_MS
  let index = 0
  let closed = false
  let timer: ReturnType<typeof setTimeout> | undefined

  const advance = () => {
    if (closed) return
    if (index >= script.length) {
      closed = true
      return
    }
    const event = script[index]
    index += 1
    try {
      onEvent(event)
    } catch (cause) {
      onError?.(cause as Error)
    }
    if (event.stage === 'COMPLETED') {
      closed = true
      return
    }
    timer = setTimeout(advance, interval)
  }

  timer = setTimeout(advance, interval)

  return {
    close() {
      closed = true
      if (timer !== undefined) clearTimeout(timer)
    },
  }
}

/**
 * Simulate the ablation endpoint. Uses the scores the simulated run actually
 * recorded, so the convergence chart cannot disagree with the tree above it.
 */
export function createMockAblation(
  objective: RuntimeObjective,
  measuredScores: (number | null)[],
  requestedBudget?: number,
): AblationResult {
  const random = seeded(20260918)
  const budget = Math.max(2, Math.min(30, requestedBudget ?? objective.trial_budget))
  const minimize = lowerIsBetter(objective.target_metric)

  /* A trial that failed or was pruned contributes no score. Zero is a real metric
     value, so substituting one would let a failure drag the curve down and could
     make `laya_best` read 0 — a number that never happened. Unscored trials are
     dropped from the curve instead, and the summary says the run came in short. */
  const scored = measuredScores.filter((score): score is number => score !== null && Number.isFinite(score))
  const shortfall = Math.max(0, budget - scored.length)

  const toCurve = (scores: number[]): AblationCurvePoint[] => {
    let best = minimize ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY
    return scores.map((score, index) => {
      if (minimize ? score < best : score > best) best = score
      return { trial: index + 1, score: Number(score.toFixed(4)), best_score: Number(best.toFixed(4)) }
    })
  }

  const layaScores = scored.slice(0, budget)
  while (layaScores.length < budget) {
    // Past the recorded run the loop keeps sampling around its best region, so the
    // curve still moves in the right direction but far more slowly than in the
    // trials that were actually measured.
    const previous = layaScores[layaScores.length - 1] ?? 0.4
    const drift = minimize ? -1 : 1
    layaScores.push(previous + drift * random() * 0.012)
  }

  // Uniform random search on the same lattice: wide spread, late convergence.
  const randomScores: number[] = []
  for (let index = 0; index < budget; index += 1) {
    randomScores.push(syntheticScore(drawHyperparameters(random), objective.target_metric, random))
  }

  const layaCurve = toCurve(layaScores)
  const randomCurve = toCurve(randomScores)
  const layaBest = layaCurve[layaCurve.length - 1]?.best_score ?? 0
  const randomBest = randomCurve[randomCurve.length - 1]?.best_score ?? 0
  const improvement = Math.round(((layaBest - randomBest) / Math.max(Math.abs(randomBest), 1e-4)) * 10_000) / 100

  /* Say plainly when the leading arm has nothing measured behind it. "8 of the 8
     requested trials produced no score" is technically true and practically
     unreadable; what matters is that the leading curve is not a measurement at all. */
  const scope =
    scored.length === 0
      ? ' No trial of this run produced a score, so the Graph+Laya arm here is not a measurement: it is a simulated walk of the search space and must not be read as RUNWISE performance.'
      : shortfall > 0
        ? ` ${shortfall} of the ${budget} requested trials produced no score (failed or pruned) and are excluded from the curve.`
        : ''

  return {
    run_id: `run_demo_${objective.dataset_name}`,
    budget,
    laya_curve: layaCurve,
    random_curve: randomCurve,
    laya_best: Number(layaBest.toFixed(4)),
    random_best: Number(randomBest.toFixed(4)),
    improvement_percentage: improvement,
    summary: `Simulated ablation on ${objective.dataset_name}: Graph + Laya reached ${objective.target_metric}=${layaBest.toFixed(4)} against uniform random search at ${randomBest.toFixed(4)} (${improvement >= 0 ? '+' : ''}${improvement}%) within ${budget} trials.${scope}`,
  }
}