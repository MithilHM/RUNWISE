/**
 * Shared vocabulary for the runtime screens.
 *
 * The backend speaks in machine enums (`LAYA_EVALUATING`, `avoid_failure_subgraph`,
 * `PRUNED`). This module is the single place those enums acquire a human label, an
 * order and a colour, so the tree, the reasoning feed and the scoreboard can never
 * disagree about what a state means.
 *
 * Kept out of `.tsx` on purpose: a module exporting both constants and components
 * trips the `react/only-export-components` rule, and these are data, not markup.
 */

import type { LayaDecisionType, RuntimeStage, TrialStatus } from '../../api/runtimeApi'

/** Pacing of the simulated stream. Three seconds per event, as the demo brief asks. */
export const MOCK_EVENT_INTERVAL_MS = 3_000

/** Time between automatic status reconciliation polls while a run is in flight. */
export const RUNTIME_RECONCILE_MS = 5_000

/** Longest ablation the server will accept (see the `budget` query constraint). */
export const ABLATION_MIN_BUDGET = 2
export const ABLATION_MAX_BUDGET = 30

/** Short phase names for the feed. The wire value alone reads as a log line, not a step. */
export const STAGE_LABELS: Record<RuntimeStage, string> = {
  QUERYING_GRAPH: 'Query graph',
  GENERATING_CANDIDATES: 'Generate candidates',
  LAYA_EVALUATING: 'Laya evaluates',
  TRAINING_WORKER: 'Train worker',
  MUTATING_GRAPH: 'Mutate graph',
  COMPLETED: 'Completed',
}

export interface DecisionMeta {
  /** Two-letter token for the compact tree badge. Exploit and Explore are `EX` and
   *  `EP` deliberately: both start with "E" and one letter cannot tell them apart. */
  token: string
  label: string
  /** The exploit/explore/avoid framing the scoreboard compares against. */
  posture: 'Exploit' | 'Explore' | 'Avoid'
  blurb: string
}

export const DECISION_META: Record<LayaDecisionType, DecisionMeta> = {
  exploit_promising_regime: {
    token: 'EX',
    label: 'Exploit promising regime',
    posture: 'Exploit',
    blurb: 'Commit the remaining budget to the best-scoring region already in the graph.',
  },
  explore_boundary: {
    token: 'EP',
    label: 'Explore boundary',
    posture: 'Explore',
    blurb: 'Probe the edge between a stable region and an uncharacterised one.',
  },
  avoid_failure_subgraph: {
    token: 'AV',
    label: 'Avoid failure subgraph',
    posture: 'Avoid',
    blurb: 'Stay outside a region the graph already records as failing.',
  },
}

export interface TrialStatusMeta {
  label: string
  /** Border colour for the tree node. */
  node: string
  /** Filled dot colour. */
  dot: string
  blurb: string
}

export const TRIAL_STATUS_META: Record<TrialStatus, TrialStatusMeta> = {
  SUCCESS: {
    label: 'Succeeded',
    node: 'border-good-400/60 bg-good-400/10',
    dot: 'bg-good-400',
    blurb: 'Trained and scored. Its metrics are in the graph.',
  },
  FAILED: {
    label: 'Failed',
    node: 'border-danger-400/60 bg-danger-400/10',
    dot: 'bg-danger-400',
    blurb: 'The worker raised. A FAILED_UNDER edge records why.',
  },
  PRUNED: {
    label: 'Pruned',
    node: 'border-danger-400/40 bg-danger-400/5',
    dot: 'bg-danger-400/60',
    // Not "no metric node": the worker still records `train_time_sec`, and that node
    // is written. Only the target-metric node is absent.
    blurb: 'Early stopping cut the run short. No target-metric node was written.',
  },
}

/** In-flight node treatment. Pulsing is the only animation in the tree. */
export const RUNNING_NODE = 'border-warn-400/70 bg-warn-400/10'

/**
 * Regime and metric choices.
 *
 * `value` is what the backend stores; `label` is what a researcher reads. PR-AUC is
 * listed first for the fraud regime because at 99:1 accuracy is uninformative, and
 * this screen should not offer a metric that cannot distinguish two candidates.
 *
 * Only maximise metrics are offered, and that is a deliberate constraint rather than
 * an oversight. Two backend behaviours make a minimised target unsafe to expose:
 *
 *  1. `SandboxedMLWorker` writes `val_loss` and, for any target metric it does not
 *     already have, stamps the PR-AUC value under whatever key was requested. A name
 *     the worker never computes would render PR-AUC wearing a log-loss label.
 *  2. `RuntimeLoop` promotes a trial to "best" with an unconditional
 *     `score > self.best_score`, for every metric. On a minimised metric that keeps
 *     the *worst* configuration, so the header, the tree badges and the ablation
 *     would all label it "best" while drawing a "lower is better" arrow.
 *
 * `isMinimized` still knows the direction, so readding `val_loss` is a one-line
 * change here once the loop compares in the right direction.
 */
export const DATASET_OPTIONS = [
  { value: 'credit_card_fraud', label: 'Fraud Detection (imbalanced 99:1)' },
  { value: 'bank_churn', label: 'Bank Churn (imbalanced 7:1)' },
] as const

export const MODEL_FAMILY_OPTIONS = [
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'xgboost', label: 'XGBoost' },
] as const

export const TARGET_METRIC_OPTIONS = [
  { value: 'val_pr_auc', label: 'PR-AUC', note: 'maximize · the metric that survives a 99:1 split' },
  { value: 'val_f1', label: 'F1', note: 'maximize · collapses minority recall into one number' },
] as const

export const DEFAULT_OBJECTIVE = {
  dataset_name: DATASET_OPTIONS[0].value,
  model_family: MODEL_FAMILY_OPTIONS[0].value,
  target_metric: TARGET_METRIC_OPTIONS[0].value,
  trial_budget: 8,
} as const

/** Metrics whose best value is the smallest. See the note on TARGET_METRIC_OPTIONS. */
const MINIMIZED_METRICS = new Set(['val_loss', 'val_log_loss'])

export function isMinimized(metric: string | null | undefined): boolean {
  return metric !== null && metric !== undefined && MINIMIZED_METRICS.has(metric)
}