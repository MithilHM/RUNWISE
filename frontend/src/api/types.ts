/**
 * Response types for the RUNWISE backend.
 *
 * These mirror `backend/app/models/contracts.py`. They are hand-written rather than
 * generated so the file can carry the semantics the UI depends on (which fields may
 * be null, which enum values are legal), but every field is checked against the live
 * OpenAPI schema by `scripts/verify_api_contract.py`; a backend rename fails that
 * check rather than silently becoming `undefined` in the UI.
 *
 * Contract notes that matter when rendering:
 *  - `evidence_status` is an epistemic grade, not a boolean. Never collapse it.
 *  - `score` is a weighted heuristic in [0, 1]. It is not a probability.
 *  - `estimated_gpu_hours` is nullable: an unknown estimate is unrecorded, not zero.
 *  - `recommendations: []` is a valid answer whose reason lives in `uncertainties`.
 */

export type MetricDirection = 'maximize' | 'minimize'
export type ExperimentStatus = 'completed' | 'failed' | 'running' | 'cancelled' | 'unknown'
export type EvidenceStatus =
  | 'observed'
  | 'user_asserted'
  | 'inferred'
  | 'confirmed'
  | 'disputed'
  | 'superseded'
  | 'unresolved'
export type ReviewItemStatus = 'open' | 'resolved' | 'dismissed'
export type ComparabilityStatus =
  | 'COMPARABLE'
  | 'COMPARABLE_WITH_CAVEATS'
  | 'NOT_COMPARABLE'
  | 'INSUFFICIENT_INFORMATION'
export type Severity = 'blocker' | 'major' | 'minor' | 'info'
export type AnswerMode = 'deterministic' | 'llm'
export type RecommendationStatus = 'proposed' | 'approved' | 'rejected' | 'edited' | 'superseded'
export type Decision = 'approved' | 'rejected' | 'edited'
export type CandidateKind =
  | 'sweep_parameter'
  | 'try_dataset_version'
  | 'new_model_family'
  | 'replicate_seed'
  | 'evaluate_missing_metric'

/** One error object. Every endpoint returns these under `detail`. */
export interface ErrorDetail {
  code: string
  message: string
  field: string | null
}

export interface MetricValue {
  name: string
  value: number | null
  direction: MetricDirection
  evaluation_split: string | null
  evaluation_protocol: string | null
}

export interface ProjectResponse {
  id: string
  name: string
  description: string | null
  created_at: string | null
  experiment_count: number
}

export interface ExperimentSummary {
  id: string
  project_id: string
  name: string
  /** `null` when the source never stated a family. The backend does not invent one. */
  model_family: string | null
  dataset_version_id: string | null
  status: ExperimentStatus
  metrics: MetricValue[]
  compute_cost_gpu_hours: number | null
  created_at: string | null
  duration_seconds: number | null
  quality_status: string | null
}

export interface ExperimentDetail extends ExperimentSummary {
  description: string | null
  configuration: Record<string, unknown> | null
  dataset_name: string | null
  dataset_version: string | null
  random_seed: number | null
  source_reference: string | null
  notes: string | null
  parent_experiment_ids: string[]
  child_experiment_ids: string[]
  claims: Record<string, unknown>[]
  review_items: Record<string, unknown>[]
}

export interface LineageNode {
  id: string
  name: string
  label: string
  model_family: string | null
  status: string | null
  quality_status: string | null
  dataset_version_id: string | null
  depth: number
  is_root: boolean
}

export interface LineageEdge {
  source: string
  target: string
  relationship_type: string
}

export interface ExperimentLineageResponse {
  root_id: string
  nodes: LineageNode[]
  edges: LineageEdge[]
}

export type RelatedExperiments = Record<string, string[]>

export interface ComparabilityReason {
  code: string
  dimension: string
  severity: Severity
  message: string
  /** The values compared, or null when that side recorded nothing. */
  left_value: unknown | null
  right_value: unknown | null
}

export interface ComparabilityAssessment {
  status: ComparabilityStatus
  reasons: ComparabilityReason[]
  matched_dimensions: string[]
  differing_dimensions: string[]
  missing_dimensions: string[]
  score: number
  score_kind: string
  confidence: string
  left_experiment_id: string | null
  right_experiment_id: string | null
  explanation: string
}

export interface GraphNodeData {
  id: string
  label: string
  properties: Record<string, unknown>
}

export interface GraphEdgeData {
  source: string
  target: string
  relationship_type: string
  properties: Record<string, unknown>
}

export interface ProjectGraphResponse {
  project_id: string
  nodes: GraphNodeData[]
  edges: GraphEdgeData[]
  truncated: boolean
  node_limit: number
}

export interface ProjectGraphSummary {
  project_id: string
  node_counts: Record<string, number>
  relationship_counts: Record<string, number>
}

export interface ImportErrorItem {
  row: number | null
  experiment_id: string | null
  reason: string
}

export interface ImportSummary {
  imported: number
  skipped: number
  failed: number
  errors: ImportErrorItem[]
}

export interface ScoreComponent {
  name: string
  weight: number
  value: number
  weighted_value: number
  rationale: string
}

export interface ValidationCriterion {
  kind: string
  description: string
  is_suggested_default: boolean
  threshold: number | null
  metric_name: string | null
}

export interface Candidate {
  kind: CandidateKind
  hypothesis: string
  configuration: Record<string, unknown>
  independent_variables: string[]
  controlled_variables: string[]
  metrics: string[]
  baseline_experiment_ids: string[]
  supporting_experiment_ids: string[]
  estimated_gpu_hours: number | null
  estimate_source: string | null
  uncertainties: string[]
  validation_criteria: ValidationCriterion[]
  model_family: string | null
  dataset_version_id: string | null
  independent_variable_values: Record<string, unknown>
}

export interface ExcludedEvidence {
  experiment_id: string
  reason: string
  detail: string
  status: EvidenceStatus
}

export interface Recommendation {
  id: string
  project_id: string
  objective: string
  hypothesis: string
  candidate_kind: CandidateKind
  baseline_experiment_ids: string[]
  configuration: Record<string, unknown>
  independent_variables: string[]
  controlled_variables: string[]
  metrics: string[]
  estimated_gpu_hours: number | null
  estimate_source: string | null
  supporting_experiment_ids: string[]
  independent_variable_values: Record<string, unknown>
  uncertainties: string[]
  validation_criteria: ValidationCriterion[]
  rationale: string
  score: number
  score_kind: string
  score_components: ScoreComponent[]
  evidence_status: EvidenceStatus
  excluded_evidence_ids: string[]
  status: RecommendationStatus
  created_at: string
}

export interface RecommendationSet {
  recommendations: Recommendation[]
  compute_budget_gpu_hours: number | null
  estimated_gpu_hours: number | null
  budget_exhausted: boolean
  unselected_candidates: Candidate[]
  considered_experiment_ids: string[]
  excluded_evidence: ExcludedEvidence[]
  uncertainties: string[]
  created_at: string
}

export interface EvidenceRef {
  experiment_id: string | null
  metric_name: string | null
  metric_value: number | null
  source_reference: string | null
  evidence_status: EvidenceStatus
  node_type: string
}

export interface Claim {
  id: string
  text: string
  status: string
  evidence_status: EvidenceStatus
  support: EvidenceRef[]
  detail: string | null
}

export interface GroundingReport {
  cited_experiment_ids: string[]
  unknown_citations: string[]
  available_experiment_count: number
  citation_coverage: number
  is_grounded: boolean
}

export interface ResearchAnswer {
  question: string
  answer: string
  mode: AnswerMode
  llm_used: boolean
  grounding: GroundingReport | null
  claims: Claim[]
  supporting_experiments: ExperimentSummary[]
  comparisons: ComparabilityAssessment[]
  recommendations: Recommendation[]
  corrections_considered: CorrectionImpact[]
  excluded_evidence: ExcludedEvidence[]
  comparable_experiment_ids: string[]
  uncertainties: string[]
  missing_information: string[]
  insufficient_evidence: boolean
  created_at: string
}

export interface ClaimPath {
  nodes: string[]
  relationship_types: string[]
}

export interface CorrectionImpact {
  correction_id: string
  source_id: string
  summary: string
  directly_affected_experiment_ids: string[]
  indirectly_affected_experiment_ids: string[]
  potentially_affected_claim_ids: string[]
  unaffected_experiment_ids: string[]
  review_item_ids: string[]
  paths: ClaimPath[]
}

export interface DependencyPath {
  nodes: string[]
  relationship_types: string[]
  reason: string | null
}

export interface CorrectionImpactResponse {
  correction_id: string
  source_id: string
  source_type: string | null
  summary: string
  directly_affected_experiment_ids: string[]
  indirectly_affected_experiment_ids: string[]
  potentially_affected_claim_ids: string[]
  unaffected_experiment_ids: string[]
  review_item_ids: string[]
  paths: DependencyPath[]
  generated_at: string | null
}

export interface ReviewItemResponse {
  id: string
  experiment_id: string | null
  reason: string
  status: ReviewItemStatus
  project_id: string | null
  correction_id: string | null
  created_at: string
  resolved_at: string | null
  resolved_by: string | null
  resolution_notes: string | null
}

export interface HealthResponse {
  status: string
  app: string
  database: {
    status: string
    host?: string
    port?: number
    graph?: string
    latency_ms?: number
    error?: string | null
  }
}

/** Query parameters accepted by the experiment history endpoint (FR-2). */
export interface ExperimentFilters {
  model_family?: string
  dataset_id?: string
  dataset_version_id?: string
  metric?: string
  evaluation_split?: string
  status?: string
  quality_status?: string
  config_field?: string
  config_value?: string
  created_after?: string
  created_before?: string
  limit?: number
  offset?: number
}

// ---------------------------------------------------------------------------
// Member B — Autonomous ML Harness Contracts (MEMBER_B.md Section 3.1 & 3.2)
// ---------------------------------------------------------------------------

export interface ColumnProfile {
  name: string
  dtype: string
  role: 'feature' | 'target'
  n_missing: number
  n_unique: number
}

export interface DroppedColumn {
  name: string
  reason: 'constant' | 'all_null' | 'high_cardinality_id' | 'leakage'
}

export interface DatasetProfile {
  id: string
  name: string
  n_rows: number
  n_cols: number
  task: 'binary' | 'multiclass'
  n_classes: number
  class_balance: Record<string, number>
  split_hash: string
  created_at: string
  target_inferred: boolean
  columns: ColumnProfile[]
  dropped_columns: DroppedColumn[]
}

export interface ParamSpec {
  name: string
  kind: 'int' | 'float' | 'categorical' | 'bool' | 'int_or_null'
  low?: number | null
  high?: number | null
  step?: number | null
  log_scale?: boolean
  choices?: any[] | null
  default: any
}

export interface ModelSpec {
  id: 'xgboost' | 'lightgbm' | 'random_forest' | string
  name: string
  library: string
  description: string
  params: ParamSpec[]
}

export interface RunCreate {
  dataset_id: string
  model_ids: string[]
  mode: 'autonomous' | 'human'
  metric: 'accuracy' | 'f1_macro' | 'roc_auc' | 'pr_auc'
  budget: number
}

export interface TrialOutcome {
  trial_id: string
  run_id: string
  step: number
  model_id: string
  params: Record<string, any>
  status: 'SUCCESS' | 'FAILED'
  metric_name: string
  metric_value: number | null
  metrics: Record<string, number | null>
  failure_reason: string | null
  duration_sec: number
  n_features: number
  seed: number
  created_at: string
}

export interface RunState {
  run_id: string
  dataset_id: string
  mode: 'autonomous' | 'human'
  status:
    | 'IDLE'
    | 'PROPOSING'
    | 'AWAITING_REVIEW'
    | 'EXECUTING'
    | 'MUTATING'
    | 'COMPLETED'
    | 'PAUSED'
    | 'FAILED'
  step: number
  budget: number
  metric_name: string
  best_metric?: number | null
  best_trial_id?: string | null
  model_ids: string[]
  created_at: string
  updated_at: string
  best_trial?: TrialOutcome | null
}

export interface Proposal {
  proposal_id: string
  step: number
  model_id: string
  params: Record<string, any>
  why: string
  strategy: string
  hypothesis: string
  confidence: number
  citations: string[]
}


export interface HarnessDecision {
  decision_id: string
  run_id: string
  step: number
  trial_id: string
  actor: 'laya' | 'researcher'
  strategy: string
  why: string
  hypothesis: string
  confidence: number
  citations: string[]
  created_at: string
}

export interface EvidenceItem {
  kind: 'Model' | 'Param' | 'Setting' | 'Trial' | 'Dataset'
  id: string
  label: string
  detail: string
}

export interface WhyRecord {
  decision: HarnessDecision
  config: Record<string, any>
  evidence: EvidenceItem[]
  highlight: string[]
}

export interface LogEntry {
  seq: number
  run_id: string
  step: number
  level: string
  stage: string
  message: string
  meta: Record<string, any>
  ts: string
}

export interface LogEvent {
  type: 'log' | 'decision' | 'trial' | 'graph_delta' | 'status'
  payload: Record<string, any>
}

export interface GraphNode {
  id: string
  label: string
  type: string
  props: Record<string, any>
}

export interface GraphEdge {
  source: string
  target: string
  type: string
  props: Record<string, any>
}

export interface GraphSnapshot {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

