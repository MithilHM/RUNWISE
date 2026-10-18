/**
 * The single place the frontend talks to the backend.
 *
 * One error shape, one base URL, no `fetch` calls in components. Every endpoint
 * returns `{"detail": [{"code", "message", "field"}]}` (see
 * `backend/app/api/errors.py`), including validation failures, so a caller can
 * always render `error.details[0].message` instead of guessing at a body.
 */

import type {
  ComparabilityAssessment,
  CorrectionImpactResponse,
  Decision,
  ErrorDetail,
  ExperimentDetail,
  ExperimentFilters,
  ExperimentLineageResponse,
  ExperimentSummary,
  HealthResponse,
  ImportSummary,
  ProjectGraphResponse,
  ProjectGraphSummary,
  ProjectResponse,
  Recommendation,
  RecommendationSet,
  RelatedExperiments,
  ResearchAnswer,
  ReviewItemResponse,
  ReviewItemStatus,
} from './types'

export class ApiError extends Error {
  readonly status: number
  readonly details: ErrorDetail[]

  constructor(status: number, details: ErrorDetail[]) {
    super(details[0]?.message ?? `Request failed with status ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.details = details
  }

  /** True when retrying later could plausibly work: database down, service absent. */
  get isTransient(): boolean {
    return this.status >= 500
  }

  get code(): string {
    return this.details[0]?.code ?? 'unknown_error'
  }
}

export const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

/** Normalise anything the server might return into the canonical detail list. */
function toDetails(status: number, body: unknown): ErrorDetail[] {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const item = d as Partial<ErrorDetail> & { msg?: string; loc?: unknown[] }
      return {
        code: item.code ?? 'error',
        message: item.message ?? item.msg ?? 'Request failed',
        field: item.field ?? null,
      }
    })
  }
  if (detail && typeof detail === 'object') {
    const item = detail as Partial<ErrorDetail>
    return [{ code: item.code ?? 'error', message: item.message ?? 'Request failed', field: item.field ?? null }]
  }
  if (typeof detail === 'string') {
    return [{ code: 'error', message: detail, field: null }]
  }
  return [{ code: 'error', message: `Request failed with status ${status}`, field: null }]
}

/**
 * Exported so a module with a non-JSON transport (the runtime SSE reader in
 * `runtimeApi`) shares one base URL and one error shape instead of forking them.
 */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
        ...init?.headers,
      },
    })
  } catch {
    throw new ApiError(0, [
      {
        code: 'network_error',
        message: 'The backend is unreachable. Start it with `uvicorn app.main:app` from backend/.',
        field: null,
      },
    ])
  }

  if (response.status === 204) {
    return undefined as T
  }

  const raw = await response.text()
  let parsed: unknown = null
  if (raw) {
    try {
      parsed = JSON.parse(raw)
    } catch {
      parsed = null
    }
  }

  if (!response.ok) {
    throw new ApiError(response.status, toDetails(response.status, parsed))
  }
  return parsed as T
}

/** Build a query string, dropping absent parameters rather than sending `undefined`. */
export function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const encoded = search.toString()
  return encoded ? `?${encoded}` : ''
}

export interface RecommendationRequest {
  objective: string
  compute_budget_gpu_hours?: number
  max_recommendations?: number
  target_metric?: string
  model_family?: string
  dataset_id?: string
  include_disputed_evidence?: boolean
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  listProjects: () => request<ProjectResponse[]>('/api/v1/projects'),

  getProject: (projectId: string) => request<ProjectResponse>(`/api/v1/projects/${projectId}`),

  listExperiments: (projectId: string, filters: ExperimentFilters = {}) => {
    const { limit, offset, ...rest } = filters
    return request<ExperimentSummary[]>(
      `/api/v1/projects/${projectId}/experiments${query({ ...rest, limit, offset })}`,
    )
  },

  getExperiment: (experimentId: string, projectId?: string) =>
    request<ExperimentDetail>(`/api/v1/experiments/${experimentId}${query({ project_id: projectId })}`),

  getComparability: (experimentId: string, params: { compare_experiment_id?: string; metric_name?: string; project_id?: string } = {}) =>
    request<ComparabilityAssessment>(`/api/v1/experiments/${experimentId}/comparability${query(params)}`),

  getLineage: (experimentId: string, params: { max_depth?: number; project_id?: string } = {}) =>
    request<ExperimentLineageResponse>(`/api/v1/experiments/${experimentId}/lineage${query(params)}`),

  getRelated: (experimentId: string, projectId?: string) =>
    request<RelatedExperiments>(`/api/v1/experiments/${experimentId}/related${query({ project_id: projectId })}`),

  getGraphSummary: (projectId: string) =>
    request<ProjectGraphSummary>(`/api/v1/projects/${projectId}/graph/summary`),

  getGraph: (projectId: string, limit = 500) =>
    request<ProjectGraphResponse>(`/api/v1/projects/${projectId}/graph${query({ limit })}`),

  importJson: (projectId: string, payload: unknown) =>
    request<ImportSummary>(`/api/v1/projects/${projectId}/experiments/import`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  importFile: async (projectId: string, file: File): Promise<ImportSummary> => {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(`${BASE_URL}/api/v1/projects/${projectId}/experiments/import`, {
      method: 'POST',
      body: form,
    })
    const body = await response.json().catch(() => null)
    if (!response.ok) {
      throw new ApiError(response.status, toDetails(response.status, body))
    }
    return body as ImportSummary
  },

  createRecommendation: (projectId: string, payload: RecommendationRequest) =>
    request<RecommendationSet>(`/api/v1/projects/${projectId}/recommendations`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listRecommendations: (projectId: string) =>
    request<Recommendation[]>(`/api/v1/projects/${projectId}/recommendations`),

  getRecommendation: (recommendationId: string) =>
    request<Recommendation>(`/api/v1/recommendations/${recommendationId}`),

  decideRecommendation: (
    recommendationId: string,
    payload: { decision: Decision; decided_by: string; note?: string; edited_configuration?: Record<string, unknown> },
  ) =>
    request<Recommendation>(`/api/v1/recommendations/${recommendationId}/decision`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  query: (projectId: string, payload: { question: string; compute_budget_gpu_hours?: number; max_recommendations?: number }) =>
    request<ResearchAnswer>(`/api/v1/projects/${projectId}/query`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  createCorrection: (
    projectId: string,
    payload: { source_id: string; source_type?: string; description: string; created_by?: string },
  ) =>
    request<CorrectionImpactResponse>(`/api/v1/projects/${projectId}/corrections`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  getCorrectionImpact: (correctionId: string) =>
    request<CorrectionImpactResponse>(`/api/v1/corrections/${correctionId}/impact`),

  getReviewQueue: (projectId: string) =>
    request<ReviewItemResponse[]>(`/api/v1/projects/${projectId}/review-queue`),

  resolveReviewItem: (
    reviewItemId: string,
    payload: { status: ReviewItemStatus; notes?: string; resolved_by?: string },
  ) =>
    request<ReviewItemResponse>(`/api/v1/review-items/${reviewItemId}/resolve`, {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
}
