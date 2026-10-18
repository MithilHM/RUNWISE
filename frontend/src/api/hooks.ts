/** React Query bindings. Components never call `api` directly. */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from '@tanstack/react-query'
import { api, type RecommendationRequest } from './client'
import type { Decision, ReviewItemStatus } from './types'

export const keys = {
  health: ['health'] as const,
  projects: ['projects'] as const,
  project: (id: string) => ['project', id] as const,
  experiments: (id: string, filters: unknown) => ['experiments', id, filters] as const,
  experiment: (id: string) => ['experiment', id] as const,
  lineage: (id: string) => ['lineage', id] as const,
  related: (id: string) => ['related', id] as const,
  comparability: (id: string, peer?: string) => ['comparability', id, peer] as const,
  graph: (id: string) => ['graph', id] as const,
  graphSummary: (id: string) => ['graphSummary', id] as const,
  reviewQueue: (id: string) => ['reviewQueue', id] as const,
  recommendations: (id: string) => ['recommendations', id] as const,
}

export const useHealth = (): UseQueryResult<Awaited<ReturnType<typeof api.health>>> =>
  useQuery({ queryKey: keys.health, queryFn: api.health, retry: false, refetchInterval: 30_000 })

export const useProjects = () => useQuery({ queryKey: keys.projects, queryFn: api.listProjects })

export const useExperiments = (projectId: string | null, filters: Record<string, unknown> = {}) =>
  useQuery({
    queryKey: keys.experiments(projectId ?? '', filters),
    queryFn: () => api.listExperiments(projectId as string, filters),
    enabled: Boolean(projectId),
  })

export const useExperiment = (experimentId: string | null, projectId?: string) =>
  useQuery({
    queryKey: [...keys.experiment(experimentId ?? ''), projectId],
    queryFn: () => api.getExperiment(experimentId as string, projectId),
    enabled: Boolean(experimentId),
  })

export const useLineage = (experimentId: string | null, projectId?: string, maxDepth = 5) =>
  useQuery({
    queryKey: [...keys.lineage(experimentId ?? ''), projectId, maxDepth],
    queryFn: () => api.getLineage(experimentId as string, { max_depth: maxDepth, project_id: projectId }),
    enabled: Boolean(experimentId),
  })

export const useRelated = (experimentId: string | null, projectId?: string) =>
  useQuery({
    queryKey: [...keys.related(experimentId ?? ''), projectId],
    queryFn: () => api.getRelated(experimentId as string, projectId),
    enabled: Boolean(experimentId),
  })

export const useComparability = (
  experimentId: string | null,
  options: { peer?: string; metricName?: string; projectId?: string } = {},
) =>
  useQuery({
    queryKey: keys.comparability(experimentId ?? '', options.peer),
    queryFn: () =>
      api.getComparability(experimentId as string, {
        compare_experiment_id: options.peer,
        metric_name: options.metricName,
        project_id: options.projectId,
      }),
    enabled: Boolean(experimentId),
  })

export const useGraph = (projectId: string | null, limit = 500) =>
  useQuery({
    queryKey: keys.graph(projectId ?? ''),
    queryFn: () => api.getGraph(projectId as string, limit),
    enabled: Boolean(projectId),
  })

export const useGraphSummary = (projectId: string | null) =>
  useQuery({
    queryKey: keys.graphSummary(projectId ?? ''),
    queryFn: () => api.getGraphSummary(projectId as string),
    enabled: Boolean(projectId),
  })

export const useReviewQueue = (projectId: string | null) =>
  useQuery({
    queryKey: keys.reviewQueue(projectId ?? ''),
    queryFn: () => api.getReviewQueue(projectId as string),
    enabled: Boolean(projectId),
  })

export const useImportFile = (projectId: string | null) => {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => api.importFile(projectId as string, file),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.experiments(projectId ?? '', {}) })
      void client.invalidateQueries({ queryKey: keys.projects })
    },
  })
}

export const useCopilotQuery = (projectId: string | null) =>
  useMutation({
    mutationFn: (payload: { question: string; compute_budget_gpu_hours?: number }) =>
      api.query(projectId as string, payload),
  })

export const useRecommendationRequest = (projectId: string | null) =>
  useMutation({
    mutationFn: (payload: RecommendationRequest) => api.createRecommendation(projectId as string, payload),
  })

export const useRecommendationDecision = () => {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (payload: { id: string; decision: Decision; decided_by: string; note?: string }) =>
      api.decideRecommendation(payload.id, {
        decision: payload.decision,
        decided_by: payload.decided_by,
        note: payload.note,
      }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['recommendations'] })
    },
  })
}

export const useCreateCorrection = (projectId: string | null) => {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (payload: { source_id: string; source_type?: string; description: string; created_by?: string }) =>
      api.createCorrection(projectId as string, { created_by: 'researcher', ...payload }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.reviewQueue(projectId ?? '') })
      void client.invalidateQueries({ queryKey: keys.experiments(projectId ?? '', {}) })
      void client.invalidateQueries({ queryKey: keys.graph(projectId ?? '') })
    },
  })
}

export const useResolveReviewItem = () => {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (payload: { id: string; status: ReviewItemStatus; notes?: string; resolved_by?: string }) =>
      api.resolveReviewItem(payload.id, {
        status: payload.status,
        notes: payload.notes,
        resolved_by: payload.resolved_by ?? 'researcher',
      }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['reviewQueue'] })
    },
  })
}

export const useSessions = (projectId: string | null) =>
  useQuery({
    queryKey: ['sessions', projectId],
    queryFn: () => api.listSessions(projectId as string),
    enabled: Boolean(projectId),
  })

export const useCreateSession = (projectId: string | null) => {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (title?: string) => api.createSession(projectId as string, title),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['sessions', projectId] })
    },
  })
}

export const useSessionMessages = (projectId: string | null, sessionId: string | null) =>
  useQuery({
    queryKey: ['sessionMessages', projectId, sessionId],
    queryFn: () => api.getSessionMessages(projectId as string, sessionId as string),
    enabled: Boolean(projectId && sessionId),
  })

export const useQuerySession = (projectId: string | null, sessionId: string | null) => {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (payload: { question: string; compute_budget_gpu_hours?: number; max_recommendations?: number }) =>
      api.querySession(projectId as string, sessionId as string, payload),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['sessionMessages', projectId, sessionId] })
      void client.invalidateQueries({ queryKey: ['sessions', projectId] })
    },
  })
}

export const useMemorySummary = (projectId: string | null) =>
  useQuery({
    queryKey: ['memorySummary', projectId],
    queryFn: () => api.getMemorySummary(projectId as string),
    enabled: Boolean(projectId),
  })

export const useMcpTools = () =>
  useQuery({
    queryKey: ['mcpTools'],
    queryFn: api.getMcpTools,
  })
