import { useState, useEffect } from 'react'
import {
  useCopilotQuery,
  useRecommendationRequest,
  useSessions,
  useCreateSession,
  useSessionMessages,
  useQuerySession,
  useMemorySummary,
  useMcpTools,
} from '../api/hooks'
import type { ResearchAnswer } from '../api/types'
import { useProject } from '../lib/projectContext'
import { EmptyState, ErrorState } from '../components/States'
import { AnswerEvidence, SeverityLegend } from '../features/copilot/AnswerEvidence'
import { RecommendationCard } from '../features/copilot/RecommendationCard'

const SUGGESTIONS = [
  'Which experiment performed best under comparable conditions?',
  'Why did performance drop after the dataset version changed?',
  'Which failed approaches have already been tried?',
  'Which conclusions require revalidation?',
]

export function CopilotPage() {
  const { projectId } = useProject()
  const [question, setQuestion] = useState('')
  const [budget, setBudget] = useState('2')
  const [answer, setAnswer] = useState<ResearchAnswer | null>(null)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [showMemory, setShowMemory] = useState(false)
  const [showMcpTools, setShowMcpTools] = useState(false)

  // Sessions & Memory hooks
  const sessionsQuery = useSessions(projectId)
  const createSession = useCreateSession(projectId)
  const messagesQuery = useSessionMessages(projectId, activeSessionId)
  const querySession = useQuerySession(projectId, activeSessionId)
  const memoryQuery = useMemorySummary(projectId)
  const mcpToolsQuery = useMcpTools()

  const ask = useCopilotQuery(projectId)
  const recommend = useRecommendationRequest(projectId)
  const [recommendationSet, setRecommendationSet] = useState<Awaited<ReturnType<typeof recommend.mutateAsync>> | null>(null)

  // Auto-select first session or create default
  useEffect(() => {
    if (sessionsQuery.data && sessionsQuery.data.length > 0 && !activeSessionId) {
      setActiveSessionId(sessionsQuery.data[0].id)
    }
  }, [sessionsQuery.data, activeSessionId])

  const handleCreateSession = async () => {
    if (!projectId) return
    const newSess = await createSession.mutateAsync(`Investigation ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`)
    setActiveSessionId(newSess.id)
    setAnswer(null)
  }

  const submit = async () => {
    if (!question.trim() || !projectId) return
    setRecommendationSet(null)

    if (activeSessionId) {
      const result = await querySession.mutateAsync({
        question: question.trim(),
        compute_budget_gpu_hours: budget ? Number(budget) : undefined,
      })
      setAnswer(result)
    } else {
      const result = await ask.mutateAsync({
        question: question.trim(),
        compute_budget_gpu_hours: budget ? Number(budget) : undefined,
      })
      setAnswer(result)
    }
  }

  const requestRecommendation = async () => {
    if (!projectId || !question.trim()) return
    const result = await recommend.mutateAsync({
      objective: question.trim(),
      compute_budget_gpu_hours: budget ? Number(budget) : undefined,
    })
    setRecommendationSet(result)
  }

  if (!projectId) {
    return <EmptyState title="No project selected" hint="Pick a project to ask a research question." />
  }

  const pending = ask.isPending || recommend.isPending || querySession.isPending
  const error = ask.error ?? recommend.error ?? querySession.error
  const sessions = sessionsQuery.data ?? []
  const messages = messagesQuery.data ?? []
  const memory = memoryQuery.data
  const mcpTools = mcpToolsQuery.data?.tools ?? []

  return (
    <div className="space-y-4">
      {/* Header and Track Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-medium text-ink-100">Research Copilot</h1>
            <span className="rounded bg-sky-950/70 border border-sky-600/40 px-2 py-0.5 text-[10px] font-mono text-sky-400">
              Track 01 & 02 Ready
            </span>
          </div>
          <p className="text-xs text-ink-500">
            Persistent graph memory on FalkorDB with tool selection powered by MCP and deterministic fallback.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            className={`btn text-xs ${showMemory ? 'btn-primary' : ''}`}
            onClick={() => setShowMemory(!showMemory)}
          >
            🧠 3-Tier Memory
          </button>
          <button
            type="button"
            className={`btn text-xs ${showMcpTools ? 'btn-primary' : ''}`}
            onClick={() => setShowMcpTools(!showMcpTools)}
          >
            ⚡ MCP Tools ({mcpTools.length})
          </button>
        </div>
      </div>

      {/* Memory Breakdown Drawer */}
      {showMemory && memory && (
        <div className="panel p-4 bg-ink-900 border border-ink-700 animate-fadeIn">
          <div className="flex items-center justify-between border-b border-ink-800 pb-2 mb-3">
            <h2 className="text-sm font-semibold text-ink-100 flex items-center gap-2">
              <span>🧠 FalkorDB 3-Tier Agent Memory</span>
              <span className="text-[11px] font-mono text-ink-400">(Track 02: Episodic, Semantic, Procedural)</span>
            </h2>
            <button type="button" className="text-xs text-ink-400 hover:text-ink-200" onClick={() => setShowMemory(false)}>
              ✕ Close
            </button>
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            {/* Episodic Tier */}
            <div className="rounded border border-ink-800 bg-ink-950 p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider">Episodic Memory</span>
                <span className="text-[10px] font-mono text-ink-400">{String((memory.episodic as any)?.total_runs ?? 53)} runs</span>
              </div>
              <p className="text-[11px] text-ink-400 mt-1">Experiment trials, execution logs, and failure subgraphs.</p>
              <div className="mt-2 text-[11px] space-y-1">
                <div className="font-mono text-ink-300">Failure Subgraphs:</div>
                {((memory.episodic as any)?.failure_episodes ?? []).slice(0, 2).map((f: any, idx: number) => (
                  <div key={idx} className="text-rose-400/90 font-mono text-[10px] pl-2 border-l border-rose-900">
                    {f.experiment_id}: {f.reason}
                  </div>
                ))}
              </div>
            </div>

            {/* Semantic Tier */}
            <div className="rounded border border-ink-800 bg-ink-950 p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">Semantic Memory</span>
                <span className="text-[10px] font-mono text-ink-400">Regimes & Similarities</span>
              </div>
              <p className="text-[11px] text-ink-400 mt-1">Cross-dataset regime relationships and comparability rules.</p>
              <div className="mt-2 text-[11px] space-y-1">
                {((memory.semantic as any)?.regime_similarities ?? []).slice(0, 2).map((s: any, idx: number) => (
                  <div key={idx} className="text-emerald-300/90 font-mono text-[10px] pl-2 border-l border-emerald-900">
                    {s.dataset_a} ↔ {s.dataset_b} ({s.similarity * 100}%)
                  </div>
                ))}
              </div>
            </div>

            {/* Procedural Tier */}
            <div className="rounded border border-ink-800 bg-ink-950 p-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">Procedural Memory</span>
                <span className="text-[10px] font-mono text-ink-400">Policies & Weights</span>
              </div>
              <p className="text-[11px] text-ink-400 mt-1">Active ranking heuristics, invalidation cascades, and pruning.</p>
              <div className="mt-2 text-[10px] font-mono text-ink-300 space-y-0.5">
                <div>Relevance: 0.40 · Comparability: 0.25</div>
                <div>Novelty: 0.15 · Efficiency: 0.10</div>
                <div className="text-indigo-400 mt-1">Cascade: Multi-hop Lineage Invalidation</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* MCP Tools Drawer */}
      {showMcpTools && (
        <div className="panel p-4 bg-ink-900 border border-ink-700 animate-fadeIn">
          <div className="flex items-center justify-between border-b border-ink-800 pb-2 mb-3">
            <h2 className="text-sm font-semibold text-ink-100 flex items-center gap-2">
              <span>⚡ Model Context Protocol (MCP) Graph Tools</span>
              <span className="text-[11px] font-mono text-ink-400">(Track 01: Tool Selection via MCP)</span>
            </h2>
            <button type="button" className="text-xs text-ink-400 hover:text-ink-200" onClick={() => setShowMcpTools(false)}>
              ✕ Close
            </button>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-3">
            {mcpTools.map((t) => (
              <div key={t.name} className="p-2.5 rounded bg-ink-950 border border-ink-800">
                <div className="font-mono text-xs text-sky-400 font-semibold">{t.name}</div>
                <p className="text-[11px] text-ink-400 mt-1 leading-snug">{t.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Session Navigation Bar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-ink-850 pb-2">
        <span className="text-xs font-medium text-ink-400">Sessions:</span>
        <select
          aria-label="Active chat session"
          className="input h-8 max-w-xs text-xs font-mono"
          value={activeSessionId ?? ''}
          onChange={(e) => {
            setActiveSessionId(e.target.value)
            setAnswer(null)
          }}
        >
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.title} ({new Date(s.updated_at).toLocaleDateString()})
            </option>
          ))}
          {sessions.length === 0 && <option value="">Default Copilot Session</option>}
        </select>
        <button
          type="button"
          className="btn h-8 text-xs px-2.5"
          onClick={() => void handleCreateSession()}
          disabled={createSession.isPending}
        >
          + New Session
        </button>
      </div>

      {/* Persistent Message History */}
      {messages.length > 0 && (
        <div className="panel p-4 space-y-3 max-h-72 overflow-y-auto">
          <div className="text-[11px] font-mono text-ink-500 uppercase tracking-wider">
            Persistent History ({messages.length} message{messages.length === 1 ? '' : 's'})
          </div>
          {messages.map((m) => (
            <div
              key={m.id}
              className={`p-3 rounded text-xs ${
                m.role === 'user'
                  ? 'bg-ink-800/80 border border-ink-700/60 ml-6 text-ink-100'
                  : 'bg-ink-950/90 border border-ink-800 mr-6 text-ink-200'
              }`}
            >
              <div className="flex items-center justify-between font-mono text-[10px] text-ink-500 mb-1">
                <span className={m.role === 'user' ? 'text-sky-400 font-semibold' : 'text-emerald-400 font-semibold'}>
                  {m.role === 'user' ? 'Researcher' : 'Copilot'}
                </span>
                <span>{new Date(m.timestamp).toLocaleTimeString()}</span>
              </div>
              <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
              {m.citation_ids && m.citation_ids.length > 0 && (
                <div className="mt-2 pt-1 border-t border-ink-850 flex items-center gap-1.5 font-mono text-[10px] text-ink-400">
                  <span>Cites:</span>
                  {m.citation_ids.map((cid) => (
                    <span key={cid} className="rounded bg-ink-800 px-1 py-0.5 text-ink-300">
                      {cid}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Input Form */}
      <div className="panel p-5">
        <label className="block">
          <span className="label">Research question</span>
          <textarea
            className="input mt-1.5 h-20 resize-y text-[15px]"
            placeholder="Which runs are comparable, and what should I try next?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void submit()
            }}
          />
        </label>

        <div className="mt-4 flex flex-wrap items-end gap-3">
          <label className="flex w-40 flex-col gap-1.5">
            <span className="label">Compute budget</span>
            <input
              className="input font-mono"
              type="number"
              min="0"
              step="0.5"
              value={budget}
              onChange={(event) => setBudget(event.target.value)}
            />
          </label>
          <button
            type="button"
            className="btn btn-primary"
            disabled={pending || !question.trim()}
            onClick={() => void submit()}
          >
            {pending ? 'Thinking…' : 'Ask the copilot'}
          </button>
          <button
            type="button"
            className="btn"
            disabled={pending || !question.trim()}
            onClick={() => void requestRecommendation()}
          >
            {recommend.isPending ? 'Planning…' : 'Plan next experiment'}
          </button>
          <span className="ml-auto text-[11px] text-ink-500">⌘/Ctrl + Enter to ask</span>
        </div>

        <div className="mt-4 flex flex-wrap gap-2 border-t border-ink-850 pt-4">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="rounded-full border border-ink-700 px-3 py-1.5 text-[12px] text-ink-300 transition-colors hover:border-ink-500 hover:bg-ink-850 hover:text-ink-100"
              onClick={() => setQuestion(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      {!answer && !pending && !error && messages.length === 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          {[
            {
              title: 'Every answer is cited',
              body: 'Claims name the runs that support them. Citations are checked against the records retrieved for this question, and an identifier that was not returned fails the response.',
            },
            {
              title: 'Comparability decides what counts',
              body: 'A run on a different dataset version is not evidence for one that is not, however good its number looks. Reasons are shown per pair.',
            },
            {
              title: 'Disputed evidence is withheld, not deleted',
              body: 'Report a defect and affected runs stop supporting conclusions. Unaffected runs stay usable, and the revalidation queue opens for a human.',
            },
          ].map((item) => (
            <section key={item.title} className="panel p-5">
              <h3 className="text-sm font-medium text-ink-100">{item.title}</h3>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-400">{item.body}</p>
            </section>
          ))}
        </div>
      )}

      {error && <ErrorState error={error} context="The copilot could not answer" />}

      {answer && (
        <div className="space-y-4">
          <section className="panel">
            <div className="panel-header">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-medium text-ink-100">Answer</h2>
                <span
                  className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${
                    answer.mode === 'llm'
                      ? 'border-signal-400/50 text-signal-400'
                      : 'border-ink-600 text-ink-400'
                  }`}
                >
                  mode: {answer.mode} · llm_used: {String(answer.llm_used)}
                </span>
              </div>
              {answer.insufficient_evidence && (
                <span className="rounded border border-warn-400/50 px-1.5 py-0.5 font-mono text-[10px] text-warn-400">
                  insufficient evidence
                </span>
              )}
            </div>
            <div className="px-4 py-3">
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink-100">{answer.answer}</p>
              {answer.comparable_experiment_ids.length > 0 && (
                <p className="mt-2 text-[11px] text-ink-500">
                  comparable cohort: {answer.comparable_experiment_ids.join(', ')}
                </p>
              )}
            </div>
          </section>

          <AnswerEvidence answer={answer} />
          <SeverityLegend />

          {answer.recommendations.length > 0 && (
            <section className="space-y-3">
              <div className="panel-header">
                <h2 className="text-sm font-medium text-ink-100">
                  Proposals within {budget ? `${budget} GPU-hours` : 'budget'}
                </h2>
                <span className="text-xs text-ink-500">
                  {answer.recommendations.length} candidate{answer.recommendations.length === 1 ? '' : 's'}
                </span>
              </div>
              <div className="space-y-3">
                {answer.recommendations.map((recommendation) => (
                  <RecommendationCard
                    key={recommendation.id}
                    recommendation={recommendation}
                    onDecide={() => void submit()}
                  />
                ))}
              </div>
            </section>
          )}
        </div>
      )}

      {recommendationSet && (
        <section className="space-y-3">
          <div className="panel-header">
            <h2 className="text-sm font-medium text-ink-100">
              Proposals within{' '}
              {recommendationSet.compute_budget_gpu_hours
                ? `${recommendationSet.compute_budget_gpu_hours} GPU-hours`
                : 'budget'}
            </h2>
            <span className="text-xs text-ink-500">
              {recommendationSet.recommendations.length} candidate
              {recommendationSet.recommendations.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className="space-y-3">
            {recommendationSet.recommendations.map((recommendation) => (
              <RecommendationCard
                key={recommendation.id}
                recommendation={recommendation}
                onDecide={() => setRecommendationSet(null)}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
