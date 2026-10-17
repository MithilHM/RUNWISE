import { useState } from 'react'
import { useCopilotQuery, useRecommendationRequest } from '../api/hooks'
import type { ResearchAnswer } from '../api/types'
import { useProject } from '../lib/projectContext'
import { EmptyState, ErrorState } from '../components/States'
import { AnswerEvidence, SeverityLegend } from '../features/copilot/AnswerEvidence'
import { RecommendationCard } from '../features/copilot/RecommendationCard'
import { formatNumber } from '../lib/format'

const SUGGESTIONS = [
  'Which experiment performed best under comparable conditions?',
  'Why did performance drop after the dataset version changed?',
  'Which failed approaches have already been tried?',
  'Which conclusions require revalidation?',
]

/**
 * Screen B: research copilot.
 *
 * The answer and its evidence are one unit. An answer with no supporting record
 * shown next to it is exactly the failure mode FR-8 exists to prevent, so claims,
 * comparability verdicts, withheld evidence and caveats are rendered as part of the
 * answer rather than behind a disclosure.
 */
export function CopilotPage() {
  const { projectId, project } = useProject()
  const [question, setQuestion] = useState('')
  const [budget, setBudget] = useState('2')
  const [answer, setAnswer] = useState<ResearchAnswer | null>(null)

  const ask = useCopilotQuery(projectId)
  const recommend = useRecommendationRequest(projectId)
  const [recommendationSet, setRecommendationSet] = useState<Awaited<ReturnType<typeof recommend.mutateAsync>> | null>(null)

  const submit = async () => {
    if (!question.trim() || !projectId) return
    setRecommendationSet(null)
    const result = await ask.mutateAsync({
      question: question.trim(),
      compute_budget_gpu_hours: budget ? Number(budget) : undefined,
    })
    setAnswer(result)
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

  const pending = ask.isPending || recommend.isPending
  const error = ask.error ?? recommend.error

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-medium text-ink-100">Research copilot</h1>
        <p className="text-xs text-ink-500">
          Answers are grounded in {project?.name ?? projectId}. Without an LLM key the backend answers
          deterministically and reports <span className="font-mono">llm_used=false</span>.
        </p>
      </div>

      <div className="panel p-4">
        <label className="block">
          <span className="label">Research question</span>
          <textarea
            className="input mt-1 h-20 resize-y"
            placeholder="Which runs are comparable, and what should I try next?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void submit()
            }}
          />
        </label>

        <div className="mt-3 flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="label">Compute budget (GPU-hours)</span>
            <input
              className="input w-32"
              type="number"
              min="0"
              step="0.5"
              value={budget}
              onChange={(event) => setBudget(event.target.value)}
            />
          </label>
          <button type="button" className="btn btn-primary" disabled={pending || !question.trim()} onClick={() => void submit()}>
            {ask.isPending ? 'Thinking…' : 'Ask'}
          </button>
          <button
            type="button"
            className="btn"
            disabled={pending || !question.trim()}
            onClick={() => void requestRecommendation()}
          >
            {recommend.isPending ? 'Planning…' : 'Recommend next experiment'}
          </button>
          <span className="text-[11px] text-ink-600">⌘/Ctrl + Enter to ask</span>
        </div>

        <div className="mt-3 flex flex-wrap gap-1.5">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="rounded border border-ink-700 px-2 py-1 text-[11px] text-ink-400 transition hover:border-ink-600 hover:text-ink-200"
              onClick={() => setQuestion(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

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
                  title={
                    answer.mode === 'llm'
                      ? 'A language model produced the wording. The evidence and every identifier were still checked against the graph.'
                      : 'Produced by deterministic templates. No language model was consulted.'
                  }
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
              <h2 className="text-sm font-medium text-ink-100">Recommendations ({answer.recommendations.length})</h2>
              {answer.recommendations.map((recommendation) => (
                <RecommendationCard key={recommendation.id} recommendation={recommendation} />
              ))}
            </section>
          )}

          {answer.corrections_considered.length > 0 && (
            <section className="panel px-4 py-3">
              <h2 className="label">Corrections considered</h2>
              <ul className="mt-2 space-y-2 text-xs">
                {answer.corrections_considered.map((impact) => (
                  <li key={impact.correction_id}>
                    <span className="mono-id">{impact.correction_id}</span>
                    <span className="ml-2 text-ink-300">{impact.summary}</span>
                    <div className="ml-2 text-ink-500">
                      {impact.directly_affected_experiment_ids.length} directly affected ·{' '}
                      {impact.indirectly_affected_experiment_ids.length} downstream ·{' '}
                      {impact.unaffected_experiment_ids.length} unaffected
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}

      {recommendationSet && (
        <section className="space-y-3">
          <div className="panel px-4 py-3">
            <div className="flex flex-wrap items-center gap-4 text-xs">
              <h2 className="text-sm font-medium text-ink-100">Recommendation set</h2>
              <span className="text-ink-300">
                {recommendationSet.recommendations.length} proposal(s),{' '}
                {formatNumber(recommendationSet.estimated_gpu_hours)} /{' '}
                {formatNumber(recommendationSet.compute_budget_gpu_hours)} GPU-h planned
              </span>
              {recommendationSet.budget_exhausted && (
                <span className="rounded border border-warn-400/50 px-1.5 py-0.5 font-mono text-[10px] text-warn-400">
                  budget was the binding constraint
                </span>
              )}
              {recommendationSet.unselected_candidates.length > 0 && (
                <span className="text-ink-500">
                  {recommendationSet.unselected_candidates.length} candidate(s) not selected
                </span>
              )}
            </div>
            {recommendationSet.considered_experiment_ids.length > 0 && (
              <p className="mt-1 text-[11px] text-ink-500">
                evidence base: {recommendationSet.considered_experiment_ids.length} run(s)
              </p>
            )}
          </div>

          {recommendationSet.recommendations.length === 0 ? (
            <div className="panel">
              <EmptyState
                title="No proposal could be grounded"
                hint="This is an answer, not a failure. The reason is listed below and in the response's uncertainties field."
              />
              <ul className="list-disc space-y-1 px-8 pb-4 text-xs text-ink-300">
                {recommendationSet.uncertainties.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          ) : (
            recommendationSet.recommendations.map((recommendation) => (
              <RecommendationCard key={recommendation.id} recommendation={recommendation} />
            ))
          )}

          {recommendationSet.uncertainties.length > 0 && (
            <div className="panel border-warn-400/30 px-4 py-3">
              <h3 className="label text-warn-400">Uncertainties</h3>
              <ul className="mt-1.5 list-disc space-y-1 pl-4 text-xs text-ink-300">
                {recommendationSet.uncertainties.map((item, index) => (
                  <li key={index}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </section>
      )}
    </div>
  )
}
