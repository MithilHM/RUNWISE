import { screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { Recommendation } from '../../api/types'
import { ScoreBreakdown } from '../../components/ScoreBreakdown'
import { renderWithProviders } from '../../test/renderWithProviders'
import { RecommendationCard } from './RecommendationCard'

const render = renderWithProviders

/**
 * The four UI rules the backend depends on (HANDOFF.md):
 *  1. the epistemic grade is rendered,
 *  2. a score is never presented as a probability,
 *  3. uncertainties are always shown,
 *  4. a missing estimate reads as unknown, not as zero.
 */

const recommendation: Recommendation = {
  id: 'rec_001',
  project_id: 'project_demo',
  objective: 'Improve validation accuracy',
  hypothesis: 'Increasing max_depth may improve generalisation',
  candidate_kind: 'sweep_parameter',
  baseline_experiment_ids: ['exp_001'],
  configuration: { max_depth: 6 },
  independent_variables: ['max_depth'],
  controlled_variables: ['dataset_version', 'evaluation_protocol'],
  metrics: ['validation_accuracy'],
  estimated_gpu_hours: null,
  estimate_source: null,
  supporting_experiment_ids: ['exp_001', 'exp_002'],
  independent_variable_values: { max_depth: 6 },
  uncertainties: ['Only one comparable baseline is available'],
  validation_criteria: [
    { kind: 'threshold', description: 'Stop when accuracy improves by less than 0.005', is_suggested_default: true, threshold: 0.005, metric_name: 'validation_accuracy' },
  ],
  rationale: 'max_depth has been observed at 3 and 5 but not 6',
  score: 0.61,
  score_kind: 'weighted_heuristic',
  score_components: [
    { name: 'relevance', weight: 0.22, value: 0.8, weighted_value: 0.176, rationale: 'family and version match' },
    { name: 'cost_efficiency', weight: 0.14, value: 0.5, weighted_value: 0.07, rationale: 'median cohort cost' },
  ],
  evidence_status: 'inferred',
  excluded_evidence_ids: ['exp_003'],
  status: 'proposed',
  created_at: '2026-10-17T10:00:00Z',
}

describe('RecommendationCard', () => {
  it('labels the proposal as inferred rather than observed', () => {
    render(<RecommendationCard recommendation={recommendation} />)
    expect(screen.getByTitle('Derived by the system, not observed')).toBeInTheDocument()
  })

  it('always shows the uncertainties', () => {
    render(<RecommendationCard recommendation={recommendation} />)
    expect(screen.getByText('Only one comparable baseline is available')).toBeInTheDocument()
  })

  it('renders an unknown compute estimate as unknown, never as zero', () => {
    render(<RecommendationCard recommendation={recommendation} />)
    expect(screen.getByText(/not recorded/)).toBeInTheDocument()
    expect(screen.queryByText(/0 GPU-h/)).not.toBeInTheDocument()
  })

  it('offers approval, which FR-9 requires for a consequential change', () => {
    render(<RecommendationCard recommendation={recommendation} />)
    expect(screen.getByRole('button', { name: 'Approve' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Reject' })).toBeInTheDocument()
  })

  it('cites the baseline and supporting runs by id', () => {
    render(<RecommendationCard recommendation={recommendation} />)
    const baseline = screen.getByText('Baseline runs').closest('div') as HTMLElement
    expect(within(baseline).getByText('exp_001')).toBeInTheDocument()
  })
})

describe('ScoreBreakdown', () => {
  it('shows the weighted components rather than a bare number', () => {
    render(
      <ScoreBreakdown score={0.61} scoreKind="weighted_heuristic" components={recommendation.score_components} />,
    )
    expect(screen.getByText('relevance')).toBeInTheDocument()
    expect(screen.getByText(/not a calibrated probability/i)).toBeInTheDocument()
  })

  it('still renders when the backend returned no components', () => {
    render(<ScoreBreakdown score={0.4} scoreKind="weighted_heuristic" components={[]} />)
    expect(screen.getByText(/No component breakdown/)).toBeInTheDocument()
  })
})
