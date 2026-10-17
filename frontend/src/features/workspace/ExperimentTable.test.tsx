import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { ExperimentSummary } from '../../api/types'
import { ExperimentTable } from './ExperimentTable'

/** A blank metric column means unrecorded. The table must not imply a zero. */

const rows: ExperimentSummary[] = [
  {
    id: 'exp_001',
    project_id: 'p1',
    name: 'baseline_xgboost',
    model_family: 'gradient_boosting',
    dataset_version_id: 'tabular_fraud_v1',
    status: 'completed',
    metrics: [{ name: 'validation_accuracy', value: 0.91, direction: 'maximize', evaluation_split: 'validation', evaluation_protocol: '' }],
    compute_cost_gpu_hours: 0.8,
    created_at: '2026-10-17T09:00:00Z',
    duration_seconds: 3200,
    quality_status: 'needs_review',
  },
  {
    id: 'exp_002',
    project_id: 'p1',
    name: 'unmeasured_run',
    model_family: 'transformer',
    dataset_version_id: null,
    status: 'failed',
    metrics: [],
    compute_cost_gpu_hours: null,
    created_at: '2026-10-17T09:05:00Z',
    duration_seconds: null,
    quality_status: 'valid',
  },
]

describe('ExperimentTable', () => {
  it('shows a review badge for a run an open correction reached', () => {
    render(<ExperimentTable experiments={rows} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('Needs Review')).toBeInTheDocument()
  })

  it('renders an absent metric as not recorded rather than 0', () => {
    render(<ExperimentTable experiments={rows} selectedId={null} onSelect={() => {}} />)
    const missingCells = screen.getAllByText('not recorded')
    expect(missingCells.length).toBeGreaterThan(0)
    expect(screen.queryByText('0 ↑')).not.toBeInTheDocument()
  })

  it('renders an absent dataset version as unrecorded', () => {
    render(<ExperimentTable experiments={rows} selectedId={null} onSelect={() => {}} />)
    expect(screen.getByText('exp_002')).toBeInTheDocument()
  })
})
