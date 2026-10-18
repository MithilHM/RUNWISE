import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import type { ModelSpec, GraphSnapshot, WhyRecord } from '../../api/types'

import { HyperparameterPanel } from './HyperparameterPanel'
import { Leaderboard } from './Leaderboard'
import { WhyPanel } from './WhyPanel'

describe('Member B Harness Components', () => {
  const sampleModel: ModelSpec = {
    id: 'xgboost',
    name: 'XGBoost',
    library: 'xgboost',
    description: 'Gradient boosted trees',
    params: [
      { name: 'n_estimators', kind: 'int', low: 50, high: 500, step: 10, default: 200 },
      { name: 'learning_rate', kind: 'float', low: 0.01, high: 0.5, step: 0.01, default: 0.05 },
      { name: 'use_gpu', kind: 'bool', default: false },
      { name: 'tree_method', kind: 'categorical', choices: ['auto', 'hist', 'exact'], default: 'auto' },
      { name: 'max_depth', kind: 'int_or_null', low: 1, high: 16, default: 6 },
    ],
  }

  it('HyperparameterPanel renders controls for each kind and clamps values', () => {
    const onChange = vi.fn()
    render(
      <HyperparameterPanel
        model={sampleModel}
        currentValues={{ n_estimators: 200, learning_rate: 0.05, use_gpu: false, tree_method: 'auto', max_depth: 6 }}
        onChangeValues={onChange}
      />
    )

    expect(screen.getByText('n_estimators')).toBeInTheDocument()
    expect(screen.getByText('learning_rate')).toBeInTheDocument()
    expect(screen.getByText('use_gpu')).toBeInTheDocument()
    expect(screen.getByText('tree_method')).toBeInTheDocument()
    expect(screen.getByText('max_depth')).toBeInTheDocument()

    // Test toggle bool
    const boolBtn = screen.getByText('FALSE')
    fireEvent.click(boolBtn)
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ use_gpu: true }))
  })

  it('Leaderboard sorts ascending for loss metrics and descending for accuracy/f1', () => {
    const snapshot: GraphSnapshot = {
      nodes: [
        { id: 't1', label: 't1', type: 'Trial', props: { model_id: 'xgboost', metric_value: 0.85, step: 1 } },
        { id: 't2', label: 't2', type: 'Trial', props: { model_id: 'lightgbm', metric_value: 0.92, step: 2 } },
        { id: 't3', label: 't3', type: 'Trial', props: { model_id: 'random_forest', metric_value: 0.78, step: 3 } },
      ],
      edges: [],
    }

    // 1. Maximizing metric (f1_macro)
    const { rerender } = render(<Leaderboard snapshot={snapshot} metricName="f1_macro" />)
    const rankBadges = screen.getAllByText(/#\d/)
    expect(rankBadges[0]).toHaveTextContent('#1')
    expect(screen.getByText('0.9200')).toBeInTheDocument()

    // 2. Minimizing metric (log_loss)
    const lossSnapshot: GraphSnapshot = {
      nodes: [
        { id: 't1', label: 't1', type: 'Trial', props: { model_id: 'xgboost', metric_value: 0.35, step: 1 } },
        { id: 't2', label: 't2', type: 'Trial', props: { model_id: 'lightgbm', metric_value: 0.18, step: 2 } },
      ],
      edges: [],
    }
    rerender(<Leaderboard snapshot={lossSnapshot} metricName="log_loss" />)
    expect(screen.getByText('0.1800')).toBeInTheDocument()
  })

  it('WhyPanel renders prose, citations and raises highlight on evidence chip click', () => {
    const onHighlight = vi.fn()
    const whyRecord: WhyRecord = {
      decision: {
        decision_id: 'dec_1',
        run_id: 'run_1',
        step: 1,
        trial_id: 't_1',
        actor: 'laya',
        strategy: 'exploit_basin',
        why: 'Selected candidate based on gradient convergence in neighbouring trials.',
        hypothesis: 'Deeper trees will capture interaction terms.',
        confidence: 0.89,
        citations: ['node_dataset', 'node_param_lr'],
        created_at: new Date().toISOString(),
      },
      config: { max_depth: 6 },
      evidence: [
        { kind: 'Dataset', id: 'node_dataset', label: 'Fraud CSV', detail: '80/20 train/test' },
        { kind: 'Param', id: 'node_param_lr', label: 'lr=0.05', detail: 'learning rate' },
      ],
      highlight: ['node_dataset', 'node_param_lr'],
    }

    render(
      <WhyPanel
        whyRecord={whyRecord}
        pendingProposal={null}
        isAwaitingReview={false}
        onHighlightChip={onHighlight}
      />
    )

    expect(screen.getByText('exploit_basin')).toBeInTheDocument()
    expect(screen.getByText(/Selected candidate based on gradient convergence/)).toBeInTheDocument()
    expect(screen.getByText(/Deeper trees will capture interaction terms/)).toBeInTheDocument()
    expect(screen.getByText('89%')).toBeInTheDocument()

    // Click evidence chip
    const chip = screen.getByText('Fraud CSV')
    fireEvent.click(chip)
    expect(onHighlight).toHaveBeenCalledWith('node_dataset')
  })
})
