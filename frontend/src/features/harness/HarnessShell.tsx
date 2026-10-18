import React, { useState, useEffect } from 'react'
import type {
  DatasetProfile,
  GraphNode,
  GraphSnapshot,
  ModelSpec,
  Proposal,
  RunState,
  WhyRecord,
} from '../../api/types'

import { UploadStep } from './UploadStep'
import { ModelPicker } from './ModelPicker'
import { HyperparameterPanel } from './HyperparameterPanel'
import { DependencyGraph } from './DependencyGraph'
import { WhyPanel } from './WhyPanel'
import { Leaderboard } from './Leaderboard'
import { LogStream } from './LogStream'

export const HarnessShell: React.FC = () => {
  // State
  const [dataset, setDataset] = useState<DatasetProfile | null>(null)
  const [models, setModels] = useState<ModelSpec[]>([])
  const [metrics, setMetrics] = useState<string[]>(['f1_macro', 'accuracy', 'roc_auc', 'pr_auc'])
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(['xgboost', 'lightgbm'])
  const [activeModelId, setActiveModelId] = useState<string | null>('xgboost')
  const [hyperparams, setHyperparams] = useState<Record<string, any>>({})

  const [mode, setMode] = useState<'autonomous' | 'human'>('autonomous')
  const [selectedMetric, setSelectedMetric] = useState<string>('f1_macro')
  const [budget, setBudget] = useState<number>(10)

  const [currentRun, setCurrentRun] = useState<RunState | null>(null)
  const [pendingProposal, setPendingProposal] = useState<Proposal | null>(null)
  const [graphSnapshot, setGraphSnapshot] = useState<GraphSnapshot>({ nodes: [], edges: [] })
  const [whyRecord, setWhyRecord] = useState<WhyRecord | null>(null)
  const [highlight, setHighlight] = useState<string[]>([])
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // 1. Fetch available models and metrics on mount
  useEffect(() => {
    fetch('/api/v1/catalog/models')
      .then((res) => res.json())
      .then((data) => {
        if (data.models) {
          setModels(data.models)
          if (!activeModelId && data.models.length > 0) {
            setActiveModelId(data.models[0].id)
          }
        }
        if (data.metrics) setMetrics(data.metrics)
      })
      .catch((err) => console.error('Failed to load catalog', err))
  }, [])

  // 2. Fetch graph snapshot whenever run is active or changes
  const fetchGraph = () => {
    const url = currentRun ? `/api/v1/graph?run_id=${currentRun.run_id}` : '/api/v1/graph'
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        if (data.nodes) setGraphSnapshot(data)
      })
      .catch((err) => console.error('Failed to fetch graph', err))
  }

  useEffect(() => {
    fetchGraph()
    const timer = setInterval(fetchGraph, 3000)
    return () => clearInterval(timer)
  }, [currentRun?.run_id])

  // 3. Poll current run state and pending proposal
  useEffect(() => {
    if (!currentRun) return
    const interval = setInterval(async () => {
      try {
        const runRes = await fetch(`/api/v1/runs/${currentRun.run_id}`)
        if (runRes.ok) {
          const updatedRun: RunState = await runRes.json()
          setCurrentRun(updatedRun)

          // If human mode and awaiting review, fetch pending proposal
          if (updatedRun.status === 'AWAITING_REVIEW') {
            const pendingRes = await fetch(`/api/v1/runs/${currentRun.run_id}/pending`)
            if (pendingRes.ok) {
              const pendingData = await pendingRes.json()
              setPendingProposal(pendingData.pending)
            }
          } else {
            setPendingProposal(null)
          }
        }
      } catch (err) {
        console.error('Error polling run state', err)
      }
    }, 1500)
    return () => clearInterval(interval)
  }, [currentRun?.run_id])

  // Handlers
  const handleToggleModel = (modelId: string) => {
    setSelectedModelIds((prev) =>
      prev.includes(modelId) ? prev.filter((m) => m !== modelId) : [...prev, modelId]
    )
  }

  const handleStartRun = async () => {
    if (!dataset) {
      setErrorMsg('Please upload a dataset first.')
      return
    }
    if (selectedModelIds.length === 0) {
      setErrorMsg('Select at least one model to run.')
      return
    }

    setErrorMsg(null)
    try {
      const res = await fetch('/api/v1/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: dataset.id,
          model_ids: selectedModelIds,
          mode: mode,
          metric: selectedMetric,
          budget: budget,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail?.[0]?.message || 'Failed to start run')
      }
      const newRun: RunState = await res.json()
      setCurrentRun(newRun)
    } catch (err: any) {
      setErrorMsg(err.message)
    }
  }

  const handlePauseRun = async () => {
    if (!currentRun) return
    try {
      const res = await fetch(`/api/v1/runs/${currentRun.run_id}/pause`, { method: 'POST' })
      if (res.ok) {
        const updated = await res.json()
        setCurrentRun(updated)
      }
    } catch (err: any) {
      setErrorMsg(err.message)
    }
  }

  const handleResumeRun = async () => {
    if (!currentRun) return
    try {
      const res = await fetch(`/api/v1/runs/${currentRun.run_id}/resume`, { method: 'POST' })
      if (res.ok) {
        const updated = await res.json()
        setCurrentRun(updated)
      }
    } catch (err: any) {
      setErrorMsg(err.message)
    }
  }

  const handleDecide = async (
    action: 'approve' | 'reject',
    overrides?: Record<string, any>,
    note?: string
  ) => {
    if (!currentRun || !pendingProposal) return
    try {
      const res = await fetch(`/api/v1/runs/${currentRun.run_id}/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          step: pendingProposal.step,
          action,
          hyperparameters: overrides,
          note,
        }),
      })
      if (res.ok) {
        const updated = await res.json()
        setCurrentRun(updated)
        setPendingProposal(null)
      } else {
        const err = await res.json()
        setErrorMsg(err.detail?.[0]?.message || 'Failed to submit decision')
      }
    } catch (err: any) {
      setErrorMsg(err.message)
    }
  }

  const handleSelectNode = async (node: GraphNode) => {
    if (!currentRun) return
    if (node.type === 'Trial' || node.type === 'Decision') {
      try {
        const trialId = node.type === 'Trial' ? node.id : node.props?.trial_id
        const res = await fetch(`/api/v1/runs/${currentRun.run_id}/why?trial_id=${trialId}`)
        if (res.ok) {
          const record: WhyRecord = await res.json()
          setWhyRecord(record)
          setHighlight(record.highlight || [])
        }
      } catch (err) {
        console.error('Failed to fetch why record for node', err)
      }
    } else {
      // Just highlight this node
      setHighlight([node.id])
    }
  }

  const handleHighlightChip = (chipId: string) => {
    // Find chip and immediate neighbours
    const neighbours = [chipId]
    for (const e of graphSnapshot.edges) {
      if (e.source === chipId) neighbours.push(e.target)
      if (e.target === chipId) neighbours.push(e.source)
    }
    setHighlight(Array.from(new Set(neighbours)))
  }

  const activeModel = models.find((m) => m.id === activeModelId) || null

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* 1. Header Bar: RUNWISE Console Controls */}
      <header className="bg-slate-900 border-b border-slate-800 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 sticky top-0 z-20">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <h1 className="font-black text-sm tracking-widest text-white uppercase">
              RUNWISE
            </h1>
          </div>
          <span className="text-slate-600">|</span>
          <span className="text-xs text-slate-400 font-mono">
            Autonomous Graph ML Harness
          </span>
        </div>

        {/* Run Controls & Parameters */}
        <div className="flex items-center gap-3">
          {/* Mode Toggle */}
          <div className="flex items-center bg-slate-950 p-0.5 rounded border border-slate-800 text-xs">
            <button
              onClick={() => setMode('autonomous')}
              className={`px-2.5 py-1 rounded font-medium transition ${
                mode === 'autonomous' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Autonomous
            </button>
            <button
              onClick={() => setMode('human')}
              className={`px-2.5 py-1 rounded font-medium transition ${
                mode === 'human' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Human Review
            </button>
          </div>

          {/* Metric Selector */}
          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span className="text-slate-400">Metric:</span>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
            >
              {metrics.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* Budget */}
          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span className="text-slate-400">Budget:</span>
            <select
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-xs"
            >
              <option value={5}>5 trials</option>
              <option value={10}>10 trials</option>
              <option value={20}>20 trials</option>
              <option value={50}>50 trials</option>
            </select>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            {(!currentRun || currentRun.status === 'COMPLETED' || currentRun.status === 'FAILED') ? (
              <button
                onClick={handleStartRun}
                className="bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs px-3.5 py-1.5 rounded transition shadow-sm"
              >
                ▶ Run
              </button>
            ) : currentRun.status === 'PAUSED' ? (
              <button
                onClick={handleResumeRun}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs px-3.5 py-1.5 rounded transition shadow-sm"
              >
                ▶ Resume
              </button>
            ) : (
              <button
                onClick={handlePauseRun}
                className="bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs px-3.5 py-1.5 rounded transition shadow-sm"
              >
                ⏸ Pause
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Error alert if any */}
      {errorMsg && (
        <div className="bg-rose-950 border-b border-rose-800 text-rose-300 px-4 py-1.5 text-xs flex justify-between items-center">
          <span>{errorMsg}</span>
          <button onClick={() => setErrorMsg(null)} className="text-slate-400 hover:text-white">
            ✕
          </button>
        </div>
      )}

      {/* Main Console Body */}
      <main className="flex-1 p-3 flex flex-col gap-3 max-w-[1720px] mx-auto w-full">
        {/* CSV Intake Header (F1) */}
        <UploadStep onDatasetLoaded={setDataset} currentProfile={dataset} />

        {/* Core 3-Column Harness Layout */}
        <div className="grid grid-cols-12 gap-3 flex-1">
          {/* Left Column: Models (F2) & Hyperparameters (F3) */}
          <div className="col-span-12 lg:col-span-3 flex flex-col gap-3">
            <ModelPicker
              models={models}
              selectedModelIds={selectedModelIds}
              onToggleModel={handleToggleModel}
              activeModelId={activeModelId}
              onSelectActiveModel={setActiveModelId}
            />
            <HyperparameterPanel
              model={activeModel}
              currentValues={hyperparams}
              onChangeValues={setHyperparams}
              onApplyOverride={(vals) => handleDecide('approve', vals)}
              isAwaitingReview={currentRun?.status === 'AWAITING_REVIEW'}
            />
          </div>

          {/* Center Column: Force-Directed Dependency Graph (F5) */}
          <div className="col-span-12 lg:col-span-6 flex flex-col gap-3">
            <DependencyGraph
              snapshot={graphSnapshot}
              highlight={highlight}
              onNodeClick={handleSelectNode}
              bestMetric={currentRun?.best_metric}
            />
          </div>

          {/* Right Column: Why This Trial (F7) & Human Review */}
          <div className="col-span-12 lg:col-span-3 flex flex-col gap-3">
            <WhyPanel
              whyRecord={whyRecord}
              pendingProposal={pendingProposal}
              isAwaitingReview={currentRun?.status === 'AWAITING_REVIEW'}
              onDecide={handleDecide}
              onHighlightChip={handleHighlightChip}
            />
          </div>
        </div>

        {/* Bottom Section: Leaderboard (Ranked Models) & Live Log Stream (F6) */}
        <div className="grid grid-cols-12 gap-3">
          <div className="col-span-12 lg:col-span-4">
            <Leaderboard
              snapshot={graphSnapshot}
              metricName={currentRun?.metric_name || selectedMetric}
              onSelectTrial={(trialId) => handleSelectNode({ id: trialId, label: trialId, type: 'Trial', props: {} })}
            />
          </div>
          <div className="col-span-12 lg:col-span-8">
            <LogStream runId={currentRun?.run_id || null} mode={currentRun?.mode} />
          </div>
        </div>
      </main>
    </div>
  )
}
