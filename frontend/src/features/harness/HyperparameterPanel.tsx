import React, { useState, useEffect } from 'react'
import type { ModelSpec, ParamSpec } from '../../api/types'


interface HyperparameterPanelProps {
  model: ModelSpec | null
  currentValues: Record<string, any>
  onChangeValues: (values: Record<string, any>) => void
  onApplyOverride?: (values: Record<string, any>) => void
  isAwaitingReview?: boolean
}

export const HyperparameterPanel: React.FC<HyperparameterPanelProps> = ({
  model,
  currentValues,
  onChangeValues,
  onApplyOverride,
  isAwaitingReview = false,
}) => {
  const [localValues, setLocalValues] = useState<Record<string, any>>(currentValues)

  useEffect(() => {
    setLocalValues(currentValues)
  }, [currentValues])

  if (!model) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs text-slate-500 italic">
        Select a model to view its hyperparameter space.
      </div>
    )
  }

  const handleNumericChange = (param: ParamSpec, rawVal: number | null) => {
    if (rawVal === null) {
      const next = { ...localValues, [param.name]: null }
      setLocalValues(next)
      onChangeValues(next)
      return
    }

    let clamped = rawVal
    if (param.low !== null && param.low !== undefined && clamped < param.low) {
      clamped = param.low
    }
    if (param.high !== null && param.high !== undefined && clamped > param.high) {
      clamped = param.high
    }

    const next = { ...localValues, [param.name]: clamped }
    setLocalValues(next)
    onChangeValues(next)
  }

  const handleBoolChange = (paramName: string, val: boolean) => {
    const next = { ...localValues, [paramName]: val }
    setLocalValues(next)
    onChangeValues(next)
  }

  const handleCategoricalChange = (paramName: string, val: any) => {
    const next = { ...localValues, [paramName]: val }
    setLocalValues(next)
    onChangeValues(next)
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs">
      <div className="flex items-center justify-between mb-3">
        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Hyperparameters ({model.name})
        </span>
        {onApplyOverride && isAwaitingReview && (
          <button
            onClick={() => onApplyOverride(localValues)}
            className="bg-cyan-600 hover:bg-cyan-500 text-white text-[10px] font-semibold px-2.5 py-1 rounded transition"
          >
            Apply Edit
          </button>
        )}
      </div>

      <div className="space-y-3 max-h-[340px] overflow-y-auto pr-1">
        {model.params.map((param) => {
          const val = localValues[param.name] !== undefined ? localValues[param.name] : param.default

          if (param.kind === 'bool') {
            return (
              <div key={param.name} className="flex items-center justify-between">
                <span className="font-mono text-slate-300">{param.name}</span>
                <button
                  type="button"
                  onClick={() => handleBoolChange(param.name, !val)}
                  className={`px-2.5 py-0.5 rounded text-[11px] font-mono font-medium transition ${
                    val
                      ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                      : 'bg-slate-800 text-slate-400 border border-slate-700'
                  }`}
                >
                  {val ? 'TRUE' : 'FALSE'}
                </button>
              </div>
            )
          }

          if (param.kind === 'categorical') {
            return (
              <div key={param.name} className="space-y-1">
                <div className="flex justify-between text-slate-300">
                  <span className="font-mono">{param.name}</span>
                </div>
                <select
                  value={val || ''}
                  onChange={(e) => handleCategoricalChange(param.name, e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 rounded p-1 text-slate-200 text-xs"
                >
                  {(param.choices || []).map((choice: any) => (
                    <option key={String(choice)} value={String(choice)}>
                      {String(choice)}
                    </option>
                  ))}
                </select>
              </div>
            )
          }

          if (param.kind === 'int_or_null') {
            const isNull = val === null || val === undefined
            return (
              <div key={param.name} className="space-y-1">
                <div className="flex justify-between items-center text-slate-300">
                  <span className="font-mono">{param.name}</span>
                  <label className="flex items-center gap-1 text-[10px] text-slate-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={isNull}
                      onChange={(e) => handleNumericChange(param, e.target.checked ? null : (param.low || 1))}
                      className="rounded bg-slate-800 border-slate-700 text-cyan-600 focus:ring-0"
                    />
                    <span>No limit (None)</span>
                  </label>
                </div>
                {!isNull && (
                  <div className="flex items-center gap-2">
                    <input
                      type="range"
                      min={param.low ?? 1}
                      max={param.high ?? 100}
                      step={param.step ?? 1}
                      value={val ?? param.low ?? 1}
                      onChange={(e) => handleNumericChange(param, Number(e.target.value))}
                      className="w-full accent-cyan-500"
                    />
                    <span className="font-mono text-cyan-400 w-10 text-right">{val}</span>
                  </div>
                )}
              </div>
            )
          }

          // int or float
          const min = param.low ?? 0
          const max = param.high ?? 100
          const step = param.step ?? (param.kind === 'int' ? 1 : 0.01)

          return (
            <div key={param.name} className="space-y-1">
              <div className="flex justify-between text-slate-300">
                <span className="font-mono">{param.name}</span>
                <span className="font-mono text-cyan-400 font-semibold">{val}</span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={val ?? min}
                  onChange={(e) => handleNumericChange(param, Number(e.target.value))}
                  className="w-full accent-cyan-500"
                />
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={val ?? min}
                  onChange={(e) => handleNumericChange(param, Number(e.target.value))}
                  className="w-16 bg-slate-800 border border-slate-700 text-slate-200 text-right rounded px-1.5 py-0.5 text-xs font-mono"
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
