import React from 'react'
import type { ModelSpec } from '../../api/types'


interface ModelPickerProps {
  models: ModelSpec[]
  selectedModelIds: string[]
  onToggleModel: (modelId: string) => void
  activeModelId: string | null
  onSelectActiveModel: (modelId: string) => void
}

export const ModelPicker: React.FC<ModelPickerProps> = ({
  models,
  selectedModelIds,
  onToggleModel,
  activeModelId,
  onSelectActiveModel,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3">
      <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
        Models ({selectedModelIds.length} active)
      </div>
      <div className="space-y-1.5">
        {models.map((model) => {
          const isSelected = selectedModelIds.includes(model.id)
          const isActive = activeModelId === model.id
          return (
            <div
              key={model.id}
              className={`flex items-center justify-between p-2 rounded border cursor-pointer transition text-xs ${
                isActive
                  ? 'bg-slate-800/90 border-cyan-500/50 text-white'
                  : 'bg-slate-950/60 border-slate-800/80 text-slate-300 hover:border-slate-700'
              }`}
              onClick={() => onSelectActiveModel(model.id)}
              title={model.description}
            >
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={(e) => {
                    e.stopPropagation()
                    onToggleModel(model.id)
                  }}
                  className="rounded bg-slate-800 border-slate-700 text-cyan-600 focus:ring-0 cursor-pointer"
                />
                <span className="font-medium">{model.name}</span>
              </div>
              <span className="text-[10px] text-slate-500 font-mono">{model.library}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
