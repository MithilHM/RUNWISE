import React, { useState } from 'react'
import type { DatasetProfile } from '../../api/types'


interface UploadStepProps {
  onDatasetLoaded: (profile: DatasetProfile) => void
  currentProfile: DatasetProfile | null
}

export const UploadStep: React.FC<UploadStepProps> = ({ onDatasetLoaded, currentProfile }) => {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmingTarget, setConfirmingTarget] = useState(false)

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/v1/datasets', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail?.[0]?.message || 'Failed to upload dataset')
      }
      const profile: DatasetProfile = await response.json()
      onDatasetLoaded(profile)
    } catch (err: any) {
      setError(err.message || 'Error uploading dataset')
    } finally {
      setUploading(false)
    }
  }

  const handleConfirmTarget = async (newTarget: string) => {
    if (!currentProfile) return
    setConfirmingTarget(true)
    try {
      const response = await fetch(`/api/v1/datasets/${currentProfile.id}/confirm-target`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: newTarget }),
      })
      if (!response.ok) {
        throw new Error('Failed to confirm target')
      }
      const updated: DatasetProfile = await response.json()
      onDatasetLoaded(updated)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setConfirmingTarget(false)
    }
  }

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-3 text-xs text-slate-200">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <label className="cursor-pointer bg-cyan-600 hover:bg-cyan-500 text-white font-medium px-3 py-1.5 rounded transition inline-flex items-center gap-2">
            <span>{uploading ? 'Profiling...' : '📁 Intake Static CSV'}</span>
            <input
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileUpload}
              disabled={uploading}
            />
          </label>

          {currentProfile ? (
            <div className="flex items-center gap-2 text-slate-300">
              <span className="font-semibold text-cyan-400">{currentProfile.name}</span>
              <span className="text-slate-500">•</span>
              <span className="uppercase px-1.5 py-0.5 rounded text-[10px] bg-slate-800 border border-slate-700">
                {currentProfile.task}
              </span>
              <span className="text-slate-500">•</span>
              <span>{currentProfile.n_rows.toLocaleString()} rows × {currentProfile.n_cols} cols</span>
            </div>
          ) : (
            <span className="text-slate-400 italic">No dataset loaded. Drop in one static CSV to start.</span>
          )}
        </div>

        {currentProfile && (
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Target Column:</span>
            <select
              className="bg-slate-800 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs focus:ring-1 focus:ring-cyan-500"
              value={currentProfile.columns.find((c) => c.role === 'target')?.name || ''}
              onChange={(e) => handleConfirmTarget(e.target.value)}
              disabled={confirmingTarget}
            >
              {currentProfile.columns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.dtype})
                </option>
              ))}
            </select>
            {currentProfile.target_inferred ? (
              <span className="text-[10px] bg-amber-950/70 border border-amber-800 text-amber-300 px-1.5 py-0.5 rounded">
                Inferred
              </span>
            ) : (
              <span className="text-[10px] bg-emerald-950/70 border border-emerald-800 text-emerald-300 px-1.5 py-0.5 rounded">
                Confirmed
              </span>
            )}
          </div>
        )}
      </div>

      {error && <div className="mt-2 text-rose-400 text-[11px]">{error}</div>}
    </div>
  )
}
