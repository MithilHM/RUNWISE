import type { ComparabilityStatus, EvidenceStatus, Severity } from '../api/types'
import { titleCase } from '../lib/format'

/**
 * Epistemic grade chips.
 *
 * `observed`, `user_asserted`, `inferred`, `disputed`, `superseded` and `unresolved`
 * are different trust claims, so each gets its own colour and label. Merging them
 * into a single "trusted" colour would make the product assert confidence the data
 * does not support (HANDOFF.md, UI rule 1).
 */

const EVIDENCE_STYLES: Record<EvidenceStatus, string> = {
  observed: 'border-good-400/50 bg-good-400/10 text-good-400',
  confirmed: 'border-good-400/50 bg-good-400/10 text-good-400',
  user_asserted: 'border-signal-400/50 bg-signal-400/10 text-signal-400',
  inferred: 'border-ink-600 bg-ink-800 text-ink-300',
  disputed: 'border-danger-400/50 bg-danger-400/10 text-danger-400',
  superseded: 'border-ink-600 bg-ink-800 text-ink-400 line-through',
  unresolved: 'border-warn-400/50 bg-warn-400/10 text-warn-400',
}

const EVIDENCE_HINTS: Record<EvidenceStatus, string> = {
  observed: 'Measured and stored in the graph',
  confirmed: 'Signed off by a researcher',
  user_asserted: 'Stated by a researcher, not independently verified',
  inferred: 'Derived by the system, not observed',
  disputed: 'Challenged by an open correction',
  superseded: 'Replaced by a newer record',
  unresolved: 'Open question; no verdict',
}

export function EvidenceStatusChip({ status }: { status: EvidenceStatus | string | null }) {
  if (!status) return null
  const key = (status in EVIDENCE_STYLES ? status : 'inferred') as EvidenceStatus
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide ${EVIDENCE_STYLES[key]}`}
      title={EVIDENCE_HINTS[key]}
    >
      {titleCase(key)}
    </span>
  )
}

const QUALITY_STYLES: Record<string, string> = {
  valid: 'border-good-400/40 text-good-400',
  observed: 'border-good-400/40 text-good-400',
  needs_review: 'border-warn-400/50 text-warn-400',
  disputed: 'border-danger-400/50 text-danger-400',
  unreliable: 'border-danger-400/50 text-danger-400',
  leakage_detected: 'border-danger-400/50 text-danger-400',
  superseded: 'border-ink-600 text-ink-400',
  unresolved: 'border-warn-400/50 text-warn-400',
}

/** Review state of a record. `needs_review` means it must not anchor a conclusion. */
export function QualityBadge({ status }: { status: string | null | undefined }) {
  if (!status || status === 'valid') return null
  const style = QUALITY_STYLES[status] ?? 'border-ink-600 text-ink-400'
  return (
    <span className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10px] ${style}`}>
      {titleCase(status)}
    </span>
  )
}

const COMPARABILITY_STYLES: Record<ComparabilityStatus, string> = {
  COMPARABLE: 'border-good-400/50 bg-good-400/10 text-good-400',
  COMPARABLE_WITH_CAVEATS: 'border-warn-400/50 bg-warn-400/10 text-warn-400',
  NOT_COMPARABLE: 'border-danger-400/50 bg-danger-400/10 text-danger-400',
  INSUFFICIENT_INFORMATION: 'border-ink-600 bg-ink-800 text-ink-300',
}

const COMPARABILITY_HINTS: Record<ComparabilityStatus, string> = {
  COMPARABLE: 'Same dataset, split, protocol and metric: a controlled comparison',
  COMPARABLE_WITH_CAVEATS: 'Usable as a baseline, with the caveats listed',
  NOT_COMPARABLE: 'A material difference blocks the comparison',
  INSUFFICIENT_INFORMATION: 'A material field was never recorded, so no verdict is possible',
}

export function ComparabilityBadge({ status }: { status: ComparabilityStatus | string }) {
  const key = (status in COMPARABILITY_STYLES ? status : 'INSUFFICIENT_INFORMATION') as ComparabilityStatus
  return (
    <span
      className={`inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-[10px] ${COMPARABILITY_STYLES[key]}`}
      title={COMPARABILITY_HINTS[key]}
    >
      {titleCase(key)}
    </span>
  )
}

const SEVERITY_STYLES: Record<Severity, string> = {
  blocker: 'bg-danger-400/15 text-danger-400',
  major: 'bg-warn-400/15 text-warn-400',
  minor: 'bg-ink-700 text-ink-300',
  info: 'bg-ink-800 text-ink-400',
}

export function SeverityDot({ severity }: { severity: Severity | string }) {
  const key = (severity in SEVERITY_STYLES ? severity : 'info') as Severity
  return (
    <span
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full ${SEVERITY_STYLES[key].split(' ')[0]}`}
      title={`${key} finding`}
    />
  )
}

export function IdChip({ id, title }: { id: string | null | undefined; title?: string }) {
  if (!id) return <span className="text-ink-500">—</span>
  return (
    <span className="mono-id" title={title ?? id}>
      {id}
    </span>
  )
}
