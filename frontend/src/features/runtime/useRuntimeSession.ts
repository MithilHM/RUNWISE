/**
 * The runtime session: one run, one transport, one reducer.
 *
 * This hook owns the only mutable part of the runtime screen. It starts a run, opens
 * the event stream, folds every event through `applyEvent`, and tears the stream down
 * on stop, on unmount and on failure.
 *
 * Transport choice is the one decision that lives here. Three inputs feed the same
 * reducer:
 *
 *  - the backend SSE stream, when `POST /api/runtime/start` succeeds;
 *  - the local simulation, when the caller asked for demo mode or the backend is
 *    unreachable;
 *  - `GET /api/runtime/{id}/status`, read once before the stream opens, so a page
 *    opened mid-run shows the trials already finished.
 *
 * A failed request never silently becomes fake data. Only a transport-level failure
 * (the API unreachable, or the runtime routes absent) falls back to the simulation,
 * and the fallback always records the real error so the banner can say what happened.
 * A rejected objective is a validation error and is reported as one.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../../api/client'
import { runtimeApi, type AblationResult, type RuntimeObjective, type RuntimeStreamHandle } from '../../api/runtimeApi'
import { ABLATION_MAX_BUDGET, ABLATION_MIN_BUDGET, DEFAULT_OBJECTIVE, RUNTIME_RECONCILE_MS } from './runtimeConstants'
import {
  applyEvent,
  initialRuntimeState,
  reconcileFromStatus,
  type RuntimePhase,
  type RuntimeState,
} from './runtimeModel'
import { createMockAblation, createMockStream, mockRunId } from './mockRuntime'

export interface RuntimeSession {
  state: RuntimeState
  /** Start a run. `simulate` forces the local replay instead of the backend. */
  start: (objective: RuntimeObjective, simulate: boolean) => Promise<void>
  /** Detach the stream. The backend loop keeps running; only the listener stops. */
  stop: () => void
  /** Clear the session and return the screen to its idle state. */
  reset: () => void
  /** Reattach to a run whose stream dropped. */
  resume: () => void
  /** Compare Graph+Laya against uniform random search over the same budget. */
  runAblation: (budget?: number) => Promise<void>
  ablation: AblationResult | null
  ablationError: unknown
  ablationPending: boolean
  /** True when the current ablation result was replayed locally rather than measured. */
  ablationSimulated: boolean
}

/** Transport-level failures are the only ones worth replaying locally. */
function isTransportFailure(error: unknown): boolean {
  if (!(error instanceof ApiError)) return false
  return error.status === 0 || error.status === 404 || error.status === 502 || error.status === 503 || error.status >= 500
}

export function useRuntimeSession(): RuntimeSession {
  const [state, setState] = useState<RuntimeState>(() => initialRuntimeState(defaultObjective(), ''))
  const [ablation, setAblation] = useState<AblationResult | null>(null)
  const [ablationError, setAblationError] = useState<unknown>(null)
  const [ablationPending, setAblationPending] = useState(false)
  /**
   * Whether the last ablation result came from the local simulation.
   *
   * Tracked separately from `state.simulated` because the two do not agree: an
   * ablation can be requested before any run exists, and with no run id there is
   * nothing to ask the server, so it is simulated — on a session whose own event
   * stream was never simulated. Gating the warning on `state.simulated` alone would
   * present a fabricated comparison with no warning at all.
   */
  const [ablationSimulated, setAblationSimulated] = useState(false)

  const streamRef = useRef<RuntimeStreamHandle | null>(null)
  const objectiveRef = useRef<RuntimeObjective>(state.objective)
  const simulatingRef = useRef(false)
  /**
   * Bumped by every start, stop and reset.
   *
   * `start` awaits two network calls before it can attach a stream. Without this,
   * an operator who presses Stop (or Clear, or Start again) during that window has
   * their action silently undone when the pending request resolves and forces the
   * session back to running. Comparing generations after each await makes a
   * superseded start return without touching state.
   */
  const generationRef = useRef(0)
  /* The reconcile poll is a safety net for a dropped SSE connection, not the primary
     path. It stops the moment the run reaches a terminal state — see `clearPoll`. */
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearPoll = useCallback(() => {
    if (pollRef.current !== null) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const closeStream = useCallback(() => {
    streamRef.current?.close()
    streamRef.current = null
    clearPoll()
  }, [clearPoll])

  const openStream = useCallback(
    (runId: string, simulate: boolean) => {
      closeStream()
      const push = (event: Parameters<typeof applyEvent>[1]) => {
        /* A terminal frame is the only thing the stream itself reports, and the
           server reuses that stage for a loop that died. Reading the authoritative
           status once here closes the window in which a failed run wears the green
           "Completed" chip until the next poll happens to land. */
        if (event.stage === 'COMPLETED') {
          clearPoll()
          if (!simulate) {
            void runtimeApi
              .getRuntimeStatus(runId)
              .then((snapshot) => setState((previous) => reconcileFromStatus(previous, snapshot)))
              .catch(() => {
                // The stream already delivered what it had; a failed confirmation read
                // must not replace a good terminal state with an error.
              })
          }
        }
        setState((previous) => applyEvent(previous, event))
      }

      const handle = simulate
        ? createMockStream({ runId, objective: objectiveRef.current }, push)
        : runtimeApi.streamRuntimeEvents(runId, push, (error) => {
            setState((previous) => ({
              ...previous,
              // The loop may still be alive server-side, so this is a listener failure
              // and not a run failure. Keep every trial already rendered.
              phase: previous.phase === 'running' ? 'error' : previous.phase,
              error: error.message,
            }))
          })

      streamRef.current = handle

      if (!simulate) {
        pollRef.current = setInterval(() => {
          void runtimeApi
            .getRuntimeStatus(runId)
            .then((snapshot) => {
              setState((previous) => {
                const next = reconcileFromStatus(previous, snapshot)
                // Terminal from here on: stop polling rather than re-rendering the
                // whole screen every five seconds for the rest of the page's life.
                if (next.phase === 'completed' || next.failed) clearPoll()
                return next
              })
            })
            .catch(() => {
              /* The status endpoint is a net, not a lifeline. A failure here means the
                 connection is already broken, and the stream's own error handler owns
                 reporting that. */
            })
        }, RUNTIME_RECONCILE_MS)
      }
    },
    [clearPoll, closeStream],
  )

  const start = useCallback(
    async (objective: RuntimeObjective, simulate: boolean) => {
      const generation = ++generationRef.current
      closeStream()
      objectiveRef.current = objective
      simulatingRef.current = simulate
      setAblation(null)
      setAblationError(null)
      setAblationSimulated(false)

      if (simulate) {
        const runId = mockRunId()
        setState({ ...initialRuntimeState(objective, runId), phase: 'running', simulated: true })
        openStream(runId, true)
        return
      }

      setState({ ...initialRuntimeState(objective, ''), phase: 'starting' })
      try {
        const started = await runtimeApi.startAutonomousRun(objective)
        // Superseded by a stop, a clear, or a newer start while the call was in flight.
        if (generationRef.current !== generation) return

        const base = initialRuntimeState(objective, started.run_id)

        /* Read the stored status before attaching: the loop may already have run a
           trial or two while the start request was in flight, and reconciling first
           means the tree opens with those trials present rather than flashing empty. */
        let reconciled = base
        try {
          reconciled = reconcileFromStatus(base, await runtimeApi.getRuntimeStatus(started.run_id))
        } catch {
          // The stream backfills every event anyway, so a failed read costs nothing.
        }
        if (generationRef.current !== generation) return

        setState({ ...reconciled, phase: 'running', simulated: false })
        openStream(started.run_id, false)
      } catch (error) {
        if (generationRef.current !== generation) return
        if (!isTransportFailure(error)) {
          setState((previous) => ({
            ...previous,
            phase: 'error',
            error: error instanceof Error ? error.message : String(error),
          }))
          return
        }
        const detail = error instanceof ApiError ? error.details[0]?.message : String(error)
        const runId = mockRunId()
        /* The transport ref must follow the fallback, not the original request. Every
           later decision — reattach, ablation — asks it which backend to talk to, and
           this run id exists nowhere but here. Leaving it false sent reattach and the
           ablation at a real endpoint with a `run_demo_*` id and a guaranteed 404. */
        simulatingRef.current = true
        setState({
          ...initialRuntimeState(objective, runId),
          phase: 'running',
          simulated: true,
          // Recorded rather than swallowed: the banner names the real failure so the
          // replay is never mistaken for a live loop.
          error: `Backend runtime unavailable (${detail ?? 'unknown error'}). Replaying a simulated run locally.`,
        })
        openStream(runId, true)
      }
    },
    [closeStream, openStream],
  )

  const stop = useCallback(() => {
    // Any pending start is now moot; leaving it live would reattach after the stop.
    generationRef.current += 1
    closeStream()
    // Detaching the listener is an operator action, not a fault. It gets its own
    // state and its own wording rather than borrowing the error channel, which is
    // reserved for things that actually went wrong.
    setState((previous) => ({ ...previous, phase: 'stopping' }))
  }, [closeStream])

  const reset = useCallback(() => {
    generationRef.current += 1
    closeStream()
    simulatingRef.current = false
    setAblation(null)
    setAblationError(null)
    setAblationSimulated(false)
    setState(initialRuntimeState(objectiveRef.current, ''))
  }, [closeStream])

  const resume = useCallback(() => {
    const runId = state.runId
    if (!runId) return
    openStream(runId, simulatingRef.current)
    setState((previous) => {
      /* Reattaching makes the server replay the run's whole history, and folding
         skips what it has already applied — the terminal frame included. Forcing
         "running" unconditionally would strand an already-finished run in a running
         state with nothing left to move it on. */
      if (previous.seenEvents.some((id) => id.endsWith('-COMPLETED'))) return previous
      /* Only a dropped stream is cleared by reattaching. A run that reported a
         failure keeps that message: reattaching cannot undo it, and wiping the one
         visible explanation would leave an unexplained error chip. */
      return { ...previous, phase: 'running', error: null }
    })
  }, [openStream, state.runId])

  const runAblation = useCallback(
    async (budget?: number) => {
      const requested = Math.min(ABLATION_MAX_BUDGET, Math.max(ABLATION_MIN_BUDGET, budget ?? state.objective.trial_budget))
      setAblationPending(true)
      setAblationError(null)
      try {
        if (simulatingRef.current || !state.runId) {
          // Compare against the scores this run actually recorded, so the chart and
          // the tree above it cannot disagree. With no run at all there are no scores
          // to compare, and the summary says so.
          const measured = state.nodes.map((node) => node.score)
          setAblationSimulated(true)
          setAblation(createMockAblation(state.objective, measured, requested))
          return
        }
        setAblationSimulated(false)
        setAblation(await runtimeApi.runAblationStudy(state.runId, requested))
      } catch (error) {
        setAblationError(error)
      } finally {
        setAblationPending(false)
      }
    },
    [state.nodes, state.objective, state.runId],
  )

  useEffect(() => () => closeStream(), [closeStream])

  return {
    state,
    start,
    stop,
    reset,
    resume,
    runAblation,
    ablation,
    ablationError,
    ablationPending,
    ablationSimulated,
  }
}

/**
 * The screen's starting objective.
 *
 * Taken from the shared defaults rather than restated, so the idle header and the
 * ablation's default budget cannot drift from what the control bar actually offers.
 */
function defaultObjective(): RuntimeObjective {
  return { ...DEFAULT_OBJECTIVE }
}

export function phaseLabel(phase: RuntimePhase): string {
  switch (phase) {
    case 'idle':
      return 'Idle'
    case 'starting':
      return 'Starting'
    case 'running':
      return 'Running'
    case 'stopping':
      // The loop behind it is still running; only this screen stopped listening.
      // "Stopped" would contradict the banner and the reattach button beside it.
      return 'Detached'
    case 'completed':
      return 'Completed'
    case 'error':
      return 'Attention'
  }
}