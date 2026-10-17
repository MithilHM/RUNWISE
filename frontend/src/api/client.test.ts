import { describe, expect, it, vi, beforeEach } from 'vitest'
import { ApiError, api } from './client'

/**
 * The client is the only place that knows the backend's error envelope. These tests
 * pin the two behaviours the UI depends on: a 4xx/5xx always becomes an ApiError the
 * screens can render, and a validation failure names the offending field.
 */

function mockFetch(status: number, body: unknown) {
  return vi.fn().mockResolvedValue(
    new Response(body === undefined ? '' : JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
}

/** Await a request that must fail and return the ApiError it produced. */
async function captureError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise
  } catch (error) {
    return error as ApiError
  }
  throw new Error('expected the request to fail, but it resolved')
}

describe('api error handling', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  it('normalises the structured detail list the backend returns', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch(404, { detail: [{ code: 'experiment_not_found', message: "No experiment with id 'x'.", field: null }] }),
    )
    await expect(api.getExperiment('x')).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
      code: 'experiment_not_found',
    })
  })

  it('normalises a plain string detail so no screen has to guess', async () => {
    vi.stubGlobal('fetch', mockFetch(400, { detail: 'Experiment not found.' }))
    const error = await captureError(api.getExperiment('x'))
    expect(error).toBeInstanceOf(ApiError)
    expect(error.details[0]).toEqual({ code: 'error', message: 'Experiment not found.', field: null })
  })

  it('names the offending field on a validation failure', async () => {
    vi.stubGlobal(
      'fetch',
      mockFetch(422, { detail: [{ code: 'validation_error', message: 'objective: field required', field: 'objective' }] }),
    )
    const error = await captureError(api.createRecommendation('p1', { objective: '' }))
    expect(error.status).toBe(422)
    expect(error.details[0].field).toBe('objective')
    expect(error.isTransient).toBe(false)
  })

  it('marks a 503 as transient and a 4xx as not', async () => {
    vi.stubGlobal('fetch', mockFetch(503, { detail: [{ code: 'graph_unavailable', message: 'no graph', field: null }] }))
    const transient = await captureError(api.listProjects())
    expect(transient.isTransient).toBe(true)

    vi.stubGlobal('fetch', mockFetch(400, { detail: [{ code: 'bad_request', message: 'no', field: null }] }))
    const permanent = await captureError(api.listProjects())
    expect(permanent.isTransient).toBe(false)
  })

  it('explains an unreachable backend instead of leaking a network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))
    const error = await captureError(api.listProjects())
    expect(error).toBeInstanceOf(ApiError)
    expect(error.code).toBe('network_error')
    expect(error.message).toContain('uvicorn')
  })
})

describe('api query strings', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch(200, []))
  })

  it('drops empty filters so the backend sees only real constraints', async () => {
    await api.listExperiments('p1', { model_family: '', status: undefined, limit: 50 })
    const url = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as string
    expect(url).toContain('limit=50')
    expect(url).not.toContain('model_family')
    expect(url).not.toContain('status')
  })

  it('keeps a filter that is set', async () => {
    await api.listExperiments('p1', { quality_status: 'needs_review' })
    const url = (globalThis.fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0] as string
    expect(url).toContain('quality_status=needs_review')
  })
})
