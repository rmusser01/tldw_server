import { afterEach, expect, it, vi } from 'vitest'
import { tldwRequest } from '../request-core'

afterEach(() => vi.unstubAllGlobals())
const config = {
  serverUrl: 'http://127.0.0.1:8000',
  authMode: 'single-user',
  apiKey: 'test-memory-key'
}
const request = (response: Response) =>
  tldwRequest(
    {
      path: '/api/v1/media/1/file',
      responseType: 'arrayBuffer',
      maxResponseBytes: 4
    },
    { getConfig: async () => config, fetchFn: vi.fn(async () => response) }
  )

it('rejects and cancels an advertised oversized response before reading bytes', async () => {
  const cancelled = vi.fn()
  const response = new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array(100))
        controller.close()
      },
      cancel: cancelled
    }),
    { headers: { 'Content-Length': '100' } }
  )
  const result = await request(response)
  expect(result).toMatchObject({ ok: false, code: 'RESPONSE_TOO_LARGE' })
  expect(cancelled).toHaveBeenCalledOnce()
})

it.each([undefined, '1'])('bounds a streamed body with Content-Length %s', async (length) => {
  const cancelled = vi.fn()
  const response = new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new Uint8Array([1, 2, 3]))
        controller.enqueue(new Uint8Array([4, 5, 6]))
        controller.enqueue(new Uint8Array([7]))
        controller.close()
      },
      cancel: cancelled
    }),
    { headers: length ? { 'Content-Length': length } : {} }
  )
  const result = await request(response)
  expect(result).toMatchObject({ ok: false, code: 'RESPONSE_TOO_LARGE' })
  expect(cancelled).toHaveBeenCalledOnce()
})

it('returns a bounded binary response with existing authenticated transport', async () => {
  const response = new Response(new Uint8Array([1, 2, 3, 4]))
  const fetchFn = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => response)
  const result = await tldwRequest(
    { path: '/api/v1/media/1/file', responseType: 'arrayBuffer', maxResponseBytes: 4 },
    { getConfig: async () => config, fetchFn }
  )
  expect(Array.from(new Uint8Array(result.data))).toEqual([1, 2, 3, 4])
  expect(new Headers(fetchFn.mock.calls[0][1]?.headers).get('X-API-KEY')).toBe('test-memory-key')
})

it('bounds unsuccessful response bodies before parsing their error details', async () => {
  const result = await request(new Response('oversized error', { status: 500 }))
  expect(result).toMatchObject({ ok: false, code: 'RESPONSE_TOO_LARGE' })
})

it('preserves small JSON error details under the binary response limit', async () => {
  const result = await tldwRequest(
    { path: '/api/v1/media/1/file', responseType: 'arrayBuffer', maxResponseBytes: 100 },
    {
      getConfig: async () => config,
      fetchFn: vi.fn(
        async () =>
          new Response('{"detail":"missing"}', {
            status: 404,
            headers: { 'Content-Type': 'application/json' }
          })
      )
    }
  )
  expect(result).toMatchObject({ ok: false, status: 404, error: 'missing' })
})

it('coalesces small streamed chunks without changing their binary contents', async () => {
  let next = 0
  const response = new Response(
    new ReadableStream({
      pull(controller) {
        if (next === 10_000) controller.close()
        else controller.enqueue(new Uint8Array([next++ % 256]))
      }
    })
  )
  const result = await tldwRequest(
    { path: '/api/v1/media/1/file', responseType: 'arrayBuffer', maxResponseBytes: 10_000 },
    {
      getConfig: async () => config,
      fetchFn: vi.fn(async () => response)
    }
  )
  expect(new Uint8Array(result.data)).toEqual(
    Uint8Array.from({ length: 10_000 }, (_, i) => i % 256)
  )
})
