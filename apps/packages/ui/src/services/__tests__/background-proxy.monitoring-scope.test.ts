import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock('wxt/browser', () => ({ browser: { runtime: { id: null } } }))
vi.mock('@/utils/safe-storage', () => ({
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(async () => undefined),
    remove: vi.fn(async () => undefined)
  })
}))
vi.mock('@/services/tldw/runtime-auth-override', () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { bgRequest } from '../background-proxy'
import { isServicePromptRequestPath } from '../tldw/service-prompt-scope-error'

const config = (user = 7) => ({
  serverUrl: 'https://notes.test',
  authMode: 'multi-user' as const,
  accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const request = (
  path = '/api/v1/monitoring/alerts?user_id=7&source=notes.create&unread_only=true&limit=20'
) =>
  bgRequest({
    path: path as '/api/v1/monitoring/alerts',
    method: 'GET',
    servicePromptConfig: {
      serverUrl: 'https://notes.test',
      authMode: 'multi-user',
      expectedUserId: 7
    },
    headers: { 'X-TLDW-Expected-User-ID': '7' }
  })

describe('scoped monitoring transport used after saving a Note', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) =>
      key === 'tldwConfig' ? config() : null
    )
    boundary.fetch.mockImplementation(
      async () =>
        new Response(JSON.stringify({ items: [] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        })
    )
    vi.stubGlobal('fetch', boundary.fetch)
  })
  afterEach(() => vi.unstubAllGlobals())

  it('dispatches the exact monitoring GET through the real transport with owner checks', async () => {
    await expect(request()).resolves.toEqual({ items: [] })
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(new URL(String(url)).searchParams.get('user_id')).toBe('7')
    expect(new Headers(init.headers).get('X-TLDW-Expected-User-ID')).toBe('7')
    expect(new Headers(init.headers).get('Authorization')).toBe(
      `Bearer ${config().accessToken}`
    )
  })

  it('rejects an account replacement while credentials are being loaded before dispatch', async () => {
    let release!: (value: unknown) => void
    const pendingConfig = new Promise((resolve) => {
      release = resolve
    })
    boundary.get.mockImplementation(async (key: string) =>
      key === 'tldwConfig' ? pendingConfig : null
    )
    const pending = request()
    const rejected = expect(pending).rejects.toMatchObject({ status: 412 })
    release(config(8))
    await rejected
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each([
    '/api/v1/monitoring/alerts/',
    '/api/v1/monitoring//alerts',
    '/api/v1/monitoring/../monitoring/alerts',
    '/api/v1/monitoring/%2e%2e/alerts',
    '/api/v1/monitoring%2falerts',
    '/api/v1/monitoring\\alerts',
    '/api/v1/monitoring/alerts/other',
    'https://other.test/api/v1/monitoring/alerts'
  ])(
    'rejects malformed or expanded monitoring path %s before dispatch',
    async (path) => {
      await expect(request(path)).rejects.toThrow(/Service Prompt config/)
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )

  it.each(['POST', 'PUT', 'DELETE'])(
    'does not authorize monitoring %s',
    (method) => {
      expect(
        isServicePromptRequestPath('/api/v1/monitoring/alerts', method)
      ).toBe(false)
    }
  )
})
