import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  bgUpload: vi.fn(),
  bgStream: vi.fn(),
  tldwRequest: vi.fn(),
  storage: new Map<string, unknown>(),
  sessionStorage: new Map<string, unknown>(),
  storageRemoveError: null as Error | null,
  envApiKey: vi.fn<() => string | null>()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgUpload: (...args: unknown[]) => mocks.bgUpload(...args),
  bgStream: (...args: unknown[]) => mocks.bgStream(...args)
}))

vi.mock("@/services/tldw/request-core", () => ({
  tldwRequest: (...args: unknown[]) => mocks.tldwRequest(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: (options?: { area?: string }) => {
    const values = options?.area === "session" ? mocks.sessionStorage : mocks.storage
    return {
      get: vi.fn(async (key: string) => values.get(key)),
      set: vi.fn(async (key: string, value: unknown) => {
        values.set(key, value)
      }),
      remove: vi.fn(async (key: string) => {
        if (options?.area !== "session" && mocks.storageRemoveError) {
          throw mocks.storageRemoveError
        }
        values.delete(key)
      })
    }
  },
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))

import { TldwApiClient } from "@/services/tldw/TldwApiClient"
import {
  activateCookieSessionConfig,
  clearRuntimeAuthOverride,
  setRuntimeSingleUserApiKeyOverride,
  isCookieSessionConfigInvalidated
} from "@/services/tldw/runtime-auth-override"

describe("TldwApiClient quickstart auth bootstrap", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", undefined)
    vi.stubEnv("NEXT_PUBLIC_API_URL", undefined)
    vi.stubEnv("NEXT_PUBLIC_X_API_KEY", undefined)
    vi.stubEnv("VITE_TLDW_API_KEY", undefined)
    vi.stubEnv("VITE_TLDW_DEFAULT_API_KEY", undefined)
    mocks.envApiKey.mockReset()
    mocks.envApiKey.mockReturnValue(null)
    vi.spyOn(
      TldwApiClient.prototype as any,
      "getEnvApiKey"
    ).mockImplementation(() => mocks.envApiKey())
    mocks.bgRequest.mockReset()
    mocks.bgUpload.mockReset()
    mocks.bgStream.mockReset()
    mocks.tldwRequest.mockReset()
    mocks.storage.clear()
    mocks.sessionStorage.clear()
    mocks.storageRemoveError = null
    window.localStorage.clear()
    activateCookieSessionConfig()
    clearRuntimeAuthOverride()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    window.localStorage.clear()
    activateCookieSessionConfig()
    clearRuntimeAuthOverride()
  })


  it.each([
    ['foreign-origin cookie marker', { authMode: 'single-user' as const, authSource: 'cookie-session' as const, serverUrl: 'https://foreign.example.test' }],
    ['wrong-mode cookie marker', { authMode: 'multi-user' as const, authSource: 'cookie-session' as const, serverUrl: window.location.origin }]
  ])('does not probe with %s rejected by actual request authentication', async (_label, stored) => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart'
    mocks.storage.set('tldwConfig', stored)
    const { tldwClient } = await import('@/services/tldw/TldwApiClient')
    await tldwClient.initialize()
    const normalized = await tldwClient.getConfig()
    expect(normalized).toMatchObject(stored)
    await expect(tldwClient.ensureConfigForRequest(true)).rejects.toThrow()
    vi.spyOn(tldwClient, 'getOpenAPISpec').mockResolvedValue({info: {version:'review-cookie-guard'}, paths: {'/api/v1/ingestion-sources':{}, '/api/v1/ingestion-sources/capabilities':{}}})
    mocks.bgRequest.mockResolvedValue({})
    const { getServerCapabilities } = await import('@/services/tldw/server-capabilities')
    const capabilities = await getServerCapabilities({forceRefresh:true})
    expect(capabilities.hasIngestionSources).toBe(true)
    expect(mocks.bgRequest.mock.calls.filter(([r]) => r.path === '/api/v1/ingestion-sources/capabilities')).toHaveLength(0)
  })

  it('allows the actual exact-origin active cookie configuration', async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = 'quickstart'
    const stored = {authMode: 'single-user' as const, authSource:'cookie-session' as const, serverUrl:window.location.origin}
    mocks.storage.set('tldwConfig',stored)
    const { tldwClient } = await import('@/services/tldw/TldwApiClient')
    await tldwClient.initialize()
    await expect(tldwClient.ensureConfigForRequest(true)).resolves.toMatchObject(stored)
    vi.spyOn(tldwClient,'getOpenAPISpec').mockResolvedValue({info:{version:'review-cookie-positive'},paths:{'/api/v1/ingestion-sources':{},'/api/v1/ingestion-sources/capabilities':{}}})
    mocks.bgRequest.mockImplementation(async (r:{path:string})=>r.path==='/api/v1/ingestion-sources/capabilities'?{can_create_local_directory:true}:{})
    const { getServerCapabilities } = await import('@/services/tldw/server-capabilities')
    const capabilities=await getServerCapabilities({forceRefresh:true})
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBe(true)
    expect(mocks.bgRequest.mock.calls.filter(([r])=>r.path==='/api/v1/ingestion-sources/capabilities')).toHaveLength(1)
  })
})
