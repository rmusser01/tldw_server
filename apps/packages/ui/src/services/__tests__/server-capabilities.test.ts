import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { existsSync, readFileSync } from "node:fs"
import { resolve } from "node:path"

const cacheState = vi.hoisted(() => ({
  values: new Map<string, unknown>()
}))

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getOpenAPISpec: vi.fn(),
  bgRequest: vi.fn(),
  storageGet: vi.fn(async (key: string) => cacheState.values.get(key)),
  storageSet: vi.fn(async (key: string, value: unknown) => {
    cacheState.values.set(key, value)
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) =>
      (mocks.getConfig as (...args: unknown[]) => unknown)(...args),
    getOpenAPISpec: (...args: unknown[]) =>
      (mocks.getOpenAPISpec as (...args: unknown[]) => unknown)(...args)
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (...args: unknown[]) =>
      (mocks.storageGet as (...args: unknown[]) => unknown)(...args),
    set: (...args: unknown[]) =>
      (mocks.storageSet as (...args: unknown[]) => unknown)(...args)
  })
}))

const importCapabilitiesModule = async () =>
  import("@/services/tldw/server-capabilities")

const serverCapabilitiesPathCandidates = [
  "src/services/tldw/server-capabilities.ts",
  "../packages/ui/src/services/tldw/server-capabilities.ts",
  "apps/packages/ui/src/services/tldw/server-capabilities.ts"
]

const serverCapabilitiesSourcePath = serverCapabilitiesPathCandidates.find((candidate) =>
  existsSync(resolve(process.cwd(), candidate))
)

if (!serverCapabilitiesSourcePath) {
  throw new Error("Unable to locate server-capabilities.ts for fallback spec contract test")
}

const openApiGuardPathCandidates = [
  "src/services/tldw/openapi-guard.ts",
  "../packages/ui/src/services/tldw/openapi-guard.ts",
  "apps/packages/ui/src/services/tldw/openapi-guard.ts"
]

const openApiGuardSourcePath = openApiGuardPathCandidates.find((candidate) =>
  existsSync(resolve(process.cwd(), candidate))
)

if (!openApiGuardSourcePath) {
  throw new Error("Unable to locate openapi-guard.ts for route contract test")
}

describe("server capabilities docs-info merge", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useRealTimers()
    cacheState.values.clear()
    mocks.getConfig.mockReset()
    mocks.getOpenAPISpec.mockReset()
    mocks.bgRequest.mockReset()
    mocks.storageGet.mockReset()
    mocks.storageSet.mockReset()
    mocks.storageGet.mockImplementation(async (key: string) =>
      cacheState.values.get(key)
    )
    mocks.storageSet.mockImplementation(async (key: string, value: unknown) => {
      cacheState.values.set(key, value)
    })
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("requires both persona route support and docs-info persona feature flag", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "test-version" },
      paths: {
        "/api/v1/persona/catalog": {},
        "/api/v1/persona/session": {},
        "/api/v1/personalization/profile": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        persona: false,
        personalization: true
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasPersona).toBe(false)
    expect(capabilities.hasPersonalization).toBe(true)
    expect(capabilities.specVersion).toBe("test-version")
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/config/docs-info",
        method: "GET",
        noAuth: true
      })
    )
  })

  it("does not enable persona when docs-info says true but persona routes are missing", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "test-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        persona: true
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasPersona).toBe(false)
  })

  it("detects Persona live-control support from the live sessions route", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "persona-live-control" },
      paths: {
        "/api/v1/persona/live/sessions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasPersona).toBe(true)
    expect(capabilities.hasPersonaLiveControl).toBe(true)
  })

  it("honors docs-info disabling Persona live-control support", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "persona-live-control-disabled" },
      paths: {
        "/api/v1/persona/live/sessions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        hasPersonaLiveControl: false
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasPersona).toBe(true)
    expect(capabilities.hasPersonaLiveControl).toBe(false)
  })

  it("falls back to openapi-only capability detection when docs-info fetch fails", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "test-version" },
      paths: {
        "/api/v1/persona/catalog": {}
      }
    })
    mocks.bgRequest.mockRejectedValue(new Error("docs-info unavailable"))

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasPersona).toBe(true)
  })

  it("detects media capability from ingest jobs endpoint alone", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ingest-jobs-only" },
      paths: {
        "/api/v1/media/ingest/jobs": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasMedia).toBe(true)
  })

  it("detects playlist preflight and keeps worker availability separate from job endpoints", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "bulk-ingest-capabilities" },
      paths: {
        "/api/v1/media/playlists/preflight": {},
        "/api/v1/media/collections": {},
        "/api/v1/media/ingest/jobs": {},
        "/api/v1/media/ingest/jobs/events/stream": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        hasMediaPlaylistPreflight: true,
        hasMediaIngestJobs: true,
        hasMediaIngestJobEvents: true,
        hasMediaIngestWorker: false,
        hasDurableMediaCollections: true,
        hasKnowledgeQaMediaScope: false
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasMedia).toBe(true)
    expect(capabilities.hasMediaPlaylistPreflight).toBe(true)
    expect(capabilities.hasMediaIngestJobs).toBe(true)
    expect(capabilities.hasMediaIngestJobEvents).toBe(true)
    expect(capabilities.hasMediaIngestWorker).toBe(false)
    expect(capabilities.hasDurableMediaCollections).toBe(true)
    expect(capabilities.hasKnowledgeQaMediaScope).toBe(false)
  })

  it("detects ingestion source capability from advertised source routes", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ingestion-sources-version" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/{source_id}": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
  })

  it("merges authenticated local-directory source creation capability", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ingestion-sources-local-directory-version" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/{source_id}": {},
        "/api/v1/ingestion-sources/capabilities": {}
      }
    })
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path === "/api/v1/ingestion-sources/capabilities") {
        return { can_create_local_directory: true }
      }
      return {}
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBe(true)
    const sourceCapabilityCall = mocks.bgRequest.mock.calls.find(
      ([request]) =>
        (request as any)?.path === "/api/v1/ingestion-sources/capabilities"
    )
    expect(sourceCapabilityCall?.[0]).toMatchObject({
      path: "/api/v1/ingestion-sources/capabilities",
      method: "GET"
    })
    expect(sourceCapabilityCall?.[0]).not.toHaveProperty("noAuth")
  })

  it("keeps generic ingestion sources enabled when authenticated source capabilities fail", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ingestion-sources-capabilities-fail" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/{source_id}": {},
        "/api/v1/ingestion-sources/capabilities": {}
      }
    })
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path === "/api/v1/ingestion-sources/capabilities") {
        throw new Error("capability endpoint unavailable")
      }
      return {}
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBeNull()
  })

  it("best-effort probes source capabilities for older authoritative source specs", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ingestion-sources-without-entitlement-route" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/{source_id}": {}
      }
    })
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path === "/api/v1/ingestion-sources/capabilities") {
        throw new Error("capability endpoint unavailable")
      }
      return {}
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBeNull()
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) =>
          (request as any)?.path === "/api/v1/ingestion-sources/capabilities"
      )
    ).toBe(true)
  })

  it("does not call authenticated source capabilities when ingestion sources are absent", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "no-ingestion-sources-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(false)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBe(false)
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) =>
          (request as any)?.path === "/api/v1/ingestion-sources/capabilities"
      )
    ).toBe(false)
  })

  it("detects web clipper capability from the advertised web clipper routes", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "web-clipper-version" },
      paths: {
        "/api/v1/web-clipper/save": {},
        "/api/v1/web-clipper/{clip_id}": {},
        "/api/v1/web-clipper/{clip_id}/enrichments": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasWebClipper).toBe(true)
  })

  it("requires the clipper save route before enabling web clipper capability", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "web-clipper-read-only-version" },
      paths: {
        "/api/v1/web-clipper/{clip_id}": {},
        "/api/v1/web-clipper/{clip_id}/enrichments": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasWebClipper).toBe(false)
  })

  it("does not infer web clipper capability from fallback discovery alone", async () => {
    mocks.getOpenAPISpec.mockRejectedValue(new Error("openapi unavailable"))
    mocks.bgRequest.mockRejectedValue(new Error("docs-info unavailable"))

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasWebClipper).toBe(false)
    expect(capabilities.specSource).toBe("fallback")
  })

  it("checks authenticated source capabilities through the bundled fallback spec", async () => {
    mocks.getOpenAPISpec.mockRejectedValue(new Error("openapi unavailable"))
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path === "/api/v1/ingestion-sources/capabilities") {
        return { can_create_local_directory: true }
      }
      throw new Error("docs-info unavailable")
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBe(true)
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) =>
          (request as any)?.path === "/api/v1/ingestion-sources/capabilities"
      )
    ).toBe(true)
  })

  it("derives hasAudio from STT-only support while keeping TTS/voice flags explicit", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-split-stt" },
      paths: {
        "/api/v1/audio/transcriptions": {},
        "/api/v1/audio/transcriptions/health": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasStt).toBe(true)
    expect(capabilities.hasTts).toBe(false)
    expect(capabilities.hasVoiceChat).toBe(false)
    expect(capabilities.hasAudio).toBe(true)
  })

  it("derives hasAudio from TTS-only support", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-split-tts" },
      paths: {
        "/api/v1/audio/speech": {},
        "/api/v1/audio/health": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasStt).toBe(false)
    expect(capabilities.hasTts).toBe(true)
    expect(capabilities.hasVoiceChat).toBe(false)
    expect(capabilities.hasAudio).toBe(true)
  })

  it("keeps legacy hasAudio true for voice-chat-only route exposure", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-split-voice-chat" },
      paths: {
        "/api/v1/audio/chat/stream": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasAudio).toBe(true)
  })

  it("infers voice chat support from combined STT and TTS routes", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-split-stt-tts" },
      paths: {
        "/api/v1/audio/transcriptions": {},
        "/api/v1/audio/speech": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasStt).toBe(true)
    expect(capabilities.hasTts).toBe(true)
    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasAudio).toBe(true)
  })

  it("keeps strict voice conversation transport off for STT plus TTS only authoritative specs", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-split-stt-tts-authoritative" },
      paths: {
        "/api/v1/audio/transcriptions": {},
        "/api/v1/audio/speech": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasVoiceConversationTransport).toBe(false)
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("keeps strict voice conversation transport off for fallback-spec-only discovery", async () => {
    mocks.getOpenAPISpec.mockRejectedValue(new Error("openapi unavailable"))
    mocks.bgRequest.mockRejectedValue(new Error("docs-info unavailable"))

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasVoiceConversationTransport).toBe(false)
    expect(capabilities.specSource).toBe("fallback")
  })

  it("enables strict voice conversation transport when the authoritative spec exposes the stream route", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-voice-transport" },
      paths: {
        "/api/v1/audio/chat/stream": {},
        "/api/v1/audio/transcriptions": {},
        "/api/v1/audio/speech": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasVoiceConversationTransport).toBe(true)
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("treats docs-info audio flags as authoritative when the spec omits websocket paths", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-docs-info-authoritative" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        hasAudio: true,
        hasStt: true,
        hasTts: true,
        hasVoiceChat: true,
        hasVoiceConversationTransport: true
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasAudio).toBe(true)
    expect(capabilities.hasStt).toBe(true)
    expect(capabilities.hasTts).toBe(true)
    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasVoiceConversationTransport).toBe(true)
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("lets docs-info explicitly disable strict voice conversation transport", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-docs-info-transport-off" },
      paths: {
        "/api/v1/audio/chat/stream": {},
        "/api/v1/audio/transcriptions": {},
        "/api/v1/audio/speech": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        hasVoiceConversationTransport: false
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasVoiceChat).toBe(true)
    expect(capabilities.hasVoiceConversationTransport).toBe(false)
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("does not reuse persisted V1 capability payloads for the new cache contract", async () => {
    cacheState.values.set("__tldwServerCapabilitiesCacheV1", {
      key: "http://127.0.0.1:8000::single-user",
      fetchedAt: Date.now(),
      capabilities: {
        hasChat: true
      }
    })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "audio-voice-transport-cache" },
      paths: {
        "/api/v1/audio/chat/stream": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(1)
    expect(capabilities.hasVoiceConversationTransport).toBe(true)
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("does not reuse persisted V2 capability payloads after the cache contract changes", async () => {
    cacheState.values.set("__tldwServerCapabilitiesCacheV2", {
      key: "http://127.0.0.1:8000::single-user",
      fetchedAt: Date.now(),
      capabilities: {
        hasChat: true
      }
    })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "web-clipper-cache-v3" },
      paths: {
        "/api/v1/web-clipper/save": {},
        "/api/v1/web-clipper/{clip_id}": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(1)
    expect(capabilities.hasWebClipper).toBe(true)
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("does not reuse persisted V3 capability payloads after the cache contract changes", async () => {
    cacheState.values.set("__tldwServerCapabilitiesCacheV3", {
      key: "http://127.0.0.1:8000::single-user",
      fetchedAt: Date.now(),
      capabilities: {
        hasChat: true
      }
    })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "cache-v4-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(1)
    expect(capabilities.specVersion).toBe("cache-v4-version")
    expect(capabilities.specSource).toBe("authoritative")
  })

  it("does not reuse persisted V4 capability payloads after entitlement unknown-state support", async () => {
    cacheState.values.set("__tldwServerCapabilitiesCacheV4", {
      key: "server:17azy7u:auth:single-user:org:none:user:single-user:key:none",
      fetchedAt: Date.now(),
      capabilities: {
        hasIngestionSources: true,
        canCreateLocalDirectoryIngestionSource: false
      }
    })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "cache-v5-entitlement-version" },
      paths: {
        "/api/v1/ingestion-sources": {}
      }
    })
    mocks.bgRequest.mockImplementation(async (request: any) => {
      if (request.path === "/api/v1/ingestion-sources/capabilities") {
        return { can_create_local_directory: true }
      }
      return {}
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(1)
    expect(capabilities.specVersion).toBe("cache-v5-entitlement-version")
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBe(true)
  })

  it("detects presentation studio and render capability from docs-info", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "presentation-studio-docs-info" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({
      capabilities: {
        hasSlides: true,
        hasPresentationStudio: true,
        hasPresentationRender: true
      }
    })

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasSlides).toBe(true)
    expect(capabilities.hasPresentationStudio).toBe(true)
    expect(capabilities.hasPresentationRender).toBe(true)
  })

  it("detects presentation studio render capability from advertised slide routes", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "presentation-studio-routes" },
      paths: {
        "/api/v1/slides/presentations": {},
        "/api/v1/slides/presentations/{presentation_id}": {},
        "/api/v1/slides/presentations/{presentation_id}/render-jobs": {},
        "/api/v1/slides/render-jobs/{job_id}": {},
        "/api/v1/slides/presentations/{presentation_id}/render-artifacts": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasSlides).toBe(true)
    expect(capabilities.hasPresentationStudio).toBe(true)
    expect(capabilities.hasPresentationRender).toBe(true)
  })

  it("reuses a fresh cached capability payload for repeated calls", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "cached-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const first = await getServerCapabilities()
    const second = await getServerCapabilities()

    expect(first.specVersion).toBe("cached-version")
    expect(second.specVersion).toBe("cached-version")
    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("refreshes capabilities after cache TTL expires", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-02-11T10:50:00.000Z"))

    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ttl-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    await getServerCapabilities()
    vi.setSystemTime(new Date("2026-02-11T10:56:01.000Z"))
    await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(2)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("keeps cache entries isolated by server URL", async () => {
    mocks.getConfig
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user"
      })
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8100",
        authMode: "single-user"
      })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "multi-server-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    await getServerCapabilities()
    await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(2)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("keeps cache entries isolated by auth scope, not just server URL and auth mode", async () => {
    mocks.getConfig
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "first-user-key"
      })
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "second-user-key"
      })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "auth-scope-cache-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    await getServerCapabilities()
    await getServerCapabilities()

    expect(mocks.getOpenAPISpec).toHaveBeenCalledTimes(2)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("tracks cache hit diagnostics across miss -> network -> memory hit", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "diag-version" },
      paths: {
        "/api/v1/chat/completions": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities, getServerCapabilitiesCacheDiagnostics } =
      await importCapabilitiesModule()

    await getServerCapabilities()
    await getServerCapabilities()

    const diagnostics = getServerCapabilitiesCacheDiagnostics()
    expect(diagnostics.calls).toBe(2)
    expect(diagnostics.networkFetches).toBe(1)
    expect(diagnostics.inMemoryHits).toBe(1)
    expect(diagnostics.persistedHits).toBe(0)
    expect(diagnostics.lastSource).toBe("in-memory")
  })

  it("records in-flight dedupe hits for concurrent requests", async () => {
    let resolveSpec: ((value: unknown) => void) | null = null
    const openApiDeferred = new Promise((resolve) => {
      resolveSpec = resolve
    })
    mocks.getOpenAPISpec.mockReturnValue(openApiDeferred)
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities, getServerCapabilitiesCacheDiagnostics } =
      await importCapabilitiesModule()

    const p1 = getServerCapabilities()
    const p2 = getServerCapabilities()
    resolveSpec?.({
      info: { version: "dedupe-version" },
      paths: { "/api/v1/chat/completions": {} }
    })

    await Promise.all([p1, p2])
    const diagnostics = getServerCapabilitiesCacheDiagnostics()
    expect(diagnostics.calls).toBe(2)
    expect(diagnostics.networkFetches).toBe(1)
    expect(diagnostics.inFlightHits).toBe(1)
  })

  it("keeps fallback spec aligned with TTS voice route checks", () => {
    const source = readFileSync(resolve(process.cwd(), serverCapabilitiesSourcePath), "utf8")

    const fallbackPathsBlock = source.match(
      /const fallbackSpec = \{[\s\S]*?paths:\s*Object\.fromEntries\(\s*\[([\s\S]*?)\]\.map/
    )?.[1]

    expect(fallbackPathsBlock).toContain('"/api/v1/audio/voices/catalog"')
    expect(fallbackPathsBlock).toContain('"/api/v1/audio/voice-conversion"')
    expect(fallbackPathsBlock).toContain('"/api/v1/audio/tts/providers/{provider}/unload"')
  })

  it("keeps strict client path metadata aligned with Chatterbox voice conversion", () => {
    const source = readFileSync(resolve(process.cwd(), openApiGuardSourcePath), "utf8")

    expect(source).toContain('| "/api/v1/audio/voice-conversion"')
    expect(source).toContain('| "/api/v1/audio/tts/providers/{provider}/unload"')
  })
})
