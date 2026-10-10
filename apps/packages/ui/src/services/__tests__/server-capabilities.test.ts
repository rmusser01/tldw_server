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
  isActiveCookieSessionConfig: vi.fn(),
  getRuntimeSingleUserApiKeyOverride: vi.fn(),
  isHostedTldwDeployment: vi.fn(),
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
  },
  isActiveCookieSessionConfig: (...args: unknown[]) =>
    (mocks.isActiveCookieSessionConfig as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: (...args: unknown[]) =>
    (mocks.getRuntimeSingleUserApiKeyOverride as (...args: unknown[]) => unknown)(...args)
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: (...args: unknown[]) =>
    (mocks.isHostedTldwDeployment as (...args: unknown[]) => unknown)(...args)
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

const isIngestionSourceCapabilitiesRequest = (request: unknown): boolean =>
  typeof request === "object" &&
  request !== null &&
  (request as { path?: unknown }).path ===
    "/api/v1/ingestion-sources/capabilities"

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
    mocks.isActiveCookieSessionConfig.mockReset()
    mocks.getRuntimeSingleUserApiKeyOverride.mockReset()
    mocks.isHostedTldwDeployment.mockReset()
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
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue(null)
    mocks.isActiveCookieSessionConfig.mockReturnValue(false)
    mocks.isHostedTldwDeployment.mockReturnValue(false)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it.each(["messages", "completion"] as const)(
    "detects Boolean strict admission only in the actual %s POST request contract",
    async (operation) => {
      const capabilitiesModule = await importCapabilitiesModule()
      const detect = capabilitiesModule.detectNativeHistoryBranchAdmission
      const path =
        operation === "messages"
          ? "/api/v1/chats/{chat_id}/messages"
          : "/api/v1/chat/completions"
      const otherPath =
        operation === "messages"
          ? "/api/v1/chat/completions"
          : "/api/v1/chats/{chat_id}/messages"
      const request = {
        requestBody: {
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/Admission" }
            }
          }
        }
      }
      const spec = {
        paths: { [path]: { post: request } },
        components: {
          schemas: {
            Admission: {
              title: "Actual deployed request annotations",
              description:
                "The deployed schema includes the singular OpenAPI example annotation.",
              example: { tldw_history_branch: true },
              type: "object",
              properties: {
                tldw_history_branch: { type: "boolean", default: true }
              }
            }
          }
        }
      }
      expect(detect(spec, operation)).toBe(true)
      expect(
        detect({ ...spec, paths: { [otherPath]: { post: request } } }, operation)
      ).toBe(false)
      expect(
        detect({ ...spec, paths: { [path]: { get: request } } }, operation)
      ).toBe(false)
      for (const schema of [
        null,
        {},
        { properties: { tldw_history_branch: { type: "string" } } },
        { $ref: "https://foreign.test/schema" },
        { $ref: "#/components/schemas/Missing" },
        { $ref: "#/components/schemas/Admission" }
      ]) {
        expect(
          detect(
            { ...spec, components: { schemas: { Admission: schema } } },
            operation
          )
        ).toBe(false)
      }
    }
  )

  it.each(["messages", "completion"] as const)(
    "recognizes the actual nullable Boolean %s field without accepting non-Boolean unions",
    async (operation) => {
      const { detectNativeHistoryBranchAdmission: detect } =
        await importCapabilitiesModule()
      const path =
        operation === "messages"
          ? "/api/v1/chats/{chat_id}/messages"
          : "/api/v1/chat/completions"
      const spec = (field: unknown) => ({
        paths: {
          [path]: {
            post: {
              requestBody: {
                content: {
                  "application/json": {
                    schema: { properties: { tldw_history_branch: field } }
                  }
                }
              }
            }
          }
        }
      })
      expect(
        detect(
          spec({ anyOf: [{ type: "boolean" }, { type: "null" }] }),
          operation
        )
      ).toBe(true)
      expect(
        detect(
          spec({ oneOf: [{ type: "boolean" }, { type: "null" }] }),
          operation
        )
      ).toBe(true)
      for (const field of [
        { anyOf: [{ type: "string" }, { type: "null" }] },
        { anyOf: [{ type: "boolean" }, { type: "string" }] },
        { type: "null" },
        { type: "boolean", enum: [true] },
        { type: "boolean", allOf: [{ enum: [true] }] },
        { type: "boolean", not: { const: false } },
        { type: "string", anyOf: [{ type: "boolean" }, { type: "null" }] },
        {
          anyOf: [{ type: "boolean" }, { type: "null" }],
          oneOf: [{ const: true }]
        },
        { oneOf: [{ type: "boolean" }, { type: "boolean" }, { type: "null" }] },
        {
          anyOf: [
            { type: "boolean", allOf: [{ enum: [true] }] },
            { type: "null" }
          ]
        }
      ])
        expect(detect(spec(field), operation)).toBe(false)
    }
  )

  it("bounds nullable Boolean component refs without accepting cycles or foreign field refs", async () => {
    const { detectNativeHistoryBranchAdmission: detect } =
      await importCapabilitiesModule()
    const spec = {
      paths: {
        "/api/v1/chat/completions": {
          post: {
            requestBody: {
              content: {
                "application/json": {
                  schema: {
                    properties: {
                      tldw_history_branch: { $ref: "#/components/schemas/Flag" }
                    }
                  }
                }
              }
            }
          }
        }
      },
      components: {
        schemas: {
          Flag: {
            anyOf: [{ $ref: "#/components/schemas/Boolean" }, { type: "null" }]
          },
          Boolean: { type: "boolean" }
        }
      }
    }
    expect(detect(spec, "completion")).toBe(true)
    const constrainedRef = {
      ...spec,
      components: {
        schemas: {
          ...spec.components.schemas,
          Flag: {
            $ref: "#/components/schemas/Boolean",
            allOf: [{ enum: [true] }]
          }
        }
      }
    }
    expect(detect(constrainedRef, "completion")).toBe(false)
    const cyclic = {
      ...spec,
      components: { schemas: { Flag: { $ref: "#/components/schemas/Flag" } } }
    }
    expect(detect(cyclic, "completion")).toBe(false)
    const foreign = {
      ...spec,
      components: {
        schemas: { Flag: { $ref: "https://foreign.test/flag", type: "boolean" } }
      }
    }
    expect(detect(foreign, "completion")).toBe(false)
  })

  it.each(["messages", "completion"] as const)(
    "does not grant strict admission from unqualified request constraints or ref siblings for %s",
    async (operation) => {
      const { detectNativeHistoryBranchAdmission: detect } =
        await importCapabilitiesModule()
      const path =
        operation === "messages"
          ? "/api/v1/chats/{chat_id}/messages"
          : "/api/v1/chat/completions"
      const request = { properties: { tldw_history_branch: { type: "boolean" } } }
      for (const schema of [
        {
          allOf: [
            request,
            { properties: { tldw_history_branch: { const: true } } }
          ]
        },
        { ...request, type: "string" },
        {
          ...request,
          not: {
            properties: { tldw_history_branch: { const: false } },
            required: ["tldw_history_branch"]
          }
        },
        { ...request, patternProperties: { "^tldw": { const: true } } },
        {
          $ref: "#/components/schemas/Request",
          allOf: [{ properties: { tldw_history_branch: { const: true } } }]
        },
        {
          $ref: "#/components/schemas/Request",
          not: {
            properties: { tldw_history_branch: { const: false } },
            required: ["tldw_history_branch"]
          }
        }
      ]) {
        const spec = {
          paths: {
            [path]: {
              post: {
                requestBody: { content: { "application/json": { schema } } }
              }
            }
          },
          components: { schemas: { Request: request } }
        }
        expect(detect(spec, operation)).toBe(false)
      }
    }
  )

  it("pins generic capability reads to the same connection as their cache key", async () => {
    const original = {
      serverUrl: "https://original.test",
      authMode: "single-user" as const,
      apiKey: "original-fixture"
    }
    mocks.getConfig.mockResolvedValue(original)
    mocks.getOpenAPISpec.mockImplementation(async () => {
      mocks.getConfig.mockResolvedValue({
        ...original,
        serverUrl: "https://replacement.test"
      })
      return { paths: { "/api/v1/chat/completions": {} } }
    })
    mocks.bgRequest.mockResolvedValue({})
    const { getServerCapabilities } = await importCapabilitiesModule()
    await getServerCapabilities()
    expect(mocks.getOpenAPISpec).toHaveBeenCalledWith({
      configSnapshot: original
    })
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/config/docs-info",
        configSnapshot: original
      })
    )
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

  it("defers protected source capability probing without configured credentials", async () => {
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "ingestion-sources-unconfigured" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/{source_id}": {},
        "/api/v1/ingestion-sources/capabilities": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBeNull()
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) =>
          isIngestionSourceCapabilitiesRequest(request)
      )
    ).toBe(false)
  })

  it.each([
    [
      "a manual single-user key",
      { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "configured-single-user-key" },
      { runtimeApiKey: null, activeCookieSession: false, hosted: false }
    ],
    [
      "a multi-user token",
      { serverUrl: "http://127.0.0.1:8000", authMode: "multi-user", accessToken: "configured-access-token" },
      { runtimeApiKey: null, activeCookieSession: false, hosted: false }
    ],
    [
      "an active cookie session",
      { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", authSource: "cookie-session" },
      { runtimeApiKey: null, activeCookieSession: true, hosted: false }
    ],
    [
      "a runtime single-user key",
      { serverUrl: "http://127.0.0.1:8000", authMode: "multi-user" },
      { runtimeApiKey: "runtime-single-user-key", activeCookieSession: false, hosted: false }
    ],
    [
      "a hosted session transport",
      null,
      { runtimeApiKey: null, activeCookieSession: false, hosted: true }
    ]
  ])("probes source capabilities with %s", async (_label, config, transport) => {
    mocks.getConfig.mockResolvedValue(config)
    mocks.getRuntimeSingleUserApiKeyOverride.mockReturnValue(transport.runtimeApiKey)
    mocks.isActiveCookieSessionConfig.mockReturnValue(transport.activeCookieSession)
    mocks.isHostedTldwDeployment.mockReturnValue(transport.hosted)
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "authenticated-source-capability" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/capabilities": {}
      }
    })
    mocks.bgRequest.mockImplementation(async (request: { path?: unknown }) =>
      isIngestionSourceCapabilitiesRequest(request)
        ? { can_create_local_directory: true }
        : {}
    )

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBe(true)
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) =>
          isIngestionSourceCapabilitiesRequest(request)
      )
    ).toBe(true)
  })

  it.each([
    ["a placeholder manual key", { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", apiKey: "REPLACE_ME" }],
    ["an inactive cookie session", { serverUrl: "http://127.0.0.1:8000", authMode: "single-user", authSource: "cookie-session" }]
  ])("defers source capability probing with %s", async (label, config) => {
    mocks.getConfig.mockResolvedValue(config)
    mocks.isActiveCookieSessionConfig.mockReturnValue(false)
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "unauthenticated-source-capability" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/capabilities": {}
      }
    })
    mocks.bgRequest.mockResolvedValue({})

    const { getServerCapabilities } = await importCapabilitiesModule()
    const capabilities = await getServerCapabilities()

    expect(capabilities.hasIngestionSources).toBe(true)
    expect(capabilities.canCreateLocalDirectoryIngestionSource).toBeNull()
    expect(
      mocks.bgRequest.mock.calls.some(
        ([request]) =>
          isIngestionSourceCapabilitiesRequest(request)
      )
    ).toBe(false)
  })

  it("recovers unknown source entitlements after credentials are configured", async () => {
    mocks.getConfig
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user"
      })
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "configured-after-discovery-key"
      })
    mocks.getOpenAPISpec.mockResolvedValue({
      info: { version: "source-capability-recovery" },
      paths: {
        "/api/v1/ingestion-sources": {},
        "/api/v1/ingestion-sources/capabilities": {}
      }
    })
    mocks.bgRequest.mockImplementation(async (request: { path?: unknown }) =>
      isIngestionSourceCapabilitiesRequest(request)
        ? { can_create_local_directory: true }
        : {}
    )

    const { getServerCapabilities } = await importCapabilitiesModule()
    const beforeConfiguration = await getServerCapabilities()
    const afterConfiguration = await getServerCapabilities()

    expect(beforeConfiguration.canCreateLocalDirectoryIngestionSource).toBeNull()
    expect(afterConfiguration.canCreateLocalDirectoryIngestionSource).toBe(true)
    expect(
      mocks.bgRequest.mock.calls.filter(
        ([request]) =>
          isIngestionSourceCapabilitiesRequest(request)
      )
    ).toHaveLength(1)
  })

  it("merges authenticated local-directory source creation capability", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "configured-single-user-key"
    })
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
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "configured-single-user-key"
    })
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
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "configured-single-user-key"
    })
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
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "configured-single-user-key"
    })
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
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user",
      apiKey: "configured-single-user-key"
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


describe("protected Edit primary request capability", () => {
  const path = (operation: "messages" | "completion") =>
    operation === "messages"
      ? "/api/v1/chats/{chat_id}/messages"
      : "/api/v1/chat/completions"
  const spec = (
    operation: "messages" | "completion",
    schema: unknown,
    components = {}
  ) => ({
    paths: {
      [path(operation)]: {
        post: { requestBody: { content: { "application/json": { schema } } } }
      }
    },
    components: { schemas: components }
  })
  it.each(["messages", "completion"] as const)(
    "accepts only the supported primary %s schema",
    async (operation) => {
      const { protectedEditRequestSchema } = await import(
        "./history-edit-fixture"
      )
      const { detectNativeHistoryEditAdmission: detect } =
        await importCapabilitiesModule()
      const schema = protectedEditRequestSchema()
      expect(detect(spec(operation, schema), operation)).toBe(true)
      expect(
        detect(
          spec(
            operation,
            { $ref: "#/components/schemas/Request" },
            { Request: schema }
          ),
          operation
        )
      ).toBe(true)
      const target = schema.properties.tldw_history_edit_v1.anyOf[0]
      const refSchema = {
        ...schema,
        properties: {
          ...schema.properties,
          tldw_history_edit_v1: {
            anyOf: [{ $ref: "#/components/schemas/Target" }, { type: "null" }]
          }
        }
      }
      expect(
        detect(spec(operation, refSchema, { Target: target }), operation)
      ).toBe(true)
    }
  )
  it.each([
    "missing field",
    "false only",
    "string branch",
    "nullable only",
    "open target",
    "optional member",
    "extra member",
    "wrong version",
    "coerced version",
    "empty revision",
    "ambiguous union",
    "constraint sibling",
    "external ref",
    "cyclic ref",
    "unrelated component",
    "wrong operation",
    "missing primary"
  ])("does not grant Edit authority from %s", async (change) => {
    const { protectedEditRequestSchema } = await import("./history-edit-fixture")
    const { detectNativeHistoryEditAdmission: detect } =
      await importCapabilitiesModule()
    const schema = protectedEditRequestSchema()
    const target = schema.properties.tldw_history_edit_v1.anyOf[0]
    if (!target.properties || !target.required)
      throw new Error("Missing target fixture")
    let value: unknown
    if (change === "missing field")
      Reflect.deleteProperty(schema.properties, "tldw_history_edit_v1")
    if (change === "false only")
      Object.assign(schema.properties, {
        tldw_history_branch: { type: "boolean", const: false }
      })
    if (change === "string branch")
      Object.assign(schema.properties, {
        tldw_history_branch: { type: "string" }
      })
    if (change === "nullable only")
      Object.assign(schema.properties, { tldw_history_edit_v1: { type: "null" } })
    if (change === "open target") target.additionalProperties = true
    if (change === "optional member") target.required.pop()
    if (change === "extra member")
      Object.assign(target.properties, { optional: { type: "string" } })
    if (change === "wrong version") target.properties.version.const = 2
    if (change === "coerced version") target.properties.version.type = "number"
    if (change === "empty revision")
      target.properties.message_revision.minLength = 0
    if (change === "ambiguous union")
      schema.properties.tldw_history_edit_v1.anyOf.push(structuredClone(target))
    if (change === "constraint sibling")
      Object.assign(schema.properties.tldw_history_edit_v1, {
        allOf: [{ type: "null" }]
      })
    if (change === "external ref")
      Object.assign(schema.properties, {
        tldw_history_edit_v1: {
          $ref: "https://foreign.test/schema"
        }
      })
    if (change === "cyclic ref")
      value = spec(
        "messages",
        { $ref: "#/components/schemas/Loop" },
        { Loop: { $ref: "#/components/schemas/Loop" } }
      )
    if (change === "unrelated component")
      value = spec("messages", { type: "object" }, { Unused: schema })
    if (change === "wrong operation") value = spec("completion", schema)
    if (change === "missing primary")
      value = {
        paths: { [path("messages")]: { post: { responses: { 200: schema } } } }
      }
    expect(detect(value ?? spec("messages", schema), "messages")).toBe(false)
  })
})
