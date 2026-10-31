import { CLEAR_TASK_RECIPE } from "@/components/Common/PromptAssist/recipes/built-in-recipes"
import {
  canApplyRecipe,
  createRecipeWorkingCopy,
  recipeEditorReducer
} from "@/components/Common/PromptAssist/recipes/recipe-editor-state"
import { getRecipePersistenceState } from "@/components/Option/Prompt/prompt-recipe-library"
import { renderSingleTextRecipe } from "@/components/Option/Prompt/structured-prompt-utils"
import { db } from "@/db/dexie/schema"
import { useRecipePersistenceOwner } from "@/hooks/useRecipePersistenceOwner"
import { apiSend } from "@/services/api-send"
import {
  classifyRecipeDispatch,
  pullFromStudio,
  pushToStudio
} from "@/services/prompt-sync"
import type { PromptCapabilities } from "@/services/prompts-api"
import { RecipePersistenceRegistry } from "@/services/recipe-persistence-registry"
import {
  forgetRecipePersistenceUnknown,
  readRecipePersistenceUncertainty,
  resolveRecipePersistenceOwnerView
} from "@/services/recipe-persistence-uncertainty"
import { resolveRecipeRequestSnapshot } from "@/services/tldw/recipe-request-snapshot"
import { tldwRequest } from "@/services/tldw/request-core"
import {
  clearRuntimeAuthOverride,
  setRuntimeSingleUserApiKeyOverride
} from "@/services/tldw/runtime-auth-override"
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const extension = vi.hoisted(() => {
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (value: unknown) => value
  })
  return {
    runtimeId: undefined as string | undefined,
    sendMessage: vi.fn(),
    listeners: new Set<
      (
        message: unknown,
        sender: unknown,
        reply: (value: unknown) => void
      ) => unknown
    >(),
    values: new Map<string, unknown>(),
    rows: new Map<string, Record<string, unknown>>()
  }
})

vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: async (key: string) => extension.values.get(key),
    set: async (key: string, value: unknown) => {
      extension.values.set(key, value)
    },
    remove: async (key: string) => {
      extension.values.delete(key)
    }
  })
}))

vi.mock("@/db/dexie/helpers", () => ({ generateID: () => "generated-id" }))
vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      get: async (id: string) => extension.rows.get(id),
      update: async (
        id: string,
        fields: object | ((row: Record<string, unknown>) => void | boolean)
      ) => {
        const current = extension.rows.get(id)
        if (!current) return 0
        const row = structuredClone(current)
        if (typeof fields === "function") {
          if (fields(row) === false) return 1
        } else {
          Object.assign(row, fields)
        }
        extension.rows.set(id, row)
        return 1
      },
      add: async (row: { id: string }) => {
        extension.rows.set(row.id, structuredClone(row))
      },
      where: (field: string) => ({
        equals: (value: unknown) => ({
          first: async () =>
            [...extension.rows.values()].find((row) => row[field] === value)
        })
      })
    }
  }
}))
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => ({
    defaultProjectId: 42,
    autoSyncWorkspacePrompts: true
  }),
  setPromptStudioDefaults: async (value: unknown) => value
}))
vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "warm",
  initBackground: async () => {}
}))
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {}
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return extension.runtimeId
      },
      getURL: (path: string) => `chrome-extension://extension-id${path}`,
      sendMessage: (...args: unknown[]) => extension.sendMessage(...args),
      onConnect: { addListener: vi.fn() },
      onStartup: { addListener: vi.fn() },
      onMessage: {
        addListener: (
          listener: Parameters<typeof extension.listeners.add>[0]
        ) => extension.listeners.add(listener)
      }
    },
    storage: {
      local: { get: async () => ({}), set: async () => {} },
      session: { get: async () => ({}), set: async () => {} },
      onChanged: { addListener: vi.fn() }
    },
    alarms: {
      clear: async () => true,
      create: async () => {},
      onAlarm: { addListener: vi.fn() }
    },
    tabs: {
      query: async () => [],
      create: vi.fn(),
      sendMessage: async () => {}
    },
    action: { onClicked: { addListener: vi.fn() } },
    contextMenus: {
      create: vi.fn(),
      removeAll: vi.fn(),
      onClicked: { addListener: vi.fn() }
    },
    i18n: { getMessage: (key: string) => key }
  }
}))

type OwnerConfig = {
  serverUrl: string
  authMode: "single-user" | "multi-user"
  authSource?: string
  apiKey?: string
  accessToken?: string
  refreshToken?: string
  orgId?: string
}

type ContractCase =
  | "manual API-key owner"
  | "runtime API-key owner"
  | "bearer principal owner"
  | "normalized deployment owner"

const localId = (label: string) =>
  `recipe-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`

const response = (status = 200, body: unknown = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" }
  })

const serverPrompt = (id: number) => ({
  id,
  project_id: 42,
  name: `Recipe ${id}`,
  version_number: 1,
  updated_at: "2026-09-11T00:00:00Z",
  prompt_format: "structured",
  prompt_schema_version: 2,
  prompt_definition: CLEAR_TASK_RECIPE.definition,
  system_prompt: "Compiled recipe",
  user_prompt: ""
})

const successfulPromptResponse = (id: number) =>
  response(200, { success: true, data: serverPrompt(id) })

const seedRecipe = (id: string, serverId?: number) => {
  extension.rows.set(id, {
    id,
    title: `Local ${id}`,
    name: `Local ${id}`,
    content: "Local recipe",
    promptFormat: "structured",
    promptSchemaVersion: 2,
    structuredPromptDefinition: structuredClone(CLEAR_TASK_RECIPE.definition),
    studioProjectId: 42,
    syncStatus: serverId ? "conflict" : "local",
    ...(serverId ? { serverId, studioPromptId: serverId } : {})
  })
}

const storedManualConfig = (apiKey = "manual-key") => ({
  ...manualConfig({ apiKey }),
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://recipes.example.test"
})

const sendToBackground = (message: unknown): Promise<unknown> =>
  new Promise((resolve) => {
    const listener = [...extension.listeners][0]
    if (!listener) throw new Error("Missing background listener")
    listener(message, { id: "extension-id" }, resolve)
  })

const startExtensionBackground = async () => {
  extension.runtimeId = "extension-id"
  extension.listeners.clear()
  vi.stubGlobal("window", undefined)
  const background = (await import("@/entries/background")).default
  background.main()
  extension.sendMessage.mockImplementation(sendToBackground)
}

const manualConfig = (overrides: Partial<OwnerConfig> = {}): OwnerConfig => ({
  serverUrl: "https://recipes.example.test:443/deploy///",
  authMode: "single-user",
  authSource: "manual",
  apiKey: "manual-key",
  orgId: "org-a",
  ...overrides
})

const bearerConfig = (overrides: Partial<OwnerConfig> = {}): OwnerConfig => ({
  serverUrl: "https://recipes.example.test/deploy",
  authMode: "multi-user",
  authSource: "manual",
  accessToken: "token-a",
  refreshToken: "refresh-a",
  orgId: "org-a",
  ...overrides
})

const resolveView = (
  config: OwnerConfig,
  options: {
    principal?: string | null
    runtimeKey?: string | null
    cookieSession?: boolean
    cookieRevision?: string
  } = {}
) =>
  resolveRecipeRequestSnapshot({
    config,
    path: "/api/v1/prompts/",
    method: "POST",
    pageOrigin: window.location.origin,
    runtimeApiKey: options.runtimeKey,
    authenticatedPrincipalId: options.principal,
    cookieSessionTransport: options.cookieSession,
    cookieSessionRevision: options.cookieRevision
  }).view

const expectCompiledLocalApply = (
  label: string,
  target: "system" | "user_message"
) => {
  const initial = createRecipeWorkingCopy(CLEAR_TASK_RECIPE, target)
  const ready = recipeEditorReducer(initial, {
    type: "runtime_value_changed",
    variableName: "task",
    value: `Local work from ${label}`
  })
  expect(canApplyRecipe(ready)).toBe(true)
  const compiled = renderSingleTextRecipe(
    ready.definition,
    ready.runtimeValues
  ).rendered_text
  expect(compiled).toContain(`Local work from ${label}`)
  expect(ready.definition.assembly_config.target_role).toBe(
    target === "system" ? "system" : "user"
  )
}

afterEach(() => {
  clearRuntimeAuthOverride()
  extension.runtimeId = undefined
  extension.sendMessage.mockReset()
  extension.listeners.clear()
  extension.values.clear()
  extension.rows.clear()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
})

describe("owner snapshot to immutable dispatch contract", () => {
  const cases: ReadonlyArray<{
    label: ContractCase
    authority: "direct" | "extension"
    config: OwnerConfig
    principal?: string
    runtimeKey?: string
  }> = [
    {
      label: "manual API-key owner",
      authority: "direct",
      config: manualConfig()
    },
    {
      label: "runtime API-key owner",
      authority: "direct",
      config: manualConfig({ apiKey: "configured-key" }),
      runtimeKey: "runtime-key"
    },
    {
      label: "bearer principal owner",
      authority: "extension",
      config: bearerConfig(),
      principal: "alice"
    },
    {
      label: "normalized deployment owner",
      authority: "extension",
      config: manualConfig({
        serverUrl: "https://recipes.example.test/deploy/",
        apiKey: "extension-key"
      })
    }
  ]

  it.each(cases)(
    "$label sends one opaque owner through the immutable request and locks duplicate writes",
    async ({ label, config, principal, runtimeKey }) => {
      if (runtimeKey) setRuntimeSingleUserApiKeyOverride(runtimeKey)
      const owner = resolveView(config, { principal, runtimeKey })
      expect(owner).toEqual({
        ownerId: expect.stringMatching(/^recipe-owner:sha256:[0-9a-f]{64}$/),
        authorizationRevision: expect.stringMatching(
          /^recipe-authorization:sha256:[0-9a-f]{64}$/
        )
      })
      expect(JSON.stringify(owner)).not.toMatch(
        /manual-key|configured-key|runtime-key|token-a|extension-key|recipes\.example/
      )

      const id = localId(label)
      const registry = new RecipePersistenceRegistry()
      const fetchFn = vi.fn(async () => response())
      const runtime = {
        getConfig: async () => config,
        getAuthenticatedPrincipal: async () => principal ?? null,
        dispatchAuthority: {
          markDispatched: (markedId: string, actualOwnerId: string) =>
            registry.reserve(markedId, actualOwnerId)
        },
        fetchFn
      }
      const payload = {
        path: "/api/v1/prompts/" as const,
        method: "POST",
        body: { name: "Local recipe" },
        recipePersistence: {
          mode: "require" as const,
          expectedOwnerId: owner!.ownerId,
          localId: id
        }
      }

      const first = await tldwRequest(payload, runtime)
      expect(first.recipePersistence).toEqual({
        state: "dispatched",
        actualOwnerId: owner!.ownerId
      })
      expect(fetchFn).toHaveBeenCalledOnce()
      expect(registry.read(id, owner!.ownerId)).toBe("scoped")
      expect(classifyRecipeDispatch(first.recipePersistence!)).toBe(
        "scoped_uncertain"
      )

      const duplicate = await tldwRequest(payload, runtime)
      expect(duplicate.recipePersistence).toEqual({
        state: "not_dispatched",
        actualOwnerId: null
      })
      expect(fetchFn).toHaveBeenCalledOnce()

      registry.clearScoped(id, owner!.ownerId)
      expect(registry.read(id, owner!.ownerId)).toBe("clear")
    }
  )

  it("uses quickstart cookie ownership only with an authoritative principal", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    const config = manualConfig({
      serverUrl: window.location.origin,
      authSource: "cookie-session",
      apiKey: undefined,
      orgId: undefined
    })
    const owner = resolveView(config, {
      principal: "cookie-user",
      cookieSession: true,
      cookieRevision: "session-1"
    })
    const noPrincipal = resolveView(config, {
      principal: null,
      cookieSession: true,
      cookieRevision: "session-1"
    })

    expect(owner?.ownerId).toMatch(/^recipe-owner:sha256:/)
    expect(noPrincipal).toBeNull()

    const fetchFn = vi.fn(async () => response())
    const result = await tldwRequest(
      {
        path: "/api/v1/prompts/",
        method: "POST",
        recipePersistence: {
          mode: "require",
          expectedOwnerId: owner!.ownerId,
          localId: "cookie-recipe"
        }
      },
      {
        getConfig: async () => config,
        getAuthenticatedPrincipal: async () => null,
        dispatchAuthority: {
          markDispatched: vi.fn()
        },
        fetchFn
      }
    )
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
    expect(fetchFn).not.toHaveBeenCalled()
    expectCompiledLocalApply("cookie owner fallback", "system")
  })

  it.each(cases)(
    "$label masks a closed view and re-resolves only a sanitized owner on reopen",
    async ({ authority, config, principal, runtimeKey }) => {
      const stored = {
        ...config,
        credentialSource: "manual",
        apiKeyPersistence: "device",
        apiKeyServerOrigin: "https://recipes.example.test"
      }
      extension.values.set("tldwConfig", stored)
      if (runtimeKey) setRuntimeSingleUserApiKeyOverride(runtimeKey)
      if (principal) {
        vi.stubGlobal("fetch", async () => response(200, { id: principal }))
      }
      if (authority === "extension") {
        extension.runtimeId = "extension-id"
        const view = resolveView(config, { principal, runtimeKey })
        extension.sendMessage.mockResolvedValue(view)
      }

      const hook = renderHook(
        ({ enabled }) => useRecipePersistenceOwner(enabled),
        { initialProps: { enabled: false } }
      )
      expect(hook.result.current).toMatchObject({
        owner: null,
        loading: false
      })
      hook.rerender({ enabled: true })
      await waitFor(() =>
        expect(hook.result.current.owner?.ownerId).toMatch(
          /^recipe-owner:sha256:/
        )
      )
      expect(Object.keys(hook.result.current.owner!).sort()).toEqual([
        "authorizationRevision",
        "ownerId"
      ])
      hook.rerender({ enabled: false })
      expect(hook.result.current.owner).toBeNull()

      await act(async () => {
        hook.rerender({ enabled: true })
      })
      await waitFor(() =>
        expect(hook.result.current.owner?.ownerId).toMatch(
          /^recipe-owner:sha256:/
        )
      )
      if (authority === "extension") {
        expect(extension.sendMessage).toHaveBeenCalledTimes(2)
        expect(extension.sendMessage).toHaveBeenNthCalledWith(1, {
          type: "tldw:recipe-owner:resolve"
        })
      }
      hook.unmount()
    }
  )
})

describe("identity drift before dispatch and refresh", () => {
  it.each([
    [
      "backend",
      bearerConfig({ serverUrl: "https://other.example.test" }),
      "alice"
    ],
    ["principal", bearerConfig(), "bob"],
    ["organization", bearerConfig({ orgId: "org-b" }), "alice"]
  ] as const)(
    "rejects a changed %s before the first fetch",
    async (_change, changedConfig, principal) => {
      const initial = bearerConfig()
      const expected = resolveView(initial, { principal: "alice" })!
      const fetchFn = vi.fn(async () => response())
      const registry = new RecipePersistenceRegistry()
      const result = await tldwRequest(
        {
          path: "/api/v1/prompts/",
          method: "POST",
          recipePersistence: {
            mode: "require",
            expectedOwnerId: expected.ownerId,
            localId: `changed-${_change}`
          }
        },
        {
          getConfig: async () => changedConfig,
          getAuthenticatedPrincipal: async () => principal,
          dispatchAuthority: {
            markDispatched: (id, ownerId) => registry.reserve(id, ownerId)
          },
          fetchFn
        }
      )

      expect(result.recipePersistence).toEqual({
        state: "not_dispatched",
        actualOwnerId: null
      })
      expect(fetchFn).not.toHaveBeenCalled()
      expectCompiledLocalApply("identity drift", "user_message")
    }
  )

  it("rejects an authentication-source change before the first fetch", async () => {
    const config = manualConfig()
    const expected = resolveView(config)!
    setRuntimeSingleUserApiKeyOverride("runtime-key")
    const fetchFn = vi.fn(async () => response())
    const result = await tldwRequest(
      {
        path: "/api/v1/prompts/",
        method: "POST",
        recipePersistence: {
          mode: "require",
          expectedOwnerId: expected.ownerId,
          localId: "changed-source"
        }
      },
      {
        getConfig: async () => config,
        dispatchAuthority: { markDispatched: vi.fn() },
        fetchFn
      }
    )
    expect(result.recipePersistence?.state).toBe("not_dispatched")
    expect(fetchFn).not.toHaveBeenCalled()
  })

  it("allows same-subject bearer refresh with one reservation", async () => {
    const initial = bearerConfig()
    const refreshed = bearerConfig({ accessToken: "token-b" })
    const initialView = resolveView(initial, { principal: "alice" })!
    const refreshedView = resolveView(refreshed, { principal: "alice" })!
    expect(refreshedView.ownerId).toBe(initialView.ownerId)
    expect(refreshedView.authorizationRevision).not.toBe(
      initialView.authorizationRevision
    )

    const registry = new RecipePersistenceRegistry()
    const markDispatched = vi.fn((id: string, ownerId: string) =>
      registry.reserve(id, ownerId)
    )
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(response(401))
      .mockResolvedValueOnce(response(200))
    const result = await tldwRequest(
      {
        path: "/api/v1/prompts/",
        method: "POST",
        recipePersistence: {
          mode: "require",
          expectedOwnerId: initialView.ownerId,
          localId: "same-sub-refresh"
        }
      },
      {
        getConfig: vi
          .fn()
          .mockResolvedValueOnce(initial)
          .mockResolvedValue(refreshed),
        getAuthenticatedPrincipal: async () => "alice",
        dispatchAuthority: { markDispatched },
        fetchFn,
        refreshAuth: async () => {}
      }
    )
    expect(result.ok).toBe(true)
    expect(markDispatched).toHaveBeenCalledOnce()
    expect(fetchFn).toHaveBeenCalledTimes(2)
    expect(registry.read("same-sub-refresh", initialView.ownerId)).toBe(
      "scoped"
    )
  })

  it.each([
    [
      "backend",
      bearerConfig({ serverUrl: "https://other.example.test" }),
      "alice"
    ],
    ["principal", bearerConfig({ accessToken: "token-b" }), "bob"],
    [
      "organization",
      bearerConfig({ accessToken: "token-b", orgId: "org-b" }),
      "alice"
    ],
    ["authentication source", manualConfig({ apiKey: "replacement-key" }), null]
  ] as const)(
    "stops after the first dispatched request when refresh changes %s",
    async (change, changed, refreshedPrincipal) => {
      const initial = bearerConfig()
      const initialView = resolveView(initial, { principal: "alice" })!
      const registry = new RecipePersistenceRegistry()
      const changedFetch = vi.fn().mockResolvedValue(response(401))
      let principalCalls = 0
      const changedResult = await tldwRequest(
        {
          path: "/api/v1/prompts/",
          method: "POST",
          recipePersistence: {
            mode: "require",
            expectedOwnerId: initialView.ownerId,
            localId: `changed-refresh-${change}`
          }
        },
        {
          getConfig: vi
            .fn()
            .mockResolvedValueOnce(initial)
            .mockResolvedValue(changed),
          getAuthenticatedPrincipal: async () =>
            principalCalls++ === 0 ? "alice" : refreshedPrincipal,
          dispatchAuthority: {
            markDispatched: (id, ownerId) => registry.reserve(id, ownerId)
          },
          fetchFn: changedFetch,
          refreshAuth: async () => {}
        }
      )
      expect(changedResult).toMatchObject({ ok: false, status: 412 })
      expect(changedResult.recipePersistence?.state).toBe("dispatched")
      expect(changedFetch).toHaveBeenCalledOnce()
      expectCompiledLocalApply("refresh drift", "user_message")
    }
  )
})

describe("uncertain outcomes, reopen, reconciliation, and recovery", () => {
  it.each([
    ["system-target direct owner", "system"],
    ["user-target direct owner", "user_message"]
  ] as const)(
    "%s keeps a scoped ambiguous mutation locked after close and reopen",
    async (label, target) => {
      const config = manualConfig()
      const owner = resolveView(config)!
      const registry = new RecipePersistenceRegistry()
      const fetchFn = vi.fn().mockRejectedValue(new Error("connection lost"))
      const payload = {
        path: "/api/v1/prompts/" as const,
        method: "POST",
        recipePersistence: {
          mode: "require" as const,
          expectedOwnerId: owner.ownerId,
          localId: localId(`${label}-ambiguous`)
        }
      }
      const runtime = {
        getConfig: async () => config,
        dispatchAuthority: {
          markDispatched: (id: string, ownerId: string) =>
            registry.reserve(id, ownerId)
        },
        fetchFn
      }
      const result = await tldwRequest(payload, runtime)
      expect(result.recipePersistence).toEqual({
        state: "dispatched",
        actualOwnerId: owner.ownerId
      })
      expect(
        registry.read(payload.recipePersistence.localId, owner.ownerId)
      ).toBe("scoped")

      // A later caller reads the application registry; caller disposal never
      // owns or clears this process-level state.
      const reopened = registry
      expect(
        reopened.read(payload.recipePersistence.localId, owner.ownerId)
      ).toBe("scoped")
      const duplicate = await tldwRequest(payload, runtime)
      expect(duplicate.recipePersistence?.state).toBe("not_dispatched")
      expect(fetchFn).toHaveBeenCalledOnce()
      expectCompiledLocalApply(label, target)
    }
  )

  it("reports a lost extension response as unknown without direct replay", async () => {
    extension.runtimeId = "extension-id"
    extension.sendMessage.mockRejectedValue(new Error("channel closed"))
    const fetchSpy = vi.spyOn(globalThis, "fetch")
    const result = await apiSend({
      path: "/api/v1/prompts/",
      method: "POST",
      recipePersistence: {
        mode: "require",
        expectedOwnerId: `recipe-owner:sha256:${"a".repeat(64)}`,
        localId: "unknown-extension"
      }
    })
    expect(result.recipePersistence).toEqual({
      state: "unknown",
      actualOwnerId: null
    })
    expect(classifyRecipeDispatch(result.recipePersistence!)).toBe(
      "unknown_owner"
    )
    expect(extension.sendMessage).toHaveBeenCalledOnce()
    expect(fetchSpy).not.toHaveBeenCalled()

    const registry = new RecipePersistenceRegistry()
    registry.markUnknown("unknown-extension")
    for (const owner of ["owner-a", "owner-b", null]) {
      expect(registry.read("unknown-extension", owner)).toBe("unknown_owner")
    }
    expect(() => registry.reserve("unknown-extension", "owner-a")).toThrow()
    expectCompiledLocalApply("lost extension response", "user_message")
  })

  it("clears only matching scoped reconciliation and Forget removes only exact unknown quarantine", () => {
    const registry = new RecipePersistenceRegistry()
    registry.markScoped("shared", "owner-a")
    registry.markScoped("shared", "owner-b")
    registry.markUnknown("shared")
    registry.markUnknown("other")

    registry.clearScoped("shared", "owner-b")
    expect(registry.read("shared", "owner-a")).toBe("unknown_owner")
    registry.forgetUnknown("shared")
    expect(registry.read("shared", "owner-a")).toBe("scoped")
    expect(registry.read("shared", "owner-b")).toBe("clear")
    expect(registry.read("other", "owner-a")).toBe("unknown_owner")

    registry.clearScoped("shared", "owner-b")
    expect(registry.read("shared", "owner-a")).toBe("scoped")
    registry.clearScoped("shared", "owner-a")
    expect(registry.read("shared", "owner-a")).toBe("clear")
  })

  it("keeps local Apply available for the deferred unavailable-capability representation", () => {
    const unavailable: PromptCapabilities = {
      availability: "unavailable",
      prompt_improvement_v1: { supported: false, limits: null },
      single_text_recipe_v2: { supported: false },
      prompt_persistence: {
        create_authorized: null,
        update_authorized: null
      }
    }
    expect(getRecipePersistenceState(true, unavailable)).toMatchObject({
      available: false,
      reason: expect.stringContaining("edit, preview, and apply")
    })
    expectCompiledLocalApply("unavailable system target", "system")
    expectCompiledLocalApply("unavailable user target", "user_message")
  })
})

describe("real sync stack through direct and extension authorities", () => {
  it.each([
    ["direct", "transport"],
    ["direct", "local commit"],
    ["direct", "local error"],
    ["extension", "transport"],
    ["extension", "local commit"],
    ["extension", "local error"]
  ] as const)(
    "%s same-owner Pull cannot release an existing PUT during %s settlement",
    async (surface, phase) => {
      const id = localId(`active-put-${surface}-${phase}`)
      extension.values.set("tldwConfig", storedManualConfig(`${id}-key`))
      if (surface === "extension") await startExtensionBackground()
      seedRecipe(id, 101)
      const owner = (await resolveRecipePersistenceOwnerView())!
      let release!: () => void
      let signalStarted!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      const started = new Promise<void>((resolve) => {
        signalStarted = resolve
      })
      const originalUpdate = db.prompts.update.bind(db.prompts)
      let localCommitHeld = false
      const update = vi
        .spyOn(db.prompts, "update")
        .mockImplementation(async (...args) => {
          const fields = args[1] as Record<string, unknown>
          if (
            !localCommitHeld &&
            ((phase === "local commit" &&
              fields.name === "Updated remote recipe") ||
              (phase === "local error" && fields.syncStatus === "error"))
          ) {
            localCommitHeld = true
            signalStarted()
            await held
          }
          return originalUpdate(...args)
        })
      let putCount = 0
      vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
        if (init.method === "PUT") {
          putCount += 1
          if (phase === "local error") throw new Error("connection lost")
          if (phase === "transport" && putCount === 1) {
            signalStarted()
            await held
          }
          return response(200, {
            success: true,
            data: {
              ...serverPrompt(101),
              name: "Updated remote recipe",
              version_number: 2
            }
          })
        }
        return successfulPromptResponse(101)
      })

      const first = pushToStudio(id, 42, { expectedOwnerId: owner.ownerId })
      try {
        await started
        const pull = await pullFromStudio(101, id)
        const second = await pushToStudio(id, 42, {
          expectedOwnerId: owner.ownerId
        })
        expect.soft(putCount).toBe(1)
        expect
          .soft(pull)
          .toMatchObject({ success: false, recipeWriteBlocked: true })
        expect.soft(second).toMatchObject({
          success: false,
          recipeOwnership: { dispatch: { state: "not_dispatched" } }
        })
        expect
          .soft(await readRecipePersistenceUncertainty(id, owner.ownerId))
          .not.toBe("clear")
      } finally {
        release()
        await first
        update.mockRestore()
      }
      if (phase === "local error") {
        expect(await first).toMatchObject({
          success: false,
          syncStatus: "error"
        })
        expect(extension.rows.get(id)?.syncStatus).toBe("error")
        expect(await readRecipePersistenceUncertainty(id, owner.ownerId)).toBe(
          "scoped"
        )
        expect(await pullFromStudio(101, id)).toMatchObject({ success: true })
      } else {
        expect(await first).toMatchObject({
          success: true,
          syncStatus: "synced"
        })
        expect(extension.rows.get(id)).toMatchObject({
          name: "Updated remote recipe",
          syncStatus: "synced"
        })
      }
      expect(await readRecipePersistenceUncertainty(id, owner.ownerId)).toBe(
        "clear"
      )
    }
  )

  it.each(["manual direct owner", "runtime-key direct owner"] as const)(
    "%s carries its resolved owner through sync, fetch, and exact reconciliation",
    async (label) => {
      const id = localId(`${label}-full-stack`)
      const config = storedManualConfig(`${id}-key`)
      extension.values.set("tldwConfig", config)
      seedRecipe(id)
      const mutationFetch = vi.fn(async () => successfulPromptResponse(101))
      vi.stubGlobal("fetch", mutationFetch)

      const owner = await resolveRecipePersistenceOwnerView()
      expect(owner?.ownerId).toMatch(/^recipe-owner:sha256:/)
      const result = await pushToStudio(id, 42, {
        expectedOwnerId: owner!.ownerId
      })

      expect(result).toMatchObject({
        success: true,
        localId: id,
        recipeOwnership: {
          localId: id,
          dispatch: {
            state: "dispatched",
            actualOwnerId: owner!.ownerId
          }
        }
      })
      expect(mutationFetch).toHaveBeenCalledOnce()
      expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
        "clear"
      )
      expect(extension.rows.get(id)).toMatchObject({
        serverId: 101,
        syncStatus: "synced"
      })
    }
  )

  it("shares one background-scoped ambiguous mutation across callers and clears it only after matching-owner pull", async () => {
    extension.values.set("tldwConfig", storedManualConfig("extension-key"))
    await startExtensionBackground()
    const id = "extension-scoped-full-stack"
    seedRecipe(id)
    const owner = await resolveRecipePersistenceOwnerView()
    expect(owner?.ownerId).toMatch(/^recipe-owner:sha256:/)
    let mutationCount = 0
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      if (String(init.method).toUpperCase() === "POST") {
        mutationCount += 1
        throw new Error("connection lost")
      }
      return successfulPromptResponse(101)
    })

    const firstAttempt = await pushToStudio(id, 42, {
      expectedOwnerId: owner!.ownerId
    })
    expect(firstAttempt).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeOwnership: {
        dispatch: { state: "dispatched", actualOwnerId: owner!.ownerId }
      }
    })
    expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
      "scoped"
    )

    // Disposing one caller drops no worker-owned state; another caller sees the
    // same marker and its retry stops before another mutation.
    const retry = await pushToStudio(id, 42, {
      expectedOwnerId: owner!.ownerId
    })
    expect(retry).toMatchObject({
      success: false,
      recipeOwnership: { dispatch: { state: "not_dispatched" } }
    })
    expect(mutationCount).toBe(1)
    expectCompiledLocalApply("shared background caller", "user_message")

    const reconciliation = await pullFromStudio(101, id)
    expect(reconciliation).toMatchObject({
      success: true,
      recipeOwnership: {
        dispatch: { state: "dispatched", actualOwnerId: owner!.ownerId }
      }
    })
    expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
      "clear"
    )
    expect(mutationCount).toBe(1)
  })

  it("quarantines a lost background response across owners without replay and Forget remains local", async () => {
    extension.values.set("tldwConfig", storedManualConfig("extension-key"))
    await startExtensionBackground()
    const id = "extension-unknown-full-stack"
    seedRecipe(id)
    const owner = await resolveRecipePersistenceOwnerView()
    let mutationCount = 0
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      if (String(init.method).toUpperCase() === "POST") mutationCount += 1
      return successfulPromptResponse(202)
    })
    extension.sendMessage.mockImplementation(async (message: unknown) => {
      const result = await sendToBackground(message)
      const request = message as {
        type?: string
        payload?: { method?: string }
      }
      if (
        request.type === "tldw:request" &&
        String(request.payload?.method).toUpperCase() === "POST"
      ) {
        throw new Error("response port closed")
      }
      return result
    })

    const lost = await pushToStudio(id, 42, {
      expectedOwnerId: owner!.ownerId
    })
    expect(lost).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeOwnership: {
        dispatch: { state: "unknown", actualOwnerId: null }
      }
    })
    expect(mutationCount).toBe(1)
    expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
      "unknown_owner"
    )
    expect(
      await readRecipePersistenceUncertainty(
        id,
        `recipe-owner:sha256:${"b".repeat(64)}`
      )
    ).toBe("unknown_owner")

    const retry = await pushToStudio(id, 42, {
      expectedOwnerId: owner!.ownerId
    })
    expect(retry.recipeOwnership?.dispatch.state).toBe("not_dispatched")
    expect(mutationCount).toBe(1)

    // Even an owner-capturing pull cannot clear an independent unknown result.
    expect(await pullFromStudio(202, id)).toMatchObject({
      success: false,
      syncStatus: "error",
      recipeWriteBlocked: true
    })
    expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
      "unknown_owner"
    )
    await forgetRecipePersistenceUnknown(id)
    expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
      "scoped"
    )
    expect(await pullFromStudio(202, id)).toMatchObject({ success: true })
    expect(await readRecipePersistenceUncertainty(id, owner!.ownerId)).toBe(
      "clear"
    )
    expect(mutationCount).toBe(1)
    expectCompiledLocalApply("unknown background caller", "user_message")
  })
})
