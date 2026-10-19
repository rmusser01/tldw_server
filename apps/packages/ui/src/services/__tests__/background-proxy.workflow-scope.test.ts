import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  runtimeId: null as string | null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args)
    }
  }
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(async () => {}),
    remove: vi.fn(async () => {})
  })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { bgRequest } from "../background-proxy"
import { deriveSingleUserApiKeyCredentialScope } from "../chat-surface-scope"
import { COOKIE_SESSION_CONFIG_KEY } from "../tldw/browser-networking"
import { requestScopeFields } from "../tldw/domains/service-prompts"
import {
  controlWorkflowRun,
  getWorkflowDefinition,
  getWorkflowInvestigation,
  getWorkflowRun,
  getWorkflowRunSteps,
  getWorkflowStepAttempts,
  listWorkflowDefinitions,
  preflightWorkflowDefinition,
  reviewWorkflowStep,
  runSavedWorkflow,
  saveWorkflowDefinition,
  type WorkflowRequestOptions
} from "../tldw/workflows"

const serverUrl = "https://workflows.example"
const config = (user = 7) => ({
  serverUrl,
  authMode: "multi-user" as const,
  authSource: "manual" as const,
  orgId: 3,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`,
  refreshToken: "fixture-refresh"
})
const options = (): WorkflowRequestOptions => ({
  requestScope: {
    config: {
      serverUrl,
      authMode: "multi-user",
      authSource: "manual",
      orgId: 3,
      expectedRefreshToken: "fixture-refresh"
    },
    userId: 7
  },
  signal: new AbortController().signal
})
const definition = {
  name: "Scoped workflow",
  version: 2,
  visibility: "private" as const,
  steps: [
    { id: "clean", type: "text_clean", config: { text: "{{ inputs.text }}" } }
  ]
}
const summary = { id: 7, name: definition.name, version: 2, is_active: true }
const runId = "9ae1f5c2-57a1-4e09-820d-7e3b9edfd65d"
const stepId = "review:step"
const encodedStepId = encodeURIComponent(stepId)
const inputs = { text: "Functional workflow input." }
const run = {
  run_id: runId,
  workflow_id: 7,
  user_id: "7",
  definition_version: 2,
  status: "queued",
  inputs
}
type Operation = {
  name: string
  path: string
  method: "GET" | "POST"
  body?: unknown
  response: unknown
  invoke: (scope: WorkflowRequestOptions) => Promise<unknown>
}
const operations: Operation[] = [
  {
    name: "list",
    path: "/api/v1/workflows",
    method: "GET",
    response: [summary],
    invoke: listWorkflowDefinitions
  },
  {
    name: "get",
    path: "/api/v1/workflows/7",
    method: "GET",
    response: { ...summary, definition },
    invoke: (scope) => getWorkflowDefinition(7, scope)
  },
  {
    name: "create",
    path: "/api/v1/workflows",
    method: "POST",
    body: definition,
    response: summary,
    invoke: (scope) => saveWorkflowDefinition(definition, null, scope)
  },
  {
    name: "version",
    path: "/api/v1/workflows/7/versions",
    method: "POST",
    body: definition,
    response: summary,
    invoke: (scope) => saveWorkflowDefinition(definition, 7, scope)
  },
  {
    name: "run",
    path: "/api/v1/workflows/7/run?mode=async",
    method: "POST",
    body: {
      inputs,
      idempotency_key: "retained-workflow-attempt",
      validation_mode: "block"
    },
    response: run,
    invoke: (scope) =>
      runSavedWorkflow(7, inputs, "retained-workflow-attempt", scope)
  },
  {
    name: "status",
    path: `/api/v1/workflows/runs/${runId}`,
    method: "GET",
    response: run,
    invoke: (scope) => getWorkflowRun(runId, scope)
  },
  {
    name: "steps",
    path: `/api/v1/workflows/runs/${runId}/steps`,
    method: "GET",
    response: {
      run_id: runId,
      steps: [{ step_id: stepId, status: "waiting_human" }]
    },
    invoke: (scope) => getWorkflowRunSteps(runId, scope)
  },
  ...(["pause", "resume", "cancel"] as const).map(
    (action): Operation => ({
      name: action,
      path: `/api/v1/workflows/runs/${runId}/${action}`,
      method: "POST",
      response: { ok: true, result: "applied" },
      invoke: (scope) => controlWorkflowRun(runId, action, scope)
    })
  ),
  ...(["approve", "reject"] as const).map(
    (action): Operation => ({
      name: action,
      path: `/api/v1/workflows/runs/${runId}/steps/${encodedStepId}/${action}`,
      method: "POST",
      body: {},
      response: { ok: true },
      invoke: (scope) => reviewWorkflowStep(runId, stepId, action, scope)
    })
  ),
  {
    name: "investigation",
    path: `/api/v1/workflows/runs/${runId}/investigation`,
    method: "GET",
    response: {
      run_id: runId,
      status: "queued",
      schema_version: 1,
      derived_from_event_seq: 0,
      attempts: [],
      evidence: {},
      recommended_actions: []
    },
    invoke: (scope) => getWorkflowInvestigation(runId, scope)
  },
  {
    name: "attempts",
    path: `/api/v1/workflows/runs/${runId}/steps/${encodedStepId}/attempts`,
    method: "GET",
    response: { run_id: runId, step_id: stepId, attempts: [] },
    invoke: (scope) => getWorkflowStepAttempts(runId, stepId, scope)
  },
  {
    name: "preflight",
    path: "/api/v1/workflows/preflight",
    method: "POST",
    body: { definition, validation_mode: "block" },
    response: { valid: true, errors: [], warnings: [] },
    invoke: (scope) =>
      preflightWorkflowDefinition(
        { definition, validation_mode: "block" },
        scope
      )
  }
]
const respond = (payload: unknown) => {
  boundary.fetch.mockImplementation(
    async () =>
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
  )
  boundary.sendMessage.mockResolvedValue({
    ok: true,
    status: 200,
    data: payload
  })
}
beforeEach(() => {
  vi.clearAllMocks()
  boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null
  )
  respond(summary)
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  document.cookie = "csrf_token=; Max-Age=0; path=/"
})

describe("canonical workflow service through captured-scope transport", () => {
  it.each(operations)(
    "dispatches canonical $name through direct transport",
    async (operation) => {
      const captured = options()
      respond(operation.response)
      await operation.invoke(captured)
      expect(boundary.fetch).toHaveBeenCalledTimes(1)
      const [url, init] = boundary.fetch.mock.calls[0]
      expect(String(url)).toBe(`${serverUrl}${operation.path}`)
      expect(init.method).toBe(operation.method)
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
      expect(new Headers(init.headers).get("Authorization")).toBe(
        `Bearer ${config().accessToken}`
      )
      expect(init.signal).toBeInstanceOf(AbortSignal)
      if (operation.body === undefined) expect(init.body).toBeUndefined()
      else expect(JSON.parse(init.body)).toEqual(operation.body)
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    }
  )

  it.each(operations)(
    "retains captured scope for extension $name",
    async (operation) => {
      boundary.runtimeId = "extension-fixture"
      const captured = options()
      respond(operation.response)
      await operation.invoke(captured)
      expect(boundary.sendMessage).toHaveBeenCalledTimes(1)
      const message = boundary.sendMessage.mock.calls[0][0]
      expect(message.type).toBe("tldw:request")
      expect(message.payload).toMatchObject({
        path: operation.path,
        method: operation.method,
        servicePromptConfig: {
          ...captured.requestScope.config,
          expectedUserId: 7
        },
        headers: { "X-TLDW-Expected-User-ID": "7" },
        timeoutMs: 15000,
        requestId: expect.any(String)
      })
      expect(message.payload.body).toEqual(operation.body)
      expect(message.payload).not.toHaveProperty("configSnapshot")
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )

  it("uses actual Quickstart cookie-session Save transport without exposing a key", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    const cookieConfig = {
      serverUrl: window.location.origin,
      authMode: "single-user" as const,
      authSource: "cookie-session" as const
    }
    boundary.get.mockImplementation(async (key: string) =>
      key === COOKIE_SESSION_CONFIG_KEY
        ? cookieConfig
        : key === "tldwConfig"
          ? config()
          : null
    )
    document.cookie = "csrf_token=workflow-csrf; path=/"
    await saveWorkflowDefinition(definition, null, {
      requestScope: {
        config: {
          ...cookieConfig,
          expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
            "single-user",
            null
          )!
        },
        userId: 1
      },
      signal: new AbortController().signal
    })
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(new URL(String(url), window.location.origin).href).toBe(
      `${window.location.origin}/api/v1/workflows`
    )
    expect(init.credentials).toBe("same-origin")
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBe("workflow-csrf")
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
    expect(new Headers(init.headers).get("X-API-KEY")).toBeNull()
    expect(new Headers(init.headers).get("Authorization")).toBeNull()
  })

  it.each([
    { name: "principal", current: config(8) },
    {
      name: "server",
      current: { ...config(), serverUrl: "https://other.example" }
    },
    { name: "auth mode", current: { ...config(), authMode: "single-user" } },
    {
      name: "auth source",
      current: { ...config(), authSource: "cookie-session" }
    },
    { name: "organization", current: { ...config(), orgId: 4 } },
    {
      name: "refresh lineage",
      current: { ...config(), refreshToken: "other-refresh" }
    }
  ])(
    "refuses Save after $name changes during credential loading",
    async ({ current }) => {
      let release!: (value: unknown) => void
      const pendingConfig = new Promise((resolve) => {
        release = resolve
      })
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig" ? pendingConfig : null
      )
      const pending = saveWorkflowDefinition(definition, null, options())
      const rejected = expect(pending).rejects.toMatchObject({
        status: 412,
        details: { detail: { code: "request_config_scope_changed" } }
      })
      release(current)
      await rejected
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    }
  )

  it.each([false, true])(
    "keeps the captured single-user key fence with changed=%s",
    async (changed) => {
      const keyConfig = {
        serverUrl,
        authMode: "single-user" as const,
        authSource: "manual" as const,
        credentialSource: "manual" as const,
        apiKeyPersistence: "device" as const,
        apiKeyServerOrigin: serverUrl,
        apiKey: "workflow-fixture-key"
      }
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig"
          ? {
              ...keyConfig,
              apiKey: changed ? "other-fixture-key" : keyConfig.apiKey
            }
          : null
      )
      const pending = saveWorkflowDefinition(definition, null, {
        requestScope: {
          config: {
            serverUrl,
            authMode: "single-user",
            authSource: "manual",
            expectedSingleUserApiKeyScope:
              deriveSingleUserApiKeyCredentialScope(
                "single-user",
                keyConfig.apiKey
              )!
          },
          userId: null
        },
        signal: new AbortController().signal
      })
      if (changed) {
        await expect(pending).rejects.toMatchObject({ status: 412 })
        expect(boundary.fetch).not.toHaveBeenCalled()
      } else {
        await expect(pending).resolves.toEqual(summary)
        expect(
          new Headers(boundary.fetch.mock.calls[0][1].headers).get("X-API-KEY")
        ).toBe(keyConfig.apiKey)
      }
    }
  )

  it("uses current access credentials for the same captured principal", async () => {
    const current = {
      ...config(),
      accessToken: `rotated.${btoa(JSON.stringify({ sub: "7" }))}.signature`
    }
    boundary.get.mockImplementation(async (key: string) =>
      key === "tldwConfig" ? current : null
    )
    await expect(
      saveWorkflowDefinition(definition, null, options())
    ).resolves.toEqual(summary)
    expect(
      new Headers(boundary.fetch.mock.calls[0][1].headers).get("Authorization")
    ).toBe(`Bearer ${current.accessToken}`)
  })

  it.each([
    ["/api/v1/workflows", "PUT"],
    ["/api/v1/workflows", "DELETE"],
    ["/api/v1/workflows/7", "POST"],
    ["/api/v1/workflows/7", "DELETE"],
    ["/api/v1/workflows/7/versions", "GET"],
    ["/api/v1/workflows/7/run", "GET"],
    ["/api/v1/workflows/preflight", "GET"],
    [`/api/v1/workflows/runs/${runId}`, "POST"],
    [`/api/v1/workflows/runs/${runId}/steps`, "POST"],
    [`/api/v1/workflows/runs/${runId}/investigation`, "POST"],
    [`/api/v1/workflows/runs/${runId}/steps/${encodedStepId}/attempts`, "POST"],
    [`/api/v1/workflows/runs/${runId}/cancel`, "GET"],
    [`/api/v1/workflows/runs/${runId}/steps/${encodedStepId}/approve`, "GET"],
    ["/api/v1/workflows/step-types", "GET"],
    ["/api/v1/workflows/auth/check", "GET"],
    ["/api/v1/workflows/auth/virtual-key", "POST"],
    ["/api/v1/workflows/config", "GET"],
    ["/api/v1/workflows/run", "POST"],
    ["/api/v1/workflows/runs", "GET"],
    [`/api/v1/workflows/runs/${runId}/retry`, "POST"],
    [`/api/v1/workflows/runs/${runId}/artifacts`, "GET"],
    ["/api/v1/workflows/webhooks/dlq", "GET"],
    ["/api/v1/workflows/unknown", "GET"],
    ["/api/v1/workflows/0", "GET"],
    ["/api/v1/workflows/-1", "GET"],
    ["/api/v1/workflows/1.5", "GET"]
  ])("refuses unrepresented scoped pair %s %s", async (path, method) => {
    await expect(
      bgRequest({
        path: path as never,
        method: method as never,
        ...requestScopeFields(options().requestScope)
      })
    ).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })

  it.each([
    "/api/v1/workflows/",
    "/api/v1/workflows//7",
    "/api/v1/workflows/../workflows/7",
    "/api/v1/workflows/%2e%2e/7",
    "/api/v1/workflows/7%2fversions",
    `/api/v1/workflows/runs/${runId}/steps/clean%5cstep/attempts`,
    "/api/v1/workflows/runs/owned\\id",
    "/api/v1/workflows/runs/%zz",
    "/api/v1/workflows/runs/./steps",
    "/api/v1/workflows/runs/a%2fb/steps",
    `/api/v1/workflows/runs/${runId}/steps/${encodedStepId}/attempts/extra`,
    `https://workflows.example/api/v1/workflows/runs/${runId}`
  ])("refuses malformed workflow pathname %s before dispatch", async (path) => {
    await expect(
      bgRequest({
        path: path as never,
        method: "GET" as never,
        ...requestScopeFields(options().requestScope)
      })
    ).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })

  it.each([
    { name: "run slash", invoke: () => getWorkflowRun("owned/run", options()) },
    {
      name: "run dot",
      invoke: () => getWorkflowInvestigation("..", options())
    },
    {
      name: "run backslash",
      invoke: () => getWorkflowRunSteps("owned\\run", options())
    },
    {
      name: "step slash",
      invoke: () => getWorkflowStepAttempts(runId, "review/step", options())
    },
    {
      name: "step dot",
      invoke: () => reviewWorkflowStep(runId, "..", "approve", options())
    },
    {
      name: "step backslash",
      invoke: () =>
        reviewWorkflowStep(runId, "review\\step", "reject", options())
    }
  ])("refuses unsafe encoded service identifier $name", async ({ invoke }) => {
    await expect(invoke()).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })
})
