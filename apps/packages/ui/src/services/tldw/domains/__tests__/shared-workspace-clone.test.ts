import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  createSharedWorkspaceCloneContext,
  type SharedWorkspaceCloneApi,
  SharedWorkspacePostCommitResponseError
} from "../shared-workspaces"
import { clonePayload, operationId } from "./shared-workspace-clone.fixture"
import { setRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override"
import { createServer } from "node:http"
import { SharedCloneManager } from "@/services/shared-clone-manager"

const nativeFetch = globalThis.fetch

const transport = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  fetch: vi.fn()
}))
vi.mock("@/services/tldw/direct-browser-config", () => ({
  resolveDirectBrowserConfig: async () => transport.config
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getAuthHeaders: async () => ({
      Authorization: `Bearer ${transport.config.accessToken}`
    })
  }
}))
vi.mock("@/services/app", () => ({ getCustomHeaders: async () => ({}) }))
vi.mock("@/services/tldw-server", () => ({
  getTldwServerURL: async () => transport.config.serverUrl
}))
const fetchMock = transport.fetch
let sharedWorkspacesApi: SharedWorkspaceCloneApi
const key = "clone-key-0000000001"
const respond = (body: unknown, status = 200) =>
  fetchMock.mockResolvedValue(new Response(JSON.stringify(body), { status }))

describe("canonical shared workspace clone client", () => {
  beforeEach(async () => {
    fetchMock.mockReset()
    transport.config = {
      serverUrl: "https://tldw.example",
      authMode: "multi-user",
      accessToken: "token-a"
    }
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
    vi.stubGlobal("fetch", transport.fetch)
    respond({ user: { id: 42 } })
    sharedWorkspacesApi = (
      await createSharedWorkspaceCloneContext(new AbortController().signal)
    ).api
    transport.fetch.mockReset()
  })
  afterEach(() => {
    setRuntimeSingleUserApiKeyOverride(null)
    document.cookie = "csrf_token=; Max-Age=0; path=/"
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  it("keeps admission, polling and uncertain replay on the original server and bearer without events", async () => {
    transport.fetch.mockRejectedValueOnce(new TypeError("Response lost"))
    await expect(
      sharedWorkspacesApi.clone(42, { name: "Original" }, key)
    ).rejects.toThrow("Response lost")
    transport.config.serverUrl = "https://other.example"
    transport.config.accessToken = "token-b"
    setRuntimeSingleUserApiKeyOverride("runtime-account-b")
    respond(clonePayload())
    await sharedWorkspacesApi.clone(42, { name: "Original" }, key)
    respond(clonePayload("running"))
    await sharedWorkspacesApi.cloneStatus(42, operationId)
    expect(transport.fetch.mock.calls.map(([url]) => url)).toEqual([
      "https://tldw.example/api/v1/sharing/shared-with-me/42/clone",
      "https://tldw.example/api/v1/sharing/shared-with-me/42/clone",
      `https://tldw.example/api/v1/sharing/shared-with-me/42/clone/${operationId}`
    ])
    for (const [, init] of transport.fetch.mock.calls) {
      const headers = new Headers(init.headers)
      expect(headers.get("Authorization")).toBe("Bearer token-a")
      expect(headers.get("X-API-KEY")).toBeNull()
      expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
    }
    expect(transport.fetch.mock.calls[1][1].body).toBe(
      transport.fetch.mock.calls[0][1].body
    )
    expect(
      new Headers(transport.fetch.mock.calls[1][1].headers).get(
        "Idempotency-Key"
      )
    ).toBe(key)
  })

  it("captures the same credentials before principal verification and preserves a deployment subpath", async () => {
    transport.config.serverUrl = "https://tldw.example/install/"
    transport.config.orgId = 7
    transport.fetch.mockImplementationOnce(async () => {
      transport.config.serverUrl = "https://other.example"
      transport.config.accessToken = "token-b"
      transport.config.orgId = 8
      return new Response(JSON.stringify({ user: { id: 42 } }))
    })
    const context = await createSharedWorkspaceCloneContext(
      new AbortController().signal
    )
    respond(clonePayload())
    await context.api.clone(42, {}, key)
    expect(context.scope).toBe('["https://tldw.example","/install","42","7"]')
    expect(transport.fetch.mock.calls.map(([url]) => url)).toEqual([
      "https://tldw.example/install/api/v1/users/me/profile?sections=identity",
      "https://tldw.example/install/api/v1/sharing/shared-with-me/42/clone"
    ])
    for (const [, init] of transport.fetch.mock.calls) {
      expect(new Headers(init.headers).get("Authorization")).toBe(
        "Bearer token-a"
      )
      expect(new Headers(init.headers).get("X-TLDW-Org-Id")).toBe("7")
    }
  })

  it.each(["configured", "runtime"])(
    "pins %s API-key credentials and excludes ambient cookies",
    async (source) => {
      transport.config = {
        serverUrl: "https://tldw.example",
        authMode: "single-user",
        apiKey: "configured-key-a"
      }
      if (source === "runtime")
        setRuntimeSingleUserApiKeyOverride("runtime-key-a")
      respond({ user: { id: 42 } })
      const { api } = await createSharedWorkspaceCloneContext(
        new AbortController().signal
      )
      transport.config.apiKey = "configured-key-b"
      setRuntimeSingleUserApiKeyOverride("runtime-key-b")
      respond(clonePayload())
      await api.clone(42, {}, key)
      const [, init] = transport.fetch.mock.calls.at(-1)!
      expect(new Headers(init.headers).get("X-API-KEY")).toBe(`${source}-key-a`)
      expect(new Headers(init.headers).get("Authorization")).toBeNull()
      expect(init.credentials).toBe("omit")
    }
  )

  it.each(["quickstart", "hosted"])(
    "uses principal-bound %s cookie requests without stored bearer or API keys",
    async (mode) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", mode)
      transport.config = {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session",
        apiKey: "ignored-key",
        accessToken: "ignored-token"
      }
      setRuntimeSingleUserApiKeyOverride("ignored-runtime-key")
      document.cookie = "csrf_token=csrf-a; path=/"
      respond({ user: { id: 42 } })
      const { api } = await createSharedWorkspaceCloneContext(
        new AbortController().signal
      )
      document.cookie = "csrf_token=csrf-b; path=/"
      respond({ detail: "Principal changed" }, 412)
      await expect(api.clone(42, {}, key)).rejects.toMatchObject({
        status: 412
      })
      const [url, init] = transport.fetch.mock.calls.at(-1)!
      expect(url).toBe(
        `${mode === "hosted" ? "/api/proxy" : "/api/v1"}/sharing/shared-with-me/42/clone`
      )
      const headers = new Headers(init.headers)
      expect(headers.get("X-TLDW-Expected-User-ID")).toBe("42")
      expect(headers.get("X-CSRF-Token")).toBe("csrf-a")
      expect(headers.get("Authorization")).toBeNull()
      expect(headers.get("X-API-KEY")).toBeNull()
      expect(init.credentials).toBe("same-origin")
      expect(init.redirect).toBe("error")
    }
  )

  it.each([401, 403, 412])(
    "never transparently refreshes or replays a rejected %s admission",
    async (status) => {
      transport.config.accessToken = "new-token"
      respond({ detail: "Rejected" }, status)
      await expect(
        sharedWorkspacesApi.clone(42, {}, key)
      ).rejects.toMatchObject({ status })
      expect(transport.fetch).toHaveBeenCalledTimes(1)
    }
  )

  it.each([302, 307, 308])(
    "rejects a %s redirect receipt without following its Location",
    async (status) => {
      transport.fetch.mockResolvedValue(
        new Response(null, {
          status,
          headers: { Location: "https://other.example/clone" }
        })
      )
      await expect(sharedWorkspacesApi.clone(42, {}, key)).rejects.toThrow(
        "redirect"
      )
      expect(transport.fetch).toHaveBeenCalledTimes(1)
      expect(transport.fetch.mock.calls[0][1].redirect).toBe("error")
    }
  )

  it("does not follow a real HTTP clone redirect or forward credentials to its target", async () => {
    let redirectedRequests = 0
    let cloneHeaders: Record<string, unknown> | undefined
    const server = createServer((request, response) => {
      if (request.url?.includes("/users/me/profile")) {
        response.setHeader("Content-Type", "application/json")
        response.end(JSON.stringify({ user: { id: 42 } }))
      } else if (request.url?.endsWith("/clone")) {
        cloneHeaders = request.headers
        response.writeHead(307, { Location: "/redirect-target" })
        response.end()
      } else {
        redirectedRequests++
        response.end(JSON.stringify(clonePayload()))
      }
    })
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject)
      server.listen(0, "127.0.0.1", resolve)
    })
    try {
      const address = server.address()
      if (!address || typeof address === "string")
        throw new Error("Missing test server")
      transport.config.serverUrl = `http://127.0.0.1:${address.port}`
      vi.stubGlobal("fetch", nativeFetch)
      const { api } = await createSharedWorkspaceCloneContext(
        new AbortController().signal
      )
      await expect(api.clone(42, {}, key)).rejects.toThrow()
      expect(cloneHeaders?.authorization).toBe("Bearer token-a")
      expect(cloneHeaders?.["x-tldw-expected-user-id"]).toBe("42")
      expect(redirectedRequests).toBe(0)
    } finally {
      server.closeAllConnections()
      await new Promise<void>((resolve) => server.close(() => resolve()))
    }
  })

  it.each([null, {}, { id: "42" }, { id: 0 }, { id: 1.5 }])(
    "does not expose commands for invalid principal %j",
    async (user) => {
      respond({ user })
      await expect(
        createSharedWorkspaceCloneContext(new AbortController().signal)
      ).rejects.toThrow()
      expect(transport.fetch).toHaveBeenCalledTimes(1)
    }
  )

  it("does not dispatch an aborted queued admission", async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(
      sharedWorkspacesApi.clone(42, {}, key, controller.signal)
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(transport.fetch).not.toHaveBeenCalled()
  })

  it.each(["bearer", "csrf"])(
    "rotates verified %s transport without mixing it into a prior in-flight admission",
    async (credential) => {
      vi.useFakeTimers()
      vi.stubGlobal("navigator", {
        locks: {
          request: (
            _name: string,
            options: unknown,
            callback?: () => unknown
          ) => Promise.resolve().then(callback ?? (options as () => unknown))
        }
      })
      window.localStorage.clear()
      if (credential === "csrf") {
        vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
        transport.config = {
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session"
        }
        document.cookie = "csrf_token=csrf-a; path=/"
      }
      let finishOld!: (value: Response) => void
      let cloneCount = 0
      transport.fetch.mockImplementation(async (url: string) => {
        if (url.includes("/users/me/profile"))
          return new Response(JSON.stringify({ user: { id: 42 } }))
        if (cloneCount++ === 0)
          return new Promise<Response>((resolve) => {
            finishOld = resolve
          })
        return new Response(JSON.stringify(clonePayload("running")))
      })
      const first = await createSharedWorkspaceCloneContext(
        new AbortController().signal
      )
      const manager = new SharedCloneManager(
        first.scope,
        window.localStorage,
        first.api,
        () => undefined
      )
      try {
        await manager.begin(42, "Research")
        await vi.advanceTimersByTimeAsync(0)
        const oldRequest = transport.fetch.mock.calls.find(([url]) =>
          url.endsWith("/clone")
        )![1]
        manager.suspend()
        transport.config.accessToken = "token-b"
        document.cookie = "csrf_token=csrf-b; path=/"
        const next = await createSharedWorkspaceCloneContext(
          new AbortController().signal
        )
        expect(next.scope).toBe(first.scope)
        manager.setVisible(true)
        manager.resume(next.api)
        await vi.advanceTimersByTimeAsync(0)
        const admissions = transport.fetch.mock.calls.filter(([url]) =>
          url.endsWith("/clone")
        )
        expect(admissions).toHaveLength(2)
        const newRequest = admissions[1][1]
        const header =
          credential === "bearer" ? "Authorization" : "X-CSRF-Token"
        expect(new Headers(oldRequest.headers).get(header)).toBe(
          credential === "bearer" ? "Bearer token-a" : "csrf-a"
        )
        expect(new Headers(newRequest.headers).get(header)).toBe(
          credential === "bearer" ? "Bearer token-b" : "csrf-b"
        )
        expect(
          new Headers(newRequest.headers).get("X-TLDW-Expected-User-ID")
        ).toBe("42")
        expect(new Headers(newRequest.headers).get("Idempotency-Key")).toBe(
          new Headers(oldRequest.headers).get("Idempotency-Key")
        )
        expect(newRequest.body).toBe(oldRequest.body)
        expect(oldRequest.signal.aborted).toBe(true)
        finishOld(new Response(JSON.stringify(clonePayload("succeeded"))))
        await vi.advanceTimersByTimeAsync(0)
        expect(manager.rows()[0].operation?.status).toBe("running")
      } finally {
        manager.dispose()
        vi.useRealTimers()
      }
    }
  )

  it("binds the clone transport to the verified principal and rejects redirects", async () => {
    respond(clonePayload())
    await sharedWorkspacesApi.clone(42, {}, key)
    const [, init] = transport.fetch.mock.calls.at(-1)!
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("42")
    expect(init.redirect).toBe("error")
    expect(init.credentials).toBe("omit")
  })

  it("sends the exact key and canonical name, not new_name", async () => {
    respond(clonePayload(), 202)
    const result = await sharedWorkspacesApi.clone(
      42,
      { name: "Research (Copy)" },
      key
    )
    expect(result.status).toBe("queued")
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe(
      "https://tldw.example/api/v1/sharing/shared-with-me/42/clone"
    )
    expect(new Headers(init.headers).get("Idempotency-Key")).toBe(key)
    expect(JSON.parse(init.body)).toEqual({ name: "Research (Copy)" })
  })

  it.each(["queued", "running", "succeeded", "failed"])(
    "reads a %s receipt",
    async (status) => {
      respond(clonePayload(status))
      expect(
        (await sharedWorkspacesApi.cloneStatus(42, operationId)).status
      ).toBe(status)
    }
  )

  it.each([
    ["wrong share", { share_id: 99 }],
    ["wrong version", { schema_version: 2 }],
    ["wrong command", { command: "delete" }],
    ["unsafe poll URL", { poll_href: "https://other.example/clone" }],
    ["missing progress", { progress: null }],
    [
      "oversized progress",
      { progress: { phase: "sources", percent: 101, message_code: "sources" } }
    ],
    ["invalid time", { started_at: "yesterday" }],
    ["invalid boolean", { retryable: "true" }],
    ["unexpected data", { source_content: "private" }]
  ])("treats %s after admission as ambiguous", async (_name, patch) => {
    respond({ ...clonePayload(), ...patch }, 202)
    await expect(sharedWorkspacesApi.clone(42, {}, key)).rejects.toBeInstanceOf(
      SharedWorkspacePostCommitResponseError
    )
  })

  it("rejects a status response from a different operation", async () => {
    respond(clonePayload())
    await expect(
      sharedWorkspacesApi.cloneStatus(
        42,
        "99f28c88-0f13-4b19-80e8-87ddc27bf22b"
      )
    ).rejects.toMatchObject({ status: 502 })
  })

  it("accepts backend-valid names counted in Unicode code points", async () => {
    const name = "\u{1F30A}".repeat(255)
    const payload = clonePayload("succeeded")
    payload.result!.name = name
    respond(payload)
    await expect(
      sharedWorkspacesApi.clone(42, { name }, key)
    ).resolves.toMatchObject({ result: { name } })
    respond(payload)
    await expect(
      sharedWorkspacesApi.cloneStatus(42, operationId)
    ).resolves.toMatchObject({ result: { name } })
  })

  it("rejects names longer than the backend character limit", async () => {
    const name = "\u{1F30A}".repeat(256)
    await expect(sharedWorkspacesApi.clone(42, { name }, key)).rejects.toThrow()
    expect(fetchMock).not.toHaveBeenCalled()
    const payload = clonePayload("succeeded")
    payload.result!.name = name
    respond(payload)
    await expect(
      sharedWorkspacesApi.cloneStatus(42, operationId)
    ).rejects.toMatchObject({ status: 502 })
  })

  it.each(["publication", "target", "counts", "warnings"])(
    "rejects invalid terminal %s",
    async (field) => {
      const payload = clonePayload("succeeded")
      const result = payload.result!
      if (field === "publication") result.publication_confirmed = false
      if (field === "target") result.workspace_id = operationId
      if (field === "counts") result.counts.sources_copied = 3
      if (field === "warnings")
        result.warnings = Array(9).fill({ code: "missing_source", count: 1 })
      respond(payload)
      await expect(
        sharedWorkspacesApi.cloneStatus(42, operationId)
      ).rejects.toMatchObject({ status: 502 })
    }
  )

  it("bounds the response before parsing", async () => {
    respond({ ...clonePayload(), diagnostics: { message: "a".repeat(70_000) } })
    await expect(sharedWorkspacesApi.clone(42, {}, key)).rejects.toBeInstanceOf(
      SharedWorkspacePostCommitResponseError
    )
  })

  it("does not send invalid keys or blank names", async () => {
    await expect(sharedWorkspacesApi.clone(42, {}, "bad key")).rejects.toThrow()
    await expect(
      sharedWorkspacesApi.clone(42, { name: " " }, key)
    ).rejects.toThrow()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("preserves structured admission conflicts", async () => {
    respond(
      {
        detail: {
          code: "clone_already_in_progress",
          operation_id: operationId,
          retryable: true
        }
      },
      409
    )
    await expect(sharedWorkspacesApi.clone(42, {}, key)).rejects.toMatchObject({
      status: 409
    })
  })

  it("preserves a Retry-After cooldown even without a structured body", async () => {
    fetchMock.mockResolvedValue(
      new Response("busy", { status: 429, headers: { "Retry-After": "20" } })
    )
    await expect(sharedWorkspacesApi.clone(42, {}, key)).rejects.toMatchObject({
      status: 429,
      detail: { retry_after_ms: 20_000 }
    })
  })
})
