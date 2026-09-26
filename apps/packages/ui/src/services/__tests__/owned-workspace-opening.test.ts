import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  createOwnedWorkspaceReadContext,
  createOwnedWorkspaceNotesContext,
  createOwnedWorkspaceMetadataContext,
  createOwnedWorkspaceLifecycleContext,
  createOwnedWorkspaceDirectoryContext
} from "../owned-workspace-opening"
import { setRuntimeSingleUserApiKeyOverride } from "../tldw/runtime-auth-override"

const config = vi.hoisted(() => ({
  current: {} as Record<string, unknown>
}))
vi.mock("@/services/tldw/direct-browser-config", () => ({
  resolveDirectBrowserConfig: async () => config.current
}))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({}) }))

const id = "47fd3ca9-a34b-5e36-be11-ebc15ce38fe6"
const metadataScope = {
  serverBase: "https://research.example/install",
  principalId: "3",
  organizationId: "7"
}
const workspace = {
  id,
  name: "Recipient copy",
  archived: false,
  deleted: false,
  workspace_profile: "research",
  study_materials_policy: "workspace",
  banner_title: null,
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null,
  created_at: "2026-09-13T12:00:00Z",
  last_modified: "2026-09-13T12:00:00Z",
  version: 2,
  assistant_defaults: {
    assistant_kind: "persona",
    assistant_id: "persona-1",
    persona_memory_mode: "read_only"
  }
}
let principal: unknown
let responses: Record<string, unknown>
let failure: { suffix: string; status: number } | null
let requests: { url: string; init: RequestInit }[]

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "advanced")
  config.current = {
    serverUrl: "https://research.example/install/",
    authMode: "multi-user",
    accessToken: "test-token-a",
    refreshToken: "test-refresh",
    orgId: 7
  }
  principal = { id: 3 }
  responses = { "": workspace, "/sources": [], "/artifacts": [], "/notes": [] }
  failure = null
  requests = []
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string, init: RequestInit) => {
      requests.push({ url: String(input), init })
      if (failure && String(input).endsWith(failure.suffix))
        return new Response(JSON.stringify({ detail: "Unavailable" }), {
          status: failure.status,
          headers: { "Content-Type": "application/json" }
        })
      if (String(input).endsWith("/auth/me"))
        return new Response("Legacy identity endpoints disabled", {
          status: 410
        })
      const body = String(input).endsWith("/users/me/profile?sections=identity")
        ? { user: principal }
        : String(input).endsWith("/persona/catalog?ensure_default=false")
          ? responses["personas"]
          : responses[String(input).split(`/workspaces/${id}`)[1]]
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    })
  )
})
afterEach(() => {
  setRuntimeSingleUserApiKeyOverride(null)
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe("owned workspace lifecycle context", () => {
  const context = (signal = new AbortController().signal) =>
    createOwnedWorkspaceLifecycleContext(id, metadataScope, signal)

  it.each([true, false])(
    "pins an explicit archived=%s write to the loaded account",
    async (archived) => {
      const lifecycle = await context()
      config.current.accessToken = "other-account"
      config.current.serverUrl = "https://other.example"
      responses[""] = { ...workspace, archived, version: 3 }
      expect((await lifecycle.setArchived(archived, 2)).archived).toBe(archived)
      const { url, init } = requests.at(-1)!
      expect(url).toBe(`${metadataScope.serverBase}/api/v1/workspaces/${id}`)
      expect(init.method).toBe("PATCH")
      expect(JSON.parse(init.body as string)).toEqual({ archived, version: 2 })
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
      expect(new Headers(init.headers).get("Authorization")).toBe(
        "Bearer test-token-a"
      )
    }
  )
  it("can inspect an archived workspace without opening it for editing", async () => {
    responses[""] = { ...workspace, archived: true }
    expect((await (await context()).get()).archived).toBe(true)
    await expect(
      (
        await createOwnedWorkspaceMetadataContext(
          id,
          metadataScope,
          new AbortController().signal
        )
      ).get()
    ).rejects.toThrow()
  })
  it.each([
    { ...workspace, archived: true },
    { ...workspace, version: 3 },
    { ...workspace, archived: true, version: 3, deleted: true },
    { ...workspace, archived: true, version: 3, id: "wrong" },
    { ...workspace, archived: "true", version: 3 }
  ])("rejects untrustworthy archive receipts", async (response) => {
    responses[""] = response
    await expect((await context()).setArchived(true, 2)).rejects.toThrow()
  })
  it.each([0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "does not send invalid version %s",
    async (version) => {
      const lifecycle = await context()
      await expect(lifecycle.setArchived(true, version)).rejects.toThrow()
      expect(requests.some(({ init }) => init.method === "PATCH")).toBe(false)
    }
  )
  it("rejects an account change before sending archive", async () => {
    principal = { id: 4 }
    await expect(context()).rejects.toMatchObject({ status: 412 })
    expect(requests).toHaveLength(1)
  })
  it.each([409, 412, 503])(
    "does not retry lifecycle failure %s",
    async (status) => {
      const lifecycle = await context()
      failure = { suffix: `/workspaces/${id}`, status }
      await expect(lifecycle.setArchived(true, 2)).rejects.toMatchObject({
        status
      })
      expect(
        requests.filter(({ init }) => init.method === "PATCH")
      ).toHaveLength(1)
    }
  )
  it("does not dispatch an aborted archive", async () => {
    const controller = new AbortController()
    const lifecycle = await context(controller.signal)
    controller.abort()
    await expect(lifecycle.setArchived(true, 2)).rejects.toThrow()
    expect(requests.some(({ init }) => init.method === "PATCH")).toBe(false)
  })
})

describe("owned workspace directory context", () => {
  const details = () => ({
    workspace_id: id,
    workspace,
    attention_state: "ready",
    project_root: {
      state: "not_configured",
      root_id: null,
      backend: null,
      display_name: null,
      path_hint: null,
      git_state: null,
      file_inventory_state: "not_started",
      file_inventory: {
        state: "not_started",
        indexed_file_count: null,
        total_file_count: null,
        updated_at: null,
        available: false
      },
      indexing_state: null,
      sandbox_mount_state: null,
      mcp_trust_state: null
    },
    sources: { items: [], summary: { total: 1, selected: 1 } },
    active_operations: []
  })
  it("returns only validated directory details, not unchecked context fields", async () => {
    const directory = await createOwnedWorkspaceDirectoryContext(
      new AbortController().signal
    )
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          ...details(),
          capabilities: { workspace_id: "ignored" }
        }),
        { headers: { "Content-Type": "application/json" } }
      )
    )
    const loaded = await directory.getContext(id)
    expect(loaded.sources.summary).toEqual({ total: 1, selected: 1 })
    expect(loaded).not.toHaveProperty("capabilities")
    expect(loaded).not.toHaveProperty("workspace")
    expect(loaded.sources).not.toHaveProperty("items")
  })
  it("preserves a same-workspace operation with populated diagnostics", async () => {
    const directory = await createOwnedWorkspaceDirectoryContext(
      new AbortController().signal
    )
    const operation = {
      operation_id: "op-1",
      workspace_id: id,
      command: "attach_root",
      status: "running",
      started_at: "2026-09-20T12:00:00Z",
      updated_at: "2026-09-20T12:01:00Z",
      retryable: false,
      diagnostics: { code: "waiting", completed_files: 2 },
      poll_href: "/unused"
    }
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(
        JSON.stringify({ ...details(), active_operations: [operation] }),
        { headers: { "Content-Type": "application/json" } }
      )
    )
    expect((await directory.getContext(id)).active_operations).toEqual([
      operation
    ])
  })
  it.each([
    () => ({ workspace_id: id, workspace }),
    () => ({ ...details(), project_root: { state: "attached" } }),
    () => ({ ...details(), sources: { summary: { total: "1", selected: 0 } } }),
    () => ({ ...details(), sources: { summary: { total: 1, selected: 2 } } }),
    () => ({
      ...details(),
      active_operations: [
        {
          operation_id: "op",
          workspace_id: "foreign",
          command: "attach",
          status: "running",
          started_at: "now",
          updated_at: "now",
          retryable: false,
          diagnostics: {},
          poll_href: "/unused"
        }
      ]
    })
  ])("rejects invalid directory projections", async (response) => {
    const directory = await createOwnedWorkspaceDirectoryContext(
      new AbortController().signal
    )
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify(response()), {
        headers: { "Content-Type": "application/json" }
      })
    )
    await expect(directory.getContext(id)).rejects.toThrow()
  })
  it("loads active and archived records through an account-pinned directory", async () => {
    const fetchMock = vi.mocked(fetch)
    const directory = await createOwnedWorkspaceDirectoryContext(
      new AbortController().signal
    )
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          items: [workspace, { ...workspace, id: "archived", archived: true }],
          total: 2
        }),
        { headers: { "Content-Type": "application/json" } }
      )
    )
    expect((await directory.list()).items.map((row) => row.archived)).toEqual([
      false,
      true
    ])
    const [url, init] = fetchMock.mock.calls.at(-1)!
    expect(url).toBe(`${metadataScope.serverBase}/api/v1/workspaces/`)
    expect(new Headers(init?.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
    expect(directory.scope).toEqual(metadataScope)
  })
  it.each([
    {},
    { items: [workspace, workspace] },
    { items: [{ ...workspace, deleted: true }] }
  ])("rejects an incomplete or invalid directory", async (response) => {
    const directory = await createOwnedWorkspaceDirectoryContext(
      new AbortController().signal
    )
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify(response), {
        headers: { "Content-Type": "application/json" }
      })
    )
    await expect(directory.list()).rejects.toThrow()
  })
  it("rejects context for a different workspace", async () => {
    const directory = await createOwnedWorkspaceDirectoryContext(
      new AbortController().signal
    )
    vi.mocked(fetch).mockResolvedValueOnce(
      new Response(JSON.stringify({ workspace_id: "wrong", workspace }), {
        headers: { "Content-Type": "application/json" }
      })
    )
    await expect(directory.getContext(id)).rejects.toThrow()
    const [url, init] = vi.mocked(fetch).mock.calls.at(-1)!
    expect(url).toBe(
      `${metadataScope.serverBase}/api/v1/workspaces/${id}/context`
    )
    expect(new Headers(init?.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
  })
})

describe("owned workspace metadata context", () => {
  const context = (signal = new AbortController().signal) =>
    createOwnedWorkspaceMetadataContext(id, metadataScope, signal)

  it("loads read-only persona options through the captured account and connection", async () => {
    responses.personas = [
      { id: "persona-1", name: "Researcher", extra: "not retained" }
    ]
    const metadata = await context()
    config.current.accessToken = "changed"
    config.current.serverUrl = "https://other.example"
    expect(await metadata.listPersonas()).toEqual([
      { id: "persona-1", name: "Researcher" }
    ])
    const { url, init } = requests.at(-1)!
    expect(url).toBe(
      `${metadataScope.serverBase}/api/v1/persona/catalog?ensure_default=false`
    )
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
    expect(new Headers(init.headers).get("Authorization")).toBe(
      "Bearer test-token-a"
    )
    expect(init.method).toBe("GET")
    expect(init.cache).toBe("no-store")
    expect(init.redirect).toBe("error")
  })

  it.each([
    null,
    {},
    [{ id: "", name: "Bad" }],
    [{ id: "p", name: 2 }],
    [
      { id: "p", name: "A" },
      { id: "p", name: "B" }
    ]
  ])("rejects malformed or duplicate persona choices %j", async (payload) => {
    responses.personas = payload
    await expect((await context()).listPersonas()).rejects.toThrow()
  })

  it("preserves a successful empty persona catalog", async () => {
    responses.personas = []
    expect(await (await context()).listPersonas()).toEqual([])
  })

  it.each([401, 403, 412, 500])(
    "does not retry failed persona catalog %s",
    async (status) => {
      failure = { suffix: "/persona/catalog?ensure_default=false", status }
      await expect((await context()).listPersonas()).rejects.toMatchObject({
        status
      })
      expect(requests).toHaveLength(2)
    }
  )

  it("does not dispatch persona catalog after cancellation", async () => {
    const controller = new AbortController()
    const metadata = await context(controller.signal)
    controller.abort()
    await expect(metadata.listPersonas()).rejects.toMatchObject({
      name: "AbortError"
    })
    expect(requests).toHaveLength(1)
  })

  it("uses pinned authenticated GET and versioned PATCH, never create-capable PUT", async () => {
    const metadata = await context()
    config.current.accessToken = "other-account"
    config.current.serverUrl = "https://other.example"
    const original = await metadata.get()
    responses[""] = { ...workspace, name: "Renamed", version: 3 }
    const result = await metadata.patch({
      name: "Renamed",
      version: original.version
    })
    expect(result.name).toBe("Renamed")
    expect(result.assistantDefaults?.assistantId).toBe("persona-1")
    const calls = requests.slice(1)
    expect(calls.map(({ init }) => init.method)).toEqual(["GET", "PATCH"])
    for (const { url, init } of calls) {
      expect(url).toBe(`${metadataScope.serverBase}/api/v1/workspaces/${id}`)
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
      expect(new Headers(init.headers).get("Authorization")).toBe(
        "Bearer test-token-a"
      )
      expect(new Headers(init.headers).get("X-TLDW-Org-Id")).toBe("7")
      expect(init.cache).toBe("no-store")
      expect(init.redirect).toBe("error")
    }
    expect(JSON.parse(String(calls[1].init.body))).toEqual({
      name: "Renamed",
      version: 2
    })
  })

  it.each([
    { ...metadataScope, principalId: "4" },
    { ...metadataScope, organizationId: "8" },
    { ...metadataScope, serverBase: "https://other.example" }
  ])(
    "rejects a different activated scope before metadata dispatch",
    async (scope) => {
      await expect(
        createOwnedWorkspaceMetadataContext(
          id,
          scope,
          new AbortController().signal
        )
      ).rejects.toMatchObject({ status: 412 })
      expect(requests).toHaveLength(1)
    }
  )

  it.each([0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1])(
    "rejects invalid PATCH version %s before dispatch",
    async (version) => {
      const metadata = await context()
      await expect(metadata.patch({ name: "Draft", version })).rejects.toThrow()
      expect(requests).toHaveLength(1)
    }
  )

  it("rejects an empty PATCH before dispatch", async () => {
    const metadata = await context()
    await expect(metadata.patch({ version: 2 })).rejects.toThrow()
    expect(requests).toHaveLength(1)
  })

  it.each([
    { archived: true },
    { archived: false },
    { workspace_profile: "project" as const }
  ])(
    "does not perform lifecycle changes through metadata editing",
    async (fields) => {
      const metadata = await context()
      await expect(metadata.patch({ version: 2, ...fields })).rejects.toThrow()
      expect(requests).toHaveLength(1)
    }
  )

  it("rejects a foreign GET response", async () => {
    const metadata = await context()
    responses[""] = { ...workspace, id: "foreign" }
    await expect(metadata.get()).rejects.toThrow()
  })

  it("discards a late response after cancellation", async () => {
    const controller = new AbortController()
    const metadata = await context(controller.signal)
    let finish!: (response: Response) => void
    vi.mocked(fetch).mockImplementationOnce(
      () =>
        new Promise<Response>((resolve) => {
          finish = resolve
        })
    )
    const pending = metadata.patch({ version: 2, name: "Draft" })
    const rejected = expect(pending).rejects.toMatchObject({
      name: "AbortError"
    })
    await vi.waitFor(() => expect(finish).toBeDefined())
    controller.abort()
    finish(
      new Response(
        JSON.stringify({ ...workspace, name: "Draft", version: 3 }),
        {
          headers: { "Content-Type": "application/json" }
        }
      )
    )
    await rejected
  })

  it.each([409, 412, 500])(
    "does not retry a rejected metadata write (%s)",
    async (status) => {
      const metadata = await context()
      failure = { suffix: `/workspaces/${id}`, status }
      await expect(
        metadata.patch({ name: "Draft", version: 2 })
      ).rejects.toMatchObject({ status })
      expect(
        requests.filter(({ init }) => init.method === "PATCH")
      ).toHaveLength(1)
      expect(requests.some(({ url }) => url.includes("refresh"))).toBe(false)
    }
  )

  it.each([
    { ...workspace, id: "foreign", version: 3 },
    { ...workspace, deleted: true, version: 3 },
    { ...workspace, archived: true, version: 3 },
    { ...workspace, name: 42, version: 3 },
    { ...workspace, version: 2 },
    { ...workspace, version: 1 }
  ])(
    "rejects foreign, unavailable, malformed or non-advancing receipts",
    async (receipt) => {
      const metadata = await context()
      responses[""] = receipt
      await expect(
        metadata.patch({ name: "Draft", version: 2 })
      ).rejects.toThrow()
    }
  )

  it("snapshots and serializes nested assistant settings before request setup", async () => {
    const metadata = await context()
    responses[""] = { ...workspace, version: 3 }
    const body = {
      version: 2,
      banner_title: null,
      audio_voice: "",
      assistantDefaults: {
        assistantKind: "persona" as const,
        assistantId: "submitted",
        personaMemoryMode: "read_write" as const,
        voice: null,
        style: null,
        toolPolicyProfileId: null
      },
      confirmReadWriteAssistantDefault: true
    }
    const pending = metadata.patch(body)
    body.assistantDefaults.assistantId = "later-edit"
    body.confirmReadWriteAssistantDefault = false
    body.version = 99
    await pending
    expect(JSON.parse(String(requests.at(-1)!.init.body))).toEqual({
      version: 2,
      banner_title: null,
      audio_voice: "",
      assistant_defaults: {
        assistant_kind: "persona",
        assistant_id: "submitted",
        persona_memory_mode: "read_write",
        voice: null,
        style: null,
        tool_policy_profile_id: null
      },
      confirm_read_write_assistant_default: true
    })
  })

  it("preserves explicit assistant-default clearing", async () => {
    const metadata = await context()
    responses[""] = { ...workspace, version: 3, assistant_defaults: null }
    const result = await metadata.patch({ version: 2, assistantDefaults: null })
    expect(JSON.parse(String(requests.at(-1)!.init.body))).toEqual({
      version: 2,
      assistant_defaults: null
    })
    expect(result.assistantDefaults).toBeNull()
  })

  it("does not dispatch after the operation lifetime is aborted", async () => {
    const controller = new AbortController()
    const metadata = await context(controller.signal)
    controller.abort()
    await expect(metadata.get()).rejects.toMatchObject({ name: "AbortError" })
    await expect(
      metadata.patch({ name: "Draft", version: 2 })
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(requests).toHaveLength(1)
  })

  it.each(["hosted", "quickstart"])(
    "asserts expected user on %s cookie metadata writes",
    async (mode) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", mode)
      config.current = {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      }
      const metadata = await createOwnedWorkspaceMetadataContext(
        id,
        {
          serverBase: window.location.origin,
          principalId: "3",
          organizationId: null
        },
        new AbortController().signal
      )
      failure = { suffix: `/workspaces/${id}`, status: 412 }
      await expect(
        metadata.patch({ name: "Draft", version: 2 })
      ).rejects.toMatchObject({ status: 412 })
      const { init } = requests.at(-1)!
      expect(init.credentials).toBe("same-origin")
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
      expect(new Headers(init.headers).get("Authorization")).toBeNull()
      expect(new Headers(init.headers).get("X-API-KEY")).toBeNull()
    }
  )
})

describe("owned workspace verified read context", () => {
  it.each([
    "",
    ".",
    "..",
    " a",
    "a ",
    "a/b",
    "a\\b",
    "a?b",
    "a#b",
    "%2e%2e",
    "a\n",
    "a".repeat(129)
  ])("rejects a malformed target %j before dispatch", async (target) => {
    await expect(
      createOwnedWorkspaceReadContext(target, new AbortController().signal)
    ).rejects.toMatchObject({ reason: "invalid-response" })
    expect(requests).toEqual([])
  })
  it("pins server, bearer and organization across every required read", async () => {
    const context = await createOwnedWorkspaceReadContext(
      id,
      new AbortController().signal
    )
    config.current.serverUrl = "https://other.example"
    config.current.accessToken = "other-user-token"
    config.current.orgId = 8
    setRuntimeSingleUserApiKeyOverride("runtime-other-user")
    const bundle = await context.load()
    expect(context.scope).toEqual({
      serverBase: "https://research.example/install",
      principalId: "3",
      organizationId: "7"
    })
    expect(bundle.workspace.assistantDefaults?.assistantId).toBe("persona-1")
    expect(requests.map(({ url }) => url)).toEqual([
      "https://research.example/install/api/v1/users/me/profile?sections=identity",
      `https://research.example/install/api/v1/workspaces/${id}`,
      `https://research.example/install/api/v1/workspaces/${id}/sources`,
      `https://research.example/install/api/v1/workspaces/${id}/artifacts`,
      `https://research.example/install/api/v1/workspaces/${id}/notes`,
      "https://research.example/install/api/v1/users/me/profile?sections=identity"
    ])
    for (const { url, init } of requests) {
      expect(init.method).toBe("GET")
      expect(init.headers).toEqual({
        Authorization: "Bearer test-token-a",
        "X-TLDW-Org-Id": "7",
        ...(url.includes("/workspaces/")
          ? { "X-TLDW-Expected-User-ID": "3" }
          : {})
      })
      expect(init.credentials).toBe("omit")
      expect(init.redirect).toBe("error")
      expect(init.cache).toBe("no-store")
    }
  })

  it("captures the runtime API key without retaining a mutable override", async () => {
    setRuntimeSingleUserApiKeyOverride("runtime-user-a")
    const context = await createOwnedWorkspaceReadContext(
      id,
      new AbortController().signal
    )
    setRuntimeSingleUserApiKeyOverride("runtime-user-b")
    await context.load()
    expect(
      requests.every(
        ({ init }) =>
          new Headers(init.headers).get("X-API-KEY") === "runtime-user-a"
      )
    ).toBe(true)
  })

  it.each([401, 403, 404, 500])(
    "retains HTTP status %i without refreshing or returning an empty bundle",
    async (status) => {
      const context = await createOwnedWorkspaceReadContext(
        id,
        new AbortController().signal
      )
      failure = { suffix: "/notes", status }
      await expect(context.load()).rejects.toMatchObject({ status })
      expect(requests.filter(({ url }) => url.endsWith("/notes"))).toHaveLength(
        1
      )
      expect(requests.every(({ init }) => init.method === "GET")).toBe(true)
    }
  )

  it.each([null, {}, { id: 0 }, { id: -1 }, { id: "3" }, { id: 1.2 }])(
    "rejects an unverified principal %j",
    async (value) => {
      principal = value
      await expect(
        createOwnedWorkspaceReadContext(id, new AbortController().signal)
      ).rejects.toMatchObject({ reason: "denied" })
      expect(requests).toHaveLength(1)
    }
  )

  it("rejects a user change during the bundle read", async () => {
    const context = await createOwnedWorkspaceReadContext(
      id,
      new AbortController().signal
    )
    principal = { id: 4 }
    await expect(context.load()).rejects.toMatchObject({ reason: "denied" })
  })

  describe.each(["hosted", "quickstart"])("%s cookie bundle reads", (mode) => {
    it.each(["", "/sources", "/artifacts", "/notes"])(
      "rejects A-to-B-to-A at %j without exposing a bundle to activation",
      async (suffix) => {
        vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", mode)
        config.current = {
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session"
        }
        const foreignContent: Record<string, unknown> = {
          "": { ...workspace, name: "Account B workspace" },
          "/sources": [
            {
              id: "source-b",
              workspace_id: id,
              media_id: 42,
              title: "Account B source",
              source_type: "text",
              url: null,
              position: 0,
              selected: true,
              added_at: workspace.created_at,
              version: 1
            }
          ],
          "/artifacts": [
            {
              id: "artifact-b",
              workspace_id: id,
              artifact_type: "summary",
              title: "Account B artifact",
              content: "Private B content",
              status: "completed",
              total_tokens: null,
              total_cost_usd: null,
              created_at: workspace.created_at,
              completed_at: null,
              version: 1
            }
          ],
          "/notes": [{ ...note, content: "Account B note" }]
        }
        const context = await createOwnedWorkspaceReadContext(
          id,
          new AbortController().signal
        )
        const originalFetch = vi.mocked(fetch).getMockImplementation()!
        const observedPrincipals: number[] = [3]
        vi.mocked(fetch).mockImplementation(async (input, init) => {
          if (!String(input).endsWith(`/workspaces/${id}${suffix}`))
            return originalFetch(input, init)
          requests.push({ url: String(input), init: init! })
          // The cookie changes only while this request is authenticated. Both
          // identity probes would see A, despite the colliding workspace in B.
          principal = { id: 4 }
          observedPrincipals.push(4)
          const expected = new Headers(init?.headers).get(
            "X-TLDW-Expected-User-ID"
          )
          const rejected = expected !== null && expected !== "4"
          const response = new Response(
            JSON.stringify(
              rejected
                ? { detail: { code: "request_config_scope_changed" } }
                : foreignContent[suffix]
            ),
            {
              status: rejected ? 412 : 200,
              headers: { "Content-Type": "application/json" }
            }
          )
          principal = { id: 3 }
          observedPrincipals.push(3)
          return response
        })
        const activate = vi.fn()
        await expect(context.load().then(activate)).rejects.toMatchObject({
          status: 412
        })
        expect(activate).not.toHaveBeenCalled()
        expect(observedPrincipals).toEqual([3, 4, 3])
        const switchedReads = requests.filter(({ url }) =>
          url.endsWith(`/workspaces/${id}${suffix}`)
        )
        expect(switchedReads).toHaveLength(1)
        const headers = new Headers(switchedReads[0].init.headers)
        expect(headers.get("X-TLDW-Expected-User-ID")).toBe("3")
        expect(headers.get("Authorization")).toBeNull()
        expect(headers.get("X-API-KEY")).toBeNull()
        expect(switchedReads[0].init.credentials).toBe("same-origin")
      }
    )
  })

  it("rejects malformed metadata before normalization can hide it", async () => {
    responses[""] = {
      ...workspace,
      assistant_defaults: { assistant_kind: "invalid" }
    }
    const context = await createOwnedWorkspaceReadContext(
      id,
      new AbortController().signal
    )
    await expect(context.load()).rejects.toMatchObject({
      reason: "invalid-response"
    })
  })

  it("does not dispatch an aborted opening", async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(
      createOwnedWorkspaceReadContext(id, controller.signal)
    ).rejects.toMatchObject({ name: "AbortError" })
    expect(requests).toEqual([])
  })

  it("aborts between context verification and loading without sending more reads", async () => {
    const controller = new AbortController()
    const context = await createOwnedWorkspaceReadContext(id, controller.signal)
    controller.abort()
    await expect(context.load()).rejects.toMatchObject({ name: "AbortError" })
    expect(requests).toHaveLength(1)
  })

  it.each([
    "file:///tmp/data",
    "https://user:password@example.com",
    "https://example.com?token=secret",
    "https://example.com#fragment"
  ])("rejects unsafe configured server %s before dispatch", async (url) => {
    config.current.serverUrl = url
    await expect(
      createOwnedWorkspaceReadContext(id, new AbortController().signal)
    ).rejects.toMatchObject({ reason: "connection" })
    expect(requests).toEqual([])
  })

  it("uses the hosted proxy and browser session, not a configured remote key", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    setRuntimeSingleUserApiKeyOverride("unused-runtime-key")
    const context = await createOwnedWorkspaceReadContext(
      id,
      new AbortController().signal
    )
    await context.load()
    expect(context.scope.serverBase).toBe(window.location.origin)
    expect(requests[0].url).toBe(
      "/api/proxy/users/me/profile?sections=identity"
    )
    expect(
      requests.every(({ init }) => init.credentials === "same-origin")
    ).toBe(true)
    expect(
      requests.every(
        ({ init }) =>
          new Headers(init.headers).get("Authorization") === null &&
          new Headers(init.headers).get("X-API-KEY") === null
      )
    ).toBe(true)
  })

  it("keeps quickstart cookie-session reads on the current origin with no header credentials", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    config.current = {
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session",
      apiKey: "unused",
      orgId: 7
    }
    const context = await createOwnedWorkspaceReadContext(
      id,
      new AbortController().signal
    )
    await context.load()
    expect(context.scope).toEqual({
      serverBase: window.location.origin,
      principalId: "3",
      organizationId: null
    })
    expect(requests[0].url).toBe("/api/v1/users/me/profile?sections=identity")
    expect(
      requests.every(
        ({ url, init }) =>
          init.credentials === "same-origin" &&
          new Headers(init.headers).get("Authorization") === null &&
          new Headers(init.headers).get("X-API-KEY") === null &&
          new Headers(init.headers).get("X-TLDW-Expected-User-ID") ===
            (url.includes("/workspaces/") ? "3" : null)
      )
    ).toBe(true)
  })
})

const scope = {
  serverBase: "https://research.example/install",
  principalId: "3",
  organizationId: "7"
}
const note = {
  id: 11,
  workspace_id: id,
  title: "Canonical note",
  content: "Evidence without a workspace tag",
  keywords_json: '["research"]',
  version: 2,
  created_at: "2026-09-13T12:00:00Z",
  last_modified: "2026-09-13T12:00:00Z"
}

describe("owned workspace scoped notes", () => {
  const context = (signal = new AbortController().signal) =>
    createOwnedWorkspaceNotesContext(id, scope, signal)

  it("lists only canonical associations through the verified connection", async () => {
    responses["/notes"] = [note]
    const notes = await context()
    expect(await notes.list()).toEqual([note])
    expect(requests.at(-1)?.url).toBe(
      `${scope.serverBase}/api/v1/workspaces/${id}/notes`
    )
    expect(
      new Headers(requests.at(-1)?.init.headers).get("X-TLDW-Expected-User-ID")
    ).toBe("3")
    expect(requests.every(({ init }) => init.method === "GET")).toBe(true)
  })

  it("creates in the exact workspace with captured credentials and explicit empty keywords", async () => {
    responses["/notes"] = note
    const notes = await context()
    config.current.serverUrl = "https://other.test"
    config.current.accessToken = "other-token"
    config.current.orgId = 8
    setRuntimeSingleUserApiKeyOverride("other-runtime-key")
    expect(
      await notes.create({ title: "Draft", content: " body ", keywords: [] })
    ).toEqual(note)
    expect(requests.at(-1)).toMatchObject({
      url: `${scope.serverBase}/api/v1/workspaces/${id}/notes`,
      init: {
        method: "POST",
        headers: {
          Authorization: "Bearer test-token-a",
          "X-TLDW-Org-Id": "7",
          "X-TLDW-Expected-User-ID": "3"
        },
        credentials: "omit",
        redirect: "error",
        cache: "no-store"
      }
    })
    expect(JSON.parse(String(requests.at(-1)?.init.body))).toEqual({
      title: "Draft",
      content: " body ",
      keywords: []
    })
  })

  it("updates with a required version in the canonical request body", async () => {
    responses["/notes/11"] = { ...note, version: 3, keywords_json: "[]" }
    const notes = await context()
    const updated = await notes.update(11, {
      title: "Updated",
      content: "",
      keywords_json: "[]",
      version: 2
    })
    expect(updated.version).toBe(3)
    expect(requests.at(-1)?.url).toBe(
      `${scope.serverBase}/api/v1/workspaces/${id}/notes/11`
    )
    expect(requests.at(-1)?.init.method).toBe("PUT")
    expect(JSON.parse(String(requests.at(-1)?.init.body))).toEqual({
      title: "Updated",
      content: "",
      keywords_json: "[]",
      version: 2
    })
  })

  it("snapshots edits and the version before asynchronous dispatch", async () => {
    responses["/notes/11"] = { ...note, version: 3 }
    const notes = await context()
    const body = { content: "Submitted draft", version: 2 }
    const pending = notes.update(11, body)
    body.content = "New unsent draft"
    body.version = 99
    await expect(pending).resolves.toMatchObject({ version: 3 })
    expect(JSON.parse(String(requests.at(-1)?.init.body))).toEqual({
      content: "Submitted draft",
      version: 2
    })
  })

  it("snapshots create keywords before asynchronous dispatch", async () => {
    responses["/notes"] = note
    const notes = await context()
    const body = { content: "Submitted draft", keywords: ["evidence"] }
    const pending = notes.create(body)
    body.content = "New unsent draft"
    body.keywords.push("later")
    await pending
    expect(JSON.parse(String(requests.at(-1)?.init.body))).toEqual({
      content: "Submitted draft",
      keywords: ["evidence"]
    })
  })

  it.each([
    { ...scope, principalId: "4" },
    { ...scope, serverBase: "https://other.test" },
    { ...scope, organizationId: "8" }
  ])(
    "rejects mismatched activation scope %j before a note request",
    async (expected) => {
      await expect(
        createOwnedWorkspaceNotesContext(
          id,
          expected,
          new AbortController().signal
        )
      ).rejects.toMatchObject({ status: 412 })
      expect(requests.every(({ url }) => !url.includes("/workspaces/"))).toBe(
        true
      )
    }
  )

  it.each([401, 403, 409, 412, 500])(
    "preserves failure %i and never retries a write",
    async (status) => {
      const notes = await context()
      failure = { suffix: "/notes/11", status }
      await expect(
        notes.update(11, { content: "local draft", version: 2 })
      ).rejects.toMatchObject({ status })
      expect(requests.filter(({ init }) => init.method === "PUT")).toHaveLength(
        1
      )
    }
  )

  it.each([0, -1, 1.5, NaN, Number.MAX_SAFE_INTEGER + 1])(
    "rejects unsafe note ID %s without dispatch",
    async (noteId) => {
      const notes = await context()
      await expect(notes.update(noteId, { version: 2 })).rejects.toThrow()
      expect(requests).toHaveLength(1)
    }
  )

  it.each([0, -1, 1.5, undefined, Number.MAX_SAFE_INTEGER + 1])(
    "rejects missing/unsafe version %s without dispatch",
    async (version) => {
      const notes = await context()
      await expect(
        notes.update(11, { version: version as number })
      ).rejects.toThrow()
      expect(requests).toHaveLength(1)
    }
  )

  it.each([
    null,
    [{ ...note, workspace_id: "foreign" }],
    [note, note],
    [{ ...note, version: 0 }]
  ])(
    "rejects invalid canonical lists %j instead of substituting empty notes",
    async (value) => {
      const notes = await context()
      responses["/notes"] = value
      await expect(notes.list()).rejects.toMatchObject({
        reason: "invalid-response"
      })
    }
  )

  it.each([
    null,
    { ...note, workspace_id: "foreign" },
    { ...note, id: 12 },
    { ...note, version: 2 }
  ])("rejects an invalid update receipt %j", async (value) => {
    const notes = await context()
    responses["/notes/11"] = value
    await expect(notes.update(11, { version: 2 })).rejects.toMatchObject({
      reason: "invalid-response"
    })
  })

  it("does not dispatch after cancellation", async () => {
    const controller = new AbortController()
    const notes = await context(controller.signal)
    controller.abort()
    await expect(notes.create({ content: "Draft" })).rejects.toMatchObject({
      name: "AbortError"
    })
    expect(requests).toHaveLength(1)
  })

  it("rejects a late response even if fetch ignores cancellation", async () => {
    const controller = new AbortController()
    const notes = await context(controller.signal)
    let complete!: (response: Response) => void
    vi.mocked(fetch).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          complete = resolve
        })
    )
    const pending = notes.create({ content: "Draft" })
    await vi.waitFor(() => expect(complete).toBeTypeOf("function"))
    controller.abort()
    complete(new Response(JSON.stringify(note), { status: 201 }))
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
  })

  it.each(["hosted", "quickstart"])(
    "asserts principal on cookie-authenticated %s writes without header credentials",
    async (mode) => {
      vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", mode)
      config.current = {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      }
      const notes = await createOwnedWorkspaceNotesContext(
        id,
        {
          serverBase: window.location.origin,
          principalId: "3",
          organizationId: null
        },
        new AbortController().signal
      )
      failure = { suffix: "/notes", status: 412 }
      await expect(
        notes.create({ content: "Must remain in account 3" })
      ).rejects.toMatchObject({ status: 412 })
      const init = requests.at(-1)!.init
      expect(init.credentials).toBe("same-origin")
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("3")
      expect(new Headers(init.headers).get("Authorization")).toBeNull()
      expect(new Headers(init.headers).get("X-API-KEY")).toBeNull()
    }
  )
})
