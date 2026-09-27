import { beforeEach, describe, expect, it, vi } from "vitest"

import { RecipePersistenceRegistry } from "../recipe-persistence-registry"
import { tldwRequest } from "../tldw/request-core"

const state = vi.hoisted(() => ({ runtimeKey: "" }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => state.runtimeKey
}))
const config = (serverUrl = "https://a.test", sub = "alice", exp = 1) => ({
  serverUrl,
  authMode: "multi-user",
  accessToken: `h.${btoa(JSON.stringify({ sub, exp }))}.s`,
  refreshToken: "refresh"
})
const payload = {
  path: "/api/v1/prompt-studio/prompts/create" as const,
  method: "POST",
  recipePersistence: {
    mode: "require" as const,
    expectedOwnerId: "recipe-owner:sha256:expected",
    localId: "local-recipe-1"
  },
  body: {}
}
const response = (status: number) =>
  new Response("{}", {
    status,
    headers: { "content-type": "application/json" }
  })

describe("request dispatch scope capture", () => {
  beforeEach(() => {
    state.runtimeKey = ""
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  })
  it("does not dispatch when the expected owner does not match the request snapshot", async () => {
    const otherOwner = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "bob",
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const result = await tldwRequest(
      {
        ...payload,
        recipePersistence: {
          ...payload.recipePersistence,
          expectedOwnerId: otherOwner.recipePersistence!.actualOwnerId!
        }
      },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "alice",
        dispatchAuthority: { markDispatched: vi.fn() },
        fetchFn
      }
    )

    expect(fetchFn).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it("does not dispatch when a trustworthy actual owner is unavailable", async () => {
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const result = await tldwRequest(payload, {
      getConfig: async () => config(),
      getAuthenticatedPrincipal: async () => null,
      dispatchAuthority: { markDispatched: vi.fn() },
      fetchFn
    })

    expect(fetchFn).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it("allows capture-only legacy dispatch with a null actual owner", async () => {
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const result = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => null,
        fetchFn
      }
    )

    expect(fetchFn).toHaveBeenCalledOnce()
    expect(result.recipePersistence).toEqual({
      state: "dispatched",
      actualOwnerId: null
    })
  })

  it("uses the runtime-key snapshot credentials instead of the stale cookie transport", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    state.runtimeKey = "runtime-key"
    const fetchFn = vi.fn().mockResolvedValue(response(200))

    await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => ({
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "stale-key"
        }),
        fetchFn
      }
    )

    const [, init] = fetchFn.mock.calls[0]
    expect(init.credentials).toBeUndefined()
    expect(new Headers(init.headers).get("X-API-KEY")).toBe("runtime-key")
  })

  it("rejects a non-exact local ID before marking or fetching", async () => {
    const capture = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "alice",
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const markDispatched = vi.fn()
    const result = await tldwRequest(
      {
        ...payload,
        recipePersistence: {
          mode: "require",
          expectedOwnerId: capture.recipePersistence!.actualOwnerId!,
          localId: " local-recipe-1 "
        }
      },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "alice",
        dispatchAuthority: { markDispatched },
        fetchFn
      }
    )

    expect(markDispatched).not.toHaveBeenCalled()
    expect(fetchFn).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it("rejects when deferred configuration resolves to a changed owner", async () => {
    const initialOwner = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "alice",
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    let release!: (value: ReturnType<typeof config>) => void
    const configPromise = new Promise<ReturnType<typeof config>>((resolve) => {
      release = resolve
    })
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const pending = tldwRequest(
      {
        ...payload,
        recipePersistence: {
          ...payload.recipePersistence,
          expectedOwnerId: initialOwner.recipePersistence!.actualOwnerId!
        }
      },
      {
        getConfig: async () => await configPromise,
        getAuthenticatedPrincipal: async () => "bob",
        dispatchAuthority: { markDispatched: vi.fn() },
        fetchFn
      }
    )

    release(config("https://a.test", "bob"))
    const result = await pending

    expect(fetchFn).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it("does not bind a changed principal to credentials captured before principal resolution", async () => {
    const alice = config("https://a.test", "alice")
    const bob = config("https://a.test", "bob")
    const bobCapture = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => bob,
        getAuthenticatedPrincipal: async () => "bob",
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    let activeConfig = alice
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const markDispatched = vi.fn()

    const result = await tldwRequest(
      {
        ...payload,
        recipePersistence: {
          mode: "require",
          expectedOwnerId: bobCapture.recipePersistence!.actualOwnerId!,
          localId: "local-recipe-1"
        }
      },
      {
        getConfig: async () => ({ ...activeConfig }),
        getAuthenticatedPrincipal: async (snapshot?: {
          headers: Readonly<Record<string, string>>
        }) => {
          activeConfig = bob
          return snapshot?.headers.Authorization ===
            `Bearer ${alice.accessToken}`
            ? "alice"
            : "bob"
        },
        dispatchAuthority: { markDispatched },
        fetchFn
      }
    )

    expect(markDispatched).not.toHaveBeenCalled()
    expect(fetchFn).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it("marks the exact local ID once immediately before the request and keeps the marker after success", async () => {
    const events: string[] = []
    const fetchFn = vi.fn().mockImplementation(async () => {
      events.push("fetch")
      return response(200)
    })
    const capture = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "alice",
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    const actualOwnerId = capture.recipePersistence?.actualOwnerId
    expect(actualOwnerId).toEqual(expect.any(String))
    const markDispatched = vi.fn(async (localId: string, ownerId: string) => {
      events.push(`mark:${localId}:${ownerId}`)
    })

    const result = await tldwRequest(
      {
        ...payload,
        recipePersistence: {
          mode: "require",
          expectedOwnerId: actualOwnerId!,
          localId: "local-recipe-1"
        }
      },
      {
        getConfig: async () => config(),
        getAuthenticatedPrincipal: async () => "alice",
        dispatchAuthority: { markDispatched },
        fetchFn
      }
    )

    expect(events).toEqual([`mark:local-recipe-1:${actualOwnerId}`, "fetch"])
    expect(markDispatched).toHaveBeenCalledOnce()
    const serializedHttpRequest = JSON.stringify({
      headers: fetchFn.mock.calls[0][1].headers,
      body: fetchFn.mock.calls[0][1].body
    })
    expect(serializedHttpRequest).not.toContain("local-recipe-1")
    expect(serializedHttpRequest).not.toContain(actualOwnerId!)
    expect(serializedHttpRequest).not.toContain("recipePersistence")
    expect(result.recipePersistence).toEqual({
      state: "dispatched",
      actualOwnerId
    })
  })

  it("does not fetch when the dispatch authority rejects the marker", async () => {
    const fetchFn = vi.fn().mockResolvedValue(response(200))
    const capture = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => ({
          serverUrl: "https://a.test",
          authMode: "single-user",
          authSource: "manual",
          apiKey: "key"
        }),
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    const result = await tldwRequest(
      {
        ...payload,
        recipePersistence: {
          mode: "require",
          expectedOwnerId: capture.recipePersistence!.actualOwnerId!,
          localId: "local-recipe-1"
        }
      },
      {
        getConfig: async () => ({
          serverUrl: "https://a.test",
          authMode: "single-user",
          authSource: "manual",
          apiKey: "key"
        }),
        dispatchAuthority: {
          markDispatched: vi.fn().mockRejectedValue(new Error("storage failed"))
        },
        fetchFn
      }
    )

    expect(fetchFn).not.toHaveBeenCalled()
    expect(result.recipePersistence).toEqual({
      state: "not_dispatched",
      actualOwnerId: null
    })
  })

  it("retains one owner and one marker during same-subject token rotation", async () => {
    const initial = config()
    const updated = config("https://a.test", "alice", 2)
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(response(401))
      .mockResolvedValueOnce(response(200))
    const capture = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => initial,
        getAuthenticatedPrincipal: async () => "alice",
        fetchFn: vi.fn().mockResolvedValue(response(200))
      }
    )
    const registry = new RecipePersistenceRegistry()
    const markDispatched = vi.fn((id: string, ownerId: string) =>
      registry.reserve(id, ownerId)
    )
    const result = await tldwRequest(
      {
        ...payload,
        recipePersistence: {
          mode: "require",
          expectedOwnerId: capture.recipePersistence!.actualOwnerId!,
          localId: "local-recipe-1"
        }
      },
      {
        getConfig: vi
          .fn()
          .mockResolvedValueOnce(initial)
          .mockResolvedValue(updated),
        getAuthenticatedPrincipal: async () => "alice",
        dispatchAuthority: { markDispatched },
        fetchFn,
        refreshAuth: async () => {}
      }
    )
    expect(result.ok).toBe(true)
    expect(result.recipePersistence).toEqual({
      state: "dispatched",
      actualOwnerId: capture.recipePersistence!.actualOwnerId
    })
    expect(fetchFn).toHaveBeenCalledTimes(2)
    expect(markDispatched).toHaveBeenCalledOnce()
    expect(fetchFn.mock.calls[1][1].headers.Authorization).toBe(
      `Bearer ${updated.accessToken}`
    )
  })

  it("does not retry an unowned captured write after bearer and organization change", async () => {
    const initial = { ...config(), orgId: "org-a" }
    const updated = { ...config("https://a.test", "bob", 2), orgId: "org-b" }
    const fetchFn = vi
      .fn()
      .mockResolvedValueOnce(response(401))
      .mockResolvedValueOnce(response(200))

    const result = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: vi
          .fn()
          .mockResolvedValueOnce(initial)
          .mockResolvedValue(updated),
        getAuthenticatedPrincipal: async () => null,
        fetchFn,
        refreshAuth: async () => {}
      }
    )

    expect(fetchFn).toHaveBeenCalledTimes(1)
    expect(result).toMatchObject({ ok: false, status: 412 })
    expect(result.recipePersistence).toEqual({
      state: "dispatched",
      actualOwnerId: null
    })
  })

  it.each([
    ["source", { authSource: "other" }, "alice"],
    ["base", { serverUrl: "https://b.test" }, "alice"],
    ["organization", { orgId: "other" }, "alice"],
    ["principal", {}, "bob"]
  ])(
    "does not retry after a changed %s",
    async (_label, changed, principal) => {
      const initial = config()
      const capture = await tldwRequest(
        { ...payload, recipePersistence: { mode: "capture" } },
        {
          getConfig: async () => initial,
          getAuthenticatedPrincipal: async () => "alice",
          fetchFn: vi.fn().mockResolvedValue(response(200))
        }
      )
      const fetchFn = vi.fn().mockResolvedValue(response(401))
      const getConfig = vi
        .fn()
        .mockResolvedValueOnce(initial)
        .mockResolvedValue({ ...initial, ...changed })
      const getAuthenticatedPrincipal = vi
        .fn()
        .mockResolvedValueOnce("alice")
        .mockResolvedValue(principal)
      const result = await tldwRequest(
        {
          ...payload,
          recipePersistence: {
            mode: "require",
            expectedOwnerId: capture.recipePersistence!.actualOwnerId!,
            localId: "local-recipe-1"
          }
        },
        {
          getConfig,
          getAuthenticatedPrincipal,
          dispatchAuthority: { markDispatched: vi.fn() },
          fetchFn,
          refreshAuth: async () => {}
        }
      )

      expect(fetchFn).toHaveBeenCalledTimes(1)
      expect(result).toMatchObject({ ok: false, status: 412 })
      expect(result.recipePersistence?.state).toBe("dispatched")
    }
  )

  it("overlays transport metadata after parsing a spoofing response body", async () => {
    const result = await tldwRequest(
      { ...payload, recipePersistence: { mode: "capture" } },
      {
        getConfig: async () => ({
          serverUrl: "https://a.test",
          authMode: "single-user",
          authSource: "manual",
          apiKey: "key"
        }),
        fetchFn: vi.fn().mockResolvedValue(
          new Response(
            JSON.stringify({
              recipePersistence: {
                state: "not_dispatched",
                actualOwnerId: null
              }
            }),
            { status: 200, headers: { "content-type": "application/json" } }
          )
        )
      }
    )

    expect(result.recipePersistence).toMatchObject({
      state: "dispatched",
      actualOwnerId: expect.any(String)
    })
  })
})
