import { afterEach, expect, it, vi } from "vitest"
import { tldwRequest } from "@/services/tldw/request-core"
import { resolveRecipeRequestSnapshot } from "@/services/tldw/recipe-request-snapshot"
const config = { serverUrl: "http://localhost:3000", authMode: "single-user", authSource: "manual", apiKey: "alice-private-key" }
const ownerFor = (value: typeof config) => resolveRecipeRequestSnapshot({ config: value, path: "/api/v1/prompt-studio/prompts/create", method: "POST" }).view!.ownerId
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })
it("binds a cookie-session read to the principal captured before a cross-tab switch", async () => {
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  const cookieConfig = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
  const expected = resolveRecipeRequestSnapshot({ config: cookieConfig, path: "/api/v1/prompt-studio/prompts/get/1", method: "GET", pageOrigin: window.location.origin, cookieSessionTransport: true, authenticatedPrincipalId: "2" }).view!.ownerId
  let currentCookieUser = "2"
  const fetchFn = vi.fn(async (_url, options) => {
    const asserted = new Headers(options?.headers).get("X-TLDW-Expected-User-ID")
    return new Response(JSON.stringify({ owner: currentCookieUser }), { status: asserted && asserted !== currentCookieUser ? 412 : 200 })
  })
  const result = await tldwRequest({ path: "/api/v1/prompt-studio/prompts/get/1", method: "GET", recipePersistence: { mode: "owner", expectedOwnerId: expected } } as any,
    { getConfig: async () => cookieConfig, fetchFn, useRuntimeAuthOverride: false, getAuthenticatedPrincipal: async () => { currentCookieUser = "3"; return "2" } })
  expect(result.status).toBe(412)
  expect(new Headers(fetchFn.mock.calls[0][1].headers).get("X-TLDW-Expected-User-ID")).toBe("2")
})
it("refuses captured Alice content when actual dispatch authority is Bob", async () => {
  const fetchFn = vi.fn(async () => new Response("{}", { status: 200 }))
  const result = await tldwRequest({ path: "/api/v1/prompt-studio/prompts/create", method: "POST", body: { name: "ALICE PRIVATE" }, recipePersistence: { mode: "owner", expectedOwnerId: ownerFor(config) } } as any,
    { getConfig: async () => ({ ...config, apiKey: "bob-key" }), fetchFn, useRuntimeAuthOverride: false })
  expect(result.status).toBe(412)
  expect(fetchFn).not.toHaveBeenCalled()
})
it("dispatches same-owner V1 data without reserving a recipe delivery lease", async () => {
  const markDispatched = vi.fn()
  const fetchFn = vi.fn(async () => new Response("{}", { status: 200 }))
  const result = await tldwRequest({ path: "/api/v1/prompt-studio/prompts/create", method: "POST", body: { name: "Owned" }, recipePersistence: { mode: "owner", expectedOwnerId: ownerFor(config) } } as any,
    { getConfig: async () => config, fetchFn, useRuntimeAuthOverride: false, dispatchAuthority: { markDispatched } })
  expect(result.ok).toBe(true)
  expect(result.recipePersistence).toEqual({ state: "dispatched", actualOwnerId: ownerFor(config) })
  expect(markDispatched).not.toHaveBeenCalled()
})
