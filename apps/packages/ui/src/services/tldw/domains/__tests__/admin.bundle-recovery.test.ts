import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ store: {} as Record<string, unknown>, fetch: vi.fn(), send: vi.fn(), runtimeId: null as string | null }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { get id() { return boundary.runtimeId }, sendMessage: (...args: unknown[]) => boundary.send(...args) } } }))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: { serializer: JSON.stringify, deserializer: (value: unknown) => value },
  createSafeStorage: () => ({ get: async (key: string) => boundary.store[key] ?? null,
    set: async (key: string, value: unknown) => { boundary.store[key] = value }, remove: async (key: string) => { delete boundary.store[key] } })
}))
vi.mock("@/services/tldw/runtime-auth-override", async (original) => ({
  ...await original<typeof import("@/services/tldw/runtime-auth-override")>(), getRuntimeSingleUserApiKeyOverride: () => null
}))

import { adminMethods } from "../admin"
const config = (user = 7, serverUrl = "https://c09.example") => ({
  serverUrl, authMode: "multi-user" as const, authSource: "manual" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`, refreshToken: `fixture-refresh-${user}`
})
const result = { status: "compatible", validations: [{ dataset: "chacha", compatible: true, message: "ok" }], warnings: [], datasets_restored: [] }
const response = (data = result, status = 200) => new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } })
const file = () => new File([new Uint8Array([80, 75, 3, 4])], "c09.zip", { type: "application/zip" })
const scope = (snapshot = config()) => ({ configSnapshot: { ...snapshot }, abortSignal: new AbortController().signal })

beforeEach(() => {
  vi.clearAllMocks(); boundary.runtimeId = null; boundary.store = { tldwConfig: config() }
  boundary.fetch.mockImplementation(async () => response())
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

describe("Admin bundle actual captured multipart transport", () => {
  it.each([false, true])("preserves one file and query validation with extension=%s", async (extension) => {
    if (extension) boundary.runtimeId = "c09-extension"
    const selected = file()
    await expect(adminMethods.importBundle(selected, { user_id: 7, dry_run: true }, scope())).resolves.toEqual(result)
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe("https://c09.example/api/v1/admin/backups/bundles/import?user_id=7&dry_run=true")
    expect(init.body).toBeInstanceOf(FormData)
    expect([...init.body.keys()]).toEqual(["file"])
    expect(init.body.get("file")).toMatchObject({ name: "c09.zip", size: 4 })
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${config().accessToken}`)
    expect(new Headers(init.headers).has("Content-Type")).toBe(false)
    expect(boundary.send).not.toHaveBeenCalled()
  })

  it("does not admit an already-aborted restore", async () => {
    const captured = scope(); const controller = new AbortController(); controller.abort()
    await expect(adminMethods.importBundle(file(), { user_id: 7, dry_run: false }, { ...captured, abortSignal: controller.signal })).rejects.toBeDefined()
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it("refreshes the same principal and retries the same native multipart body", async () => {
    const captured = scope()
    const refreshed = `fixture.${btoa(JSON.stringify({ sub: "7", rotation: 1 }))}.signature`
    let importCalls = 0
    boundary.fetch.mockImplementation(async (url: string) => {
      if (String(url).includes("/auth/refresh")) return new Response(JSON.stringify({ access_token: refreshed, refresh_token: "fixture-rotated-7" }), { headers: { "content-type": "application/json" } })
      return response(result, ++importCalls === 1 ? 401 : 200)
    })
    await expect(adminMethods.importBundle(file(), { user_id: 7, dry_run: true }, captured)).resolves.toEqual(result)
    const imports = boundary.fetch.mock.calls.filter(([url]) => String(url).includes("/bundles/import?"))
    expect(imports).toHaveLength(2)
    expect(imports[1][1].body).toBe(imports[0][1].body)
    expect(imports[1][1].body).toBeInstanceOf(FormData)
    expect(new Headers(imports[1][1].headers).get("Authorization")).toBe(`Bearer ${refreshed}`)
    expect(boundary.fetch.mock.calls.filter(([url]) => String(url).includes("/auth/refresh"))).toHaveLength(1)
  })

  it.each(["account", "server"])("never refreshes the successor %s after the captured restore receives401", async (changed) => {
    const captured = scope()
    boundary.fetch.mockImplementation(async (url: string) => {
      if (!String(url).includes("/auth/refresh")) {
        boundary.store.tldwConfig = changed === "account" ? config(8) : config(7, "https://other-c09.example")
        return response(result, 401)
      }
      return new Response(JSON.stringify({ access_token: config(8).accessToken, refresh_token: "fixture-rotated" }), { headers: { "content-type": "application/json" } })
    })
    await expect(adminMethods.importBundle(file(), { user_id: 7, dry_run: false }, captured)).rejects.toBeDefined()
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    expect(String(boundary.fetch.mock.calls[0][0])).toContain("https://c09.example/")
    expect(boundary.send).not.toHaveBeenCalled()
  })
})
