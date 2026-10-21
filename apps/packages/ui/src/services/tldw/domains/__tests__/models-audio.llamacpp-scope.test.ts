import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ store: {} as Record<string, unknown>, fetch: vi.fn(), send: vi.fn(), wires: 0, runtimeId: null as string | null }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { get id() { return boundary.runtimeId }, sendMessage: (...args: unknown[]) => boundary.send(...args) } } }))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: { serializer: JSON.stringify, deserializer: (value: unknown) => value },
  createSafeStorage: () => ({ get: async (key: string) => boundary.store[key] ?? null,
    set: async (key: string, value: unknown) => { boundary.store[key] = value }, remove: async (key: string) => { delete boundary.store[key] } })
}))
vi.mock("@/services/tldw/runtime-auth-override", async (original) => ({
  ...await original<typeof import("@/services/tldw/runtime-auth-override")>(), getRuntimeSingleUserApiKeyOverride: () => null
}))

import { modelsAudioMethods } from "../models-audio"
const config = (user = 7, serverUrl = "https://c10.example") => ({
  serverUrl, authMode: "multi-user" as const, authSource: "manual" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`, refreshToken: `fixture-refresh-${user}`
})
const calls = [
  ["getLlamacppConfig", []],
  ["getLlamacppStatus", []],
  ["getLlamacppInventory", []],
  ["getLlamacppAssets", []],
  ["listLlamacppProfiles", []],
  ["listLlamacppInstances", []],
  ["startLlamacppProfile", ["analysis"]],
  ["stopLlamacppProfile", ["analysis"]],
  ["useLlamacppProfileInChat", ["analysis"]],
  ["updateLlamacppProfile", ["analysis", { name: "Changed" }]],
  ["deleteLlamacppProfile", ["analysis"]],
  ["startLlamacppModel", ["model-one", undefined]],
  ["stopLlamacppServer", []],
  ["registerLlamacppModelPath", ["/models/fixture.gguf"]]
] as const
beforeEach(() => {
  vi.clearAllMocks(); boundary.runtimeId = null; boundary.store = { tldwConfig: config() }
  boundary.wires = 0
  boundary.fetch.mockImplementation(async (_url: unknown, init: RequestInit) => {
    init.signal?.throwIfAborted()
    boundary.wires += 1
    return new Response(JSON.stringify({ ok: true }), { headers: { "content-type": "application/json" } })
  })
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

describe("managed model captured request authority", () => {
  it.each(calls)("keeps %s on its captured authority after the active account changes", async (name, args) => {
    const options = { configSnapshot: config(), abortSignal: new AbortController().signal }
    boundary.store.tldwConfig = config(8)
    const method = modelsAudioMethods[name] as (...values: unknown[]) => Promise<unknown>
    await expect(method(...args, options)).resolves.toEqual({ ok: true })
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toContain("https://c10.example/api/v1/llamacpp/")
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${config().accessToken}`)
    expect(boundary.wires).toBe(1)
    expect(boundary.send).not.toHaveBeenCalled()
  })

  it("does not send an already-retired runtime start", async () => {
    const controller = new AbortController(); controller.abort()
    const method = modelsAudioMethods.startLlamacppProfile as (...values: unknown[]) => Promise<unknown>
    await expect(method("analysis", { configSnapshot: config(), abortSignal: controller.signal })).rejects.toBeDefined()
    expect(boundary.wires).toBe(0)
    expect(boundary.fetch.mock.calls.every(([, init]) => init.signal.aborted)).toBe(true)
  })
})
