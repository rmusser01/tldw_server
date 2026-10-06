import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"
import { TldwApiClient } from "../TldwApiClient"
import { clearRuntimeAuthOverride, setRuntimeSingleUserApiKeyOverride, activateCookieSessionConfig, invalidateCookieSessionConfig } from "../runtime-auth-override"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn(), bgStream: vi.fn(), bgUpload: vi.fn() }))

let client: TldwApiClient
beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  clearRuntimeAuthOverride()
  activateCookieSessionConfig()
  vi.stubEnv("NEXT_PUBLIC_API_URL", "https://separate-api.example.test")
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "self_host")
  for (const key of ["NEXT_PUBLIC_X_API_KEY", "NEXT_PUBLIC_API_BEARER", "VITE_TLDW_API_KEY", "VITE_TLDW_DEFAULT_API_KEY"]) vi.stubEnv(key, "")
  client = new TldwApiClient()
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllEnvs(); clearRuntimeAuthOverride(); activateCookieSessionConfig() })

it("proves raw absence without initialization or writes", async () => {
  const initialize = vi.spyOn(client, "initialize")
  const set = vi.spyOn(Storage.prototype, "set")
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(true)
  expect(initialize).not.toHaveBeenCalled()
  expect(set).not.toHaveBeenCalled()
})

it.each([
  ["local", "tldwConfig", {}], ["local", "tldwConfig", "broken-json"], ["local", "tldwConfig", null],
  ["sync", "tldwConfig", { serverUrl: "https://old.example.test" }],
  ["local", "tldwCookieSessionConfig", {}], ["session", "tldwManualSessionApiKey", {}],
  ["local", "tldwRefreshRotation", {}], ["local", "tldwRuntimeSessionSingleUserApiKey", "legacy-key"],
  ["local", "apiKey", "legacy-key"], ["local", "accessToken", "legacy-token"],
  ["local", "access_token", "legacy-login"], ["local", "tldwRuntimeAuthMetadata", {}]
] as const)("rejects prior or malformed %s %s authority", async (area, key, value) => {
  await new Storage({ area }).set(key, value)
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(false)
})

it.each(["local", "sync", "session"] as const)("rejects a failed %s read even when getConfig returns null", async area => {
  const internal = client as unknown as { storage: Storage; sessionStorage: Storage }
  const store = area === "sync" ? null : internal[area === "local" ? "storage" : "sessionStorage"]
  const get = Storage.prototype.get
  vi.spyOn(Storage.prototype, "get").mockImplementation(async function<T>(this: Storage, key: string) {
    if (store ? this === store : this !== internal.storage && this !== internal.sessionStorage) throw new Error("storage unavailable")
    return await get.call(this, key) as T | undefined
  })
  expect(await client.getConfig()).toBeNull()
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(false)
})

it.each(["NEXT_PUBLIC_X_API_KEY", "NEXT_PUBLIC_API_BEARER", "VITE_TLDW_API_KEY", "VITE_TLDW_DEFAULT_API_KEY"])("rejects %s authority", async key => {
  vi.stubEnv(key, "configured-synthetic-authority")
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(false)
})

it("rejects runtime override and cached configuration", async () => {
  setRuntimeSingleUserApiKeyOverride("runtime-key")
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(false)
  clearRuntimeAuthOverride()
  ;(client as unknown as { config: unknown }).config = { serverUrl: "https://old.example.test", authMode: "single-user" }
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(false)
})

it("rejects same-origin quickstart where cookie authority is not observable", async () => {
  vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(false)
})

it("rechecks raw authority after an awaited read", async () => {
  const storage = (client as unknown as { storage: Storage }).storage
  const original = storage.get.bind(storage)
  let release!: () => void
  let entered!: () => void
  const began = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<void>(resolve => { release = resolve })
  vi.spyOn(storage, "get").mockImplementationOnce(async key => {
    const value = await original(key)
    entered()
    await held
    return value
  })
  const pending = client.hasNoWebUiConnectionConfiguration()
  await began
  localStorage.setItem("tldwConfig", "null")
  release()
  expect(await pending).toBe(false)
})

it("accepts URL hints and the initial bootstrap cookie disable without granting them ownership", async () => {
  localStorage.setItem("tldw-api-host", "https://initial-api.example.test")
  await new Storage({ area: "local" }).set("tldwServerUrl", "https://initial-api.example.test")
  invalidateCookieSessionConfig()
  expect(await client.hasNoWebUiConnectionConfiguration()).toBe(true)
})


it("keeps readback and initial publication inside the participating credential lock", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(navigator, "locks")
  let tail = Promise.resolve<unknown>(undefined)
  let lockHeld = false
  Object.defineProperty(navigator, "locks", { configurable: true, value: { request: (_name: string, work: () => unknown) => {
    const result = tail.then(async () => { lockHeld = true; try { return await work() } finally { lockHeld = false } })
    tail = result.catch(() => undefined)
    return result
  } } })
  let release!: () => void
  let entered!: () => void
  const held = new Promise<void>(done => { release = done })
  const began = new Promise<void>(done => { entered = done })
  let written = false
  let readbackHeld = false
  const set = Storage.prototype.set
  const get = Storage.prototype.get
  const order: string[] = []
  vi.spyOn(Storage.prototype, "set").mockImplementation(async function (key, value) {
    await set.call(this, key, value)
    if (key === "tldwConfig" && (value as { apiKey?: string })?.apiKey === "synthetic-first") written = true
    if (key === "tldwConfig" && (value as { apiKey?: string })?.apiKey === "synthetic-second") order.push("competing-write")
  })
  vi.spyOn(Storage.prototype, "get").mockImplementation(async function<T>(key: string) {
    const value = await get.call(this, key)
    if (written && !readbackHeld && key === "tldwConfig") {
      readbackHeld = true
      entered()
      await held
    }
    return value as T | undefined
  })
  const initialSetup = {
    token: {}, assertCurrent: () => {}, admit: () => {}, cancel: () => {}, complete: async () => true,
    afterBoundary: () => { order.push(lockHeld ? "initial-publication-under-lock" : "initial-publication-outside-lock") }
  }
  try {
    const first = client.saveManualSingleUserCredential({ serverUrl: "https://first.example.test", apiKey: "synthetic-first", persistence: "device", initialSetup })
    await began
    const second = client.saveManualSingleUserCredential({ serverUrl: "https://second.example.test", apiKey: "synthetic-second", persistence: "device" })
    release()
    await Promise.allSettled([first, second])
    expect(order).toEqual(["initial-publication-under-lock", "competing-write"])
    expect((await client.getConfig())?.serverUrl).toBe("https://second.example.test")
  } finally {
    release()
    if (descriptor) Object.defineProperty(navigator, "locks", descriptor)
    else Reflect.deleteProperty(navigator, "locks")
  }
})
