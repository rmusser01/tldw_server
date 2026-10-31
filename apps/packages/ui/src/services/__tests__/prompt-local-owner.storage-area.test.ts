import { afterEach, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"
import { resolvePromptOwnerId } from "../prompt-local-owner"
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: async () => null }))
const area = (values: Record<string, unknown>) => ({
  get: async (keys: any, callback?: any) => {
    const wanted = keys == null ? Object.keys(values) : typeof keys === "string" ? [keys] : keys
    const result = Object.fromEntries(wanted.filter((key: string) => key in values).map((key: string) => [key, values[key]]))
    callback?.(result); return result
  },
  set: async (next: any) => { Object.assign(values, next) }
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.restoreAllMocks(); localStorage.clear(); sessionStorage.clear() })
it("refuses unavailable authenticated ownership when canonical credentials are only in local storage", async () => {
  localStorage.clear(); sessionStorage.clear(); vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "")
  vi.stubGlobal("chrome", { storage: {
    local: area({ tldwConfig: { serverUrl: "https://server.test", authMode: "multi-user", accessToken: "known-authenticated" } }),
    sync: area({}), session: area({})
  } })
  // Both configured clients use their actual local-area storage adapter.
  await new Storage({ area: "local" }).set("tldwConfig", { serverUrl: "https://server.test", authMode: "multi-user", accessToken: "known-authenticated" })
  await expect(resolvePromptOwnerId(null)).rejects.toThrow(/unavailable/)
})
it("does not interpret unreadable local credentials as explicit offline absence", async () => {
  localStorage.clear(); sessionStorage.clear()
  vi.stubGlobal("chrome", { storage: {
    local: { get: async () => { throw new Error("storage unavailable") } },
    sync: area({}), session: area({})
  } })
  vi.spyOn(globalThis.Storage.prototype, "getItem").mockImplementation(() => { throw new Error("storage unavailable") })
  await expect(resolvePromptOwnerId(null)).rejects.toThrow(/unavailable/)
})
