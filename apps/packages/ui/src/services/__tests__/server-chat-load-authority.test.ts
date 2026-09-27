import { afterEach, describe, expect, it, vi } from "vitest"
import { watchServerChatLoadAuthority } from "../server-chat-load-authority"
import { tldwClient } from "../tldw/TldwApiClient"
import { activateCookieSessionConfig, invalidateCookieSessionConfig } from "../tldw/runtime-auth-override"
import { COOKIE_SESSION_CONFIG_KEY } from "../tldw/browser-networking"
import { resolveDirectBrowserConfig } from "../tldw/direct-browser-config"
import { invalidateRefreshSessionIfCurrent, storeRefreshRotationIfCurrent, type CredentialStorage } from "../tldw/single-user-credential"
import type { ServicePromptSnapshot } from "../service-prompts"
const token = (sub: string) => `test.${btoa(JSON.stringify({ sub }))}.signature`
const original = { serverUrl: "http://chat.test", authMode: "multi-user" as const, accessToken: token("A"), refreshToken: "synthetic-source-refresh" }
const setup = () => {
  const records = new Map<string, unknown>([["tldwConfig", original]])
  const storage: CredentialStorage = {
    get: async <T>(key: string) => records.get(key) as T | undefined,
    set: async <T>(key: string, value: T) => { records.set(key, value) }, remove: async key => { records.delete(key) }
  }
  vi.spyOn(tldwClient, "getConfig").mockImplementation(() => resolveDirectBrowserConfig(storage))
  const controller = new AbortController()
  const snapshot = { requestScope: { config: { serverUrl: original.serverUrl, authMode: original.authMode }, userId: "A" } } as ServicePromptSnapshot
  const stop = watchServerChatLoadAuthority(snapshot, controller)
  return { storage, records, controller, stop }
}
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllEnvs(); activateCookieSessionConfig() })
describe("pending Chat load canonical generation", () => {
  it("retires actual invalidation markers without changing the raw credential pair", async () => {
    const { storage, records, controller, stop } = setup()
    expect(await invalidateRefreshSessionIfCurrent(storage, original)).toBe(true)
    await vi.waitFor(() => expect(controller.signal.aborted).toBe(true))
    expect(records.get("tldwConfig")).toEqual(original)
    stop()
  })
  it.each(["A", "B", "opaque"])("checks the verified owner after canonical rotation to %s", async subject => {
    const { storage, controller, stop } = setup()
    await storeRefreshRotationIfCurrent(storage, original, original.refreshToken, { accessToken: subject === "opaque" ? "opaque" : token(subject), refreshToken: "synthetic-rotated-refresh" })
    window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { refreshSessionInvalidated: true } }))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(controller.signal.aborted).toBe(subject !== "A")
    stop()
  })
  it.each(["current", "invalidated", "new-target"])("checks quickstart single-user cookie authority: %s", async state => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "quickstart")
    activateCookieSessionConfig()
    const { storage, records, controller: unusedController, stop: stopUnused } = setup()
    stopUnused()
    expect(unusedController.signal.aborted).toBe(false)
    records.delete("tldwConfig")
    const cookieConfig = { serverUrl: window.location.origin, authMode: "single-user", authSource: "cookie-session" }
    await storage.set(COOKIE_SESSION_CONFIG_KEY, cookieConfig)
    const controller = new AbortController()
    const stop = watchServerChatLoadAuthority({ requestScope: { config: cookieConfig, userId: null } } as ServicePromptSnapshot, controller)
    if (state === "invalidated") invalidateCookieSessionConfig()
    if (state === "new-target") await storage.set(COOKIE_SESSION_CONFIG_KEY, { ...cookieConfig, serverUrl: "http://other.test" })
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(controller.signal.aborted).toBe(state !== "current")
    stop()
  })

  it("keeps a later login safe from a released load's delayed check", async () => {
    const { storage, controller, stop } = setup()
    let release!: () => void
    vi.mocked(tldwClient.getConfig).mockImplementationOnce(async () => { await new Promise<void>(resolve => { release = resolve }); return null })
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    stop()
    await storage.set("tldwConfig", { ...original, accessToken: token("B"), refreshToken: "synthetic-B" })
    const replacement = new AbortController()
    const stopReplacement = watchServerChatLoadAuthority({ requestScope: { config: { serverUrl: original.serverUrl, authMode: original.authMode }, userId: "B" } } as ServicePromptSnapshot, replacement)
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
    release()
    await new Promise(resolve => setTimeout(resolve, 0))
    expect(controller.signal.aborted).toBe(false)
    expect(replacement.signal.aborted).toBe(false)
    stopReplacement()
  })
})
