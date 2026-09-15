import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
vi.mock("@plasmohq/storage", async () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn(async () => ({})), bgUpload: vi.fn() }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
import { TldwApiClient, tldwClient, type TldwConfig } from "../tldw/TldwApiClient"
import { TldwAuthService } from "../tldw/TldwAuth"
import { createFlashcardsGenerateHandoff, FLASHCARDS_GENERATE_HANDOFF_PREFIX } from "../tldw/flashcards-generate-handoff"

const tokenFor = (id: number, suffix = "original") => `test.${btoa(JSON.stringify({ sub: String(id) }))}.${suffix}`
const config: TldwConfig = { serverUrl: "https://source.test", authMode: "multi-user", accessToken: tokenFor(1) }
const record = async () => {
  const token = await createFlashcardsGenerateHandoff({ text: "Private source" }, "verified-owner")
  return FLASHCARDS_GENERATE_HANDOFF_PREFIX + token
}
describe("existing auth cleanup owns private Flashcards transfers", () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.localStorage.setItem("tldwConfig", JSON.stringify(config))
    let tail = Promise.resolve()
    const locks = { request: (_name: string, work: () => unknown) => { const next = tail.then(work); tail = next.then(() => undefined, () => undefined); return next } }
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: locks } }))
  })
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it.each([
    config,
    { serverUrl: config.serverUrl, authMode: "single-user", authSource: "cookie-session" },
    { serverUrl: config.serverUrl, authMode: "single-user", apiKey: "synthetic" }
  ])("clears transfers through actual logout for $authMode $authSource", async authConfig => {
    vi.spyOn(tldwClient, "getConfig").mockResolvedValue(authConfig as TldwConfig)
    vi.spyOn(tldwClient, "updateConfig").mockResolvedValue(undefined)
    vi.spyOn(tldwClient, "clearCookieSingleUserSession").mockResolvedValue(undefined)
    vi.spyOn(tldwClient, "clearManualSingleUserCredentials").mockResolvedValue(undefined)
    const key = await record()
    window.localStorage.setItem("other-transfer", "preserve")
    await new TldwAuthService().logout()
    await vi.waitFor(() => expect(window.localStorage.getItem(key)).toBeNull())
    expect(window.localStorage.getItem("other-transfer")).toBe("preserve")
  })

  it.each([2, 1])("actual config update clears changed owner and preserves known-user rotation (%s)", async user => {
    const client = new TldwApiClient()
    vi.spyOn(client, "initialize").mockResolvedValue(undefined)
    vi.spyOn(client, "getConfig").mockResolvedValue(config)
    const key = await record()
    await client.updateConfig({ accessToken: tokenFor(user, "rotated") })
    if (user === 2) await vi.waitFor(() => expect(window.localStorage.getItem(key)).toBeNull())
    else expect(window.localStorage.getItem(key)).not.toBeNull()
  })
})
