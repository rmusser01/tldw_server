import { beforeEach, expect, it, vi } from "vitest"
import { resolvePromptOwnerId } from "../prompt-local-owner"
const state = vi.hoisted(() => ({ config: null as any }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({ resolveRecipePersistenceOwnerView: async () => null }))
vi.mock("@/services/tldw/direct-browser-config", () => ({ resolveDirectBrowserConfig: async () => state.config }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({}) }))
beforeEach(() => { state.config = null })
it.each([{ authMode: "multi-user", accessToken: "token" }, { authMode: "single-user", authSource: "cookie-session" }])("refuses authenticated local writes when the account cannot resolve %j", async config => {
  state.config = config
  await expect(resolvePromptOwnerId(null)).rejects.toThrow(/unavailable/i)
})
it("retains explicitly unconfigured offline local ownership", async () => {
  await expect(resolvePromptOwnerId(null)).resolves.toBeNull()
})
