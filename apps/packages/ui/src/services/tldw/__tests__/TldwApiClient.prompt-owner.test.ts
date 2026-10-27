import { beforeEach, expect, it, vi } from "vitest"
const mocks = vi.hoisted(() => ({ send: vi.fn(), background: vi.fn() }))
vi.mock("@/services/api-send", () => ({ apiSend: mocks.send }))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.background, bgStream: vi.fn(), bgUpload: vi.fn() }))
import { tldwClient } from "../TldwApiClient"
const owner = "recipe-owner:sha256:" + "a".repeat(64)
beforeEach(() => {
  vi.clearAllMocks()
  vi.spyOn(tldwClient as any, "resolveApiPath").mockResolvedValue("/api/v1/prompts")
  mocks.send.mockResolvedValue({ ok: true, status: 200, data: { id: 1 } })
  mocks.background.mockResolvedValue({ id: 1 })
})
it.each(["search", "create", "update"])("passes captured owner to real %s prompt dispatch", async action => {
  const options = { expectedOwnerId: owner }
  if (action === "search") await tldwClient.searchPrompts("Private", options)
  if (action === "create") await tldwClient.createPrompt({ title: "Private", content: "PRIVATE" }, options)
  if (action === "update") await tldwClient.updatePrompt(1, { title: "Private", content: "PRIVATE" }, options)
  expect(mocks.send).toHaveBeenCalledWith(expect.objectContaining({ recipePersistence: { mode: "owner", expectedOwnerId: owner } }))
  expect(mocks.background).not.toHaveBeenCalled()
})
