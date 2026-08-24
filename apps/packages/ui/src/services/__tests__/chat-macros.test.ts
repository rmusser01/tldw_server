import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  apiSend: vi.fn()
}))

vi.mock("@/services/api-send", () => ({
  apiSend: (...args: unknown[]) => mocks.apiSend(...args)
}))

import {
  cancelChatMacroRun,
  cloneChatMacro,
  getChatMacroRun,
  listChatMacros,
  setChatMacroEnabled,
  updateChatMacroSettings,
  validateChatMacro
} from "@/services/chat-macros"

describe("chat macros service", () => {
  beforeEach(() => {
    mocks.apiSend.mockReset()
    mocks.apiSend.mockResolvedValue({ ok: true, status: 200, data: {} })
  })

  it("lists chat macros through the REST API", async () => {
    await listChatMacros()

    expect(mocks.apiSend).toHaveBeenCalledWith({
      path: "/api/v1/chat/macros",
      method: "GET"
    })
  })

  it("cancels macro runs through the run cancel endpoint", async () => {
    await cancelChatMacroRun("run-1")

    expect(mocks.apiSend).toHaveBeenCalledWith({
      path: "/api/v1/chat/macros/runs/run-1/cancel",
      method: "POST"
    })
  })

  it("updates macro enabled state through the macro update endpoint", async () => {
    await setChatMacroEnabled("wrapup", false)

    expect(mocks.apiSend).toHaveBeenCalledWith({
      path: "/api/v1/chat/macros/wrapup",
      method: "PUT",
      body: expect.objectContaining({ enabled: false })
    })
  })

  it("encodes macro and run identifiers in endpoint paths", async () => {
    await cloneChatMacro("wrapup", { name: "team wrapup", command: "team wrapup" })
    await getChatMacroRun("run/with slash")

    expect(mocks.apiSend).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/chat/macros/wrapup/clone",
        method: "POST",
        body: { name: "team wrapup", command: "team wrapup" }
      })
    )
    expect(mocks.apiSend).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/chat/macros/runs/run%2Fwith%20slash",
        method: "GET"
      })
    )
  })

  it("saves settings and validates macro YAML", async () => {
    await updateChatMacroSettings({ output_profiles: { default: { format: "markdown" } } })
    await validateChatMacro("name: wrapup")

    expect(mocks.apiSend).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/chat/macros/settings",
        method: "PUT",
        body: { settings: { output_profiles: { default: { format: "markdown" } } } }
      })
    )
    expect(mocks.apiSend).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/chat/macros/validate",
        method: "POST",
        body: { raw: "name: wrapup" }
      })
    )
  })
})
