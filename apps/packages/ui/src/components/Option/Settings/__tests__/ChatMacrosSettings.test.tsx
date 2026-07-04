import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  listChatMacros: vi.fn(),
  setChatMacroEnabled: vi.fn(),
  cloneChatMacro: vi.fn(),
  getChatMacroSettings: vi.fn(),
  updateChatMacroSettings: vi.fn(),
  validateChatMacro: vi.fn()
}))

vi.mock("@/services/chat-macros", () => ({
  listChatMacros: (...args: unknown[]) => mocks.listChatMacros(...args),
  setChatMacroEnabled: (...args: unknown[]) => mocks.setChatMacroEnabled(...args),
  cloneChatMacro: (...args: unknown[]) => mocks.cloneChatMacro(...args),
  getChatMacroSettings: (...args: unknown[]) => mocks.getChatMacroSettings(...args),
  updateChatMacroSettings: (...args: unknown[]) => mocks.updateChatMacroSettings(...args),
  validateChatMacro: (...args: unknown[]) => mocks.validateChatMacro(...args)
}))

import { ChatMacrosSettings } from "../ChatMacrosSettings"

const macroListResponse = {
  ok: true,
  status: 200,
  data: {
    macros: [
      {
        name: "wrapup",
        command: "wrapup",
        description: "Summarize the active chat",
        enabled: true,
        source: "builtin",
        immutable: true,
        digest: "digest-1",
        builtin_version: 1,
        schema_version: 1
      }
    ],
    count: 1
  }
}

describe("ChatMacrosSettings", () => {
  beforeEach(() => {
    mocks.listChatMacros.mockReset()
    mocks.setChatMacroEnabled.mockReset()
    mocks.cloneChatMacro.mockReset()
    mocks.getChatMacroSettings.mockReset()
    mocks.updateChatMacroSettings.mockReset()
    mocks.validateChatMacro.mockReset()

    mocks.listChatMacros.mockResolvedValue(macroListResponse)
    mocks.getChatMacroSettings.mockResolvedValue({
      ok: true,
      status: 200,
      data: {
        settings: {
          output_profiles: {
            default: { format: "markdown" }
          }
        }
      }
    })
    mocks.setChatMacroEnabled.mockResolvedValue(macroListResponse)
    mocks.cloneChatMacro.mockResolvedValue({ ok: true, status: 201, data: {} })
    mocks.updateChatMacroSettings.mockResolvedValue({ ok: true, status: 200, data: {} })
    mocks.validateChatMacro.mockResolvedValue({
      ok: true,
      status: 200,
      data: { valid: false, error: "unknown macro argument: nope" }
    })
  })

  it("lists macros and toggles enabled state", async () => {
    render(<ChatMacrosSettings />)

    expect(await screen.findByText("/wrapup")).toBeInTheDocument()
    expect(screen.getByText("builtin")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("switch", { name: "Toggle /wrapup" }))

    await waitFor(() =>
      expect(mocks.setChatMacroEnabled).toHaveBeenCalledWith("wrapup", false)
    )
  })

  it("clones wrapup and saves output profile settings", async () => {
    render(<ChatMacrosSettings />)

    await screen.findByText("/wrapup")
    fireEvent.change(screen.getByLabelText("Clone macro name"), {
      target: { value: "team_wrapup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Clone /wrapup" }))

    await waitFor(() =>
      expect(mocks.cloneChatMacro).toHaveBeenCalledWith("wrapup", {
        name: "team_wrapup",
        command: "team_wrapup"
      })
    )

    fireEvent.change(screen.getByLabelText("Macro settings JSON"), {
      target: { value: '{"output_profiles":{"default":{"format":"plain"}}}' }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save macro settings" }))

    await waitFor(() =>
      expect(mocks.updateChatMacroSettings).toHaveBeenCalledWith({
        output_profiles: { default: { format: "plain" } }
      })
    )
  })

  it("shows backend validation errors inline", async () => {
    render(<ChatMacrosSettings />)

    await screen.findByText("/wrapup")
    fireEvent.change(screen.getByLabelText("Validate macro YAML"), {
      target: { value: "name: wrapup" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Validate macro" }))

    expect(await screen.findByText("unknown macro argument: nope")).toBeInTheDocument()
  })
})
