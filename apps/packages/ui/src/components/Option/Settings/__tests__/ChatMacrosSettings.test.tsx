import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type {
  ChatMacroDetail,
  ChatMacroSettings,
  ChatMacroSummary
} from "@/services/chat-macros"

const mocks = vi.hoisted(() => ({
  cloneChatMacro: vi.fn(),
  createChatMacro: vi.fn(),
  deleteChatMacro: vi.fn(),
  getChatMacro: vi.fn(),
  getChatMacroSettings: vi.fn(),
  listChatMacros: vi.fn(),
  setChatMacroEnabled: vi.fn(),
  updateChatMacro: vi.fn(),
  updateChatMacroSettings: vi.fn(),
  validateChatMacro: vi.fn(),
  confirmDanger: vi.fn()
}))

vi.mock("@/services/chat-macros", () => ({
  cloneChatMacro: mocks.cloneChatMacro,
  createChatMacro: mocks.createChatMacro,
  deleteChatMacro: mocks.deleteChatMacro,
  getChatMacro: mocks.getChatMacro,
  getChatMacroSettings: mocks.getChatMacroSettings,
  listChatMacros: mocks.listChatMacros,
  setChatMacroEnabled: mocks.setChatMacroEnabled,
  updateChatMacro: mocks.updateChatMacro,
  updateChatMacroSettings: mocks.updateChatMacroSettings,
  validateChatMacro: mocks.validateChatMacro
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

import { ChatMacrosSettings } from "../ChatMacrosSettings"

const success = <T,>(data: T) => ({ ok: true, status: 200, data })

const deferred = <T,>() => {
  let resolve: (value: T) => void
  let reject: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve: resolve!, reject: reject! }
}

const makeMacro = (overrides: Partial<ChatMacroSummary> = {}): ChatMacroSummary => ({
  name: "wrapup",
  command: "wrapup",
  description: "Summarize the active chat",
  enabled: true,
  source: "builtin",
  immutable: true,
  digest: "digest-wrapup",
  builtin_version: 1,
  schema_version: 1,
  validation_status: "valid",
  validation_error: null,
  ...overrides
})

const builtinMacro = makeMacro()
const userMacro = makeMacro({
  name: "research",
  command: "research",
  description: "Collect evidence",
  source: "user",
  immutable: false,
  digest: "digest-research"
})

const macroListResponse = (macros: ChatMacroSummary[]) =>
  success({ macros, count: macros.length })

const makeSettings = (overrides: Partial<ChatMacroSettings> = {}): ChatMacroSettings => ({
  disabled_builtins: [],
  user_macro_enabled: { research: true },
  output_profiles: {
    default: {
      format: "structured_sections",
      sections: ["summary"],
      section_titles: {},
      include_branch_outputs: false
    }
  },
  unrelated_setting: { preserve: true },
  ...overrides
})

const makeDetail = (summary: ChatMacroSummary): ChatMacroDetail => ({
  summary,
  definition: {
    schema_version: 1,
    name: summary.name,
    command: summary.command,
    description: summary.description || "",
    enabled: summary.enabled,
    args: {},
    context: {},
    execution: {},
    steps: [],
    output_profile: "default",
    permissions: { tool_calls: [], skills: [] }
  },
  raw: [
    "schema_version: 1",
    `name: ${summary.name}`,
    `command: ${summary.command}`,
    `description: ${summary.description || ""}`
  ].join("\n"),
  supporting_files: {}
})

describe("ChatMacrosSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.listChatMacros.mockResolvedValue(macroListResponse([builtinMacro, userMacro]))
    mocks.getChatMacroSettings.mockResolvedValue(success({ settings: makeSettings() }))
    mocks.getChatMacro.mockImplementation((name: string) => {
      const macro = [builtinMacro, userMacro].find((candidate) => candidate.name === name) || userMacro
      return Promise.resolve(success(makeDetail(macro)))
    })
    mocks.setChatMacroEnabled.mockResolvedValue(success(makeDetail(builtinMacro)))
    mocks.cloneChatMacro.mockResolvedValue(success(makeDetail(userMacro)))
    mocks.createChatMacro.mockResolvedValue(success(makeDetail(userMacro)))
    mocks.updateChatMacro.mockResolvedValue(success(makeDetail(userMacro)))
    mocks.deleteChatMacro.mockResolvedValue(success(undefined))
    mocks.updateChatMacroSettings.mockImplementation((settings: ChatMacroSettings) =>
      Promise.resolve(success({ settings }))
    )
    mocks.validateChatMacro.mockResolvedValue(success({ valid: true, macro: { name: "handoff" } }))
    mocks.confirmDanger.mockResolvedValue(true)
  })

  it("lists macros with visible source, enabled, and validation state and preserves toggle behavior", async () => {
    const user = userEvent.setup()
    render(<ChatMacrosSettings />)

    expect(await screen.findByRole("button", { name: "Select /wrapup" })).toBeInTheDocument()
    expect(screen.getByText("builtin")).toBeInTheDocument()
    expect(screen.getAllByText("Valid")).toHaveLength(2)

    await user.click(screen.getByRole("switch", { name: "Toggle /wrapup" }))

    await waitFor(() =>
      expect(mocks.setChatMacroEnabled).toHaveBeenCalledWith("wrapup", false)
    )
  })

  it("renders returned catalog validation metadata instead of deriving it", async () => {
    const invalidMacro = makeMacro({
      name: "stale_research",
      command: "stale_research",
      source: "user",
      immutable: false,
      validation_status: "invalid",
      validation_error: "Output profile is unavailable."
    })
    mocks.listChatMacros.mockResolvedValueOnce(macroListResponse([builtinMacro, invalidMacro]))

    render(<ChatMacrosSettings />)

    expect(await screen.findByText("Valid")).toBeInTheDocument()
    expect(screen.getByText("Invalid")).toHaveAttribute("title", "Output profile is unavailable.")
  })

  it("imports through the manager input into a fresh draft without persisting and accepts the same file twice", async () => {
    const user = userEvent.setup()
    render(<ChatMacrosSettings />)

    await user.click(await screen.findByRole("button", { name: "Select /research" }))
    expect(await screen.findByLabelText("Name")).toHaveValue("research")

    const upload = screen.getByLabelText("Import macro YAML file") as HTMLInputElement
    const click = vi.spyOn(upload, "click")
    await user.click(screen.getByRole("button", { name: "Import macro" }))
    expect(click).toHaveBeenCalledTimes(1)

    const file = new File(["name: imported"], "imported.yaml", { type: "text/yaml" })
    await user.upload(upload, file)

    expect(await screen.findByLabelText("Macro YAML")).toHaveValue("name: imported")
    expect(screen.getByLabelText("Name")).toHaveValue("imported")
    expect(screen.getByLabelText("Name")).not.toHaveAttribute("readonly")
    expect(mocks.createChatMacro).not.toHaveBeenCalled()
    expect(mocks.updateChatMacro).not.toHaveBeenCalled()

    await user.type(screen.getByLabelText("Macro YAML"), "\nchanged")
    await user.click(screen.getByRole("button", { name: "Import macro" }))
    await user.upload(upload, file)

    expect(await screen.findByLabelText("Macro YAML")).toHaveValue("name: imported")
    expect(mocks.createChatMacro).not.toHaveBeenCalled()
    expect(mocks.updateChatMacro).not.toHaveBeenCalled()
  })

  it("loads a selected user macro and exposes only disable and clone actions for a selected built-in", async () => {
    const user = userEvent.setup()
    render(<ChatMacrosSettings />)

    await user.click(await screen.findByRole("button", { name: "Select /research" }))
    expect(await screen.findByLabelText("Name")).toHaveValue("research")
    expect(mocks.getChatMacro).toHaveBeenCalledWith("research")

    await user.click(screen.getByRole("button", { name: "Select /wrapup" }))
    expect((await screen.findByLabelText("Macro YAML") as HTMLTextAreaElement).value).toContain("name: wrapup")
    expect(screen.getByRole("switch", { name: "Toggle /wrapup" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Clone macro" })).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Save macro" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Delete macro" })).not.toBeInTheDocument()
  })

  it("refreshes the catalog and selects the new clone from the selected built-in", async () => {
    const user = userEvent.setup()
    const clonedMacro = makeMacro({
      name: "team_wrapup",
      command: "team_wrapup",
      source: "user",
      immutable: false,
      digest: "digest-team-wrapup"
    })
    mocks.listChatMacros
      .mockResolvedValueOnce(macroListResponse([builtinMacro]))
      .mockResolvedValueOnce(macroListResponse([builtinMacro, clonedMacro]))
    mocks.cloneChatMacro.mockResolvedValueOnce(success(makeDetail(clonedMacro)))

    render(<ChatMacrosSettings />)

    await user.click(await screen.findByRole("button", { name: "Clone macro" }))
    await user.type(await screen.findByLabelText("Clone macro name"), "team_wrapup")
    await user.click(screen.getByRole("button", { name: "Clone /wrapup" }))

    await waitFor(() =>
      expect(mocks.cloneChatMacro).toHaveBeenCalledWith("wrapup", {
        name: "team_wrapup",
        command: "team_wrapup"
      })
    )
    await waitFor(() => expect(mocks.getChatMacro).toHaveBeenCalledWith("team_wrapup"))
  })

  it("refreshes after create and delete while preserving a matching selection when possible", async () => {
    const user = userEvent.setup()
    const handoff = makeMacro({
      name: "handoff",
      command: "handoff",
      source: "user",
      immutable: false,
      digest: "digest-handoff"
    })
    mocks.listChatMacros
      .mockResolvedValueOnce(macroListResponse([builtinMacro, userMacro]))
      .mockResolvedValueOnce(macroListResponse([builtinMacro, userMacro, handoff]))
      .mockResolvedValueOnce(macroListResponse([builtinMacro, handoff]))
    mocks.createChatMacro.mockResolvedValueOnce(success(makeDetail(handoff)))
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: true, macro: { name: "handoff" } }))

    render(<ChatMacrosSettings />)

    await screen.findByRole("button", { name: "Select /wrapup" })
    await user.click(screen.getByRole("button", { name: "New macro" }))
    await user.type(screen.getByLabelText("Name"), "handoff")
    await user.type(screen.getByLabelText("Command"), "handoff")
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    await waitFor(() => expect(mocks.getChatMacro).toHaveBeenCalledWith("handoff"))

    await user.click(screen.getByRole("button", { name: "Select /research" }))
    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Delete macro" }))

    await waitFor(() => expect(mocks.deleteChatMacro).toHaveBeenCalledWith("research"))
    await waitFor(() => expect(screen.queryByRole("button", { name: "Select /research" })).not.toBeInTheDocument())
  })

  it("renders output profiles only after settings succeed", async () => {
    const user = userEvent.setup()
    const settings = deferred<ReturnType<typeof success<{ settings: ChatMacroSettings }>>>()
    mocks.getChatMacroSettings.mockReturnValueOnce(settings.promise)

    render(<ChatMacrosSettings />)

    await user.click(screen.getByRole("tab", { name: "Output profiles" }))
    expect(screen.getByText("Loading output profiles")).toBeInTheDocument()
    expect(screen.queryByLabelText("Profile")).not.toBeInTheDocument()

    await act(async () => {
      settings.resolve(success({ settings: makeSettings() }))
    })

    expect(await screen.findByLabelText("Profile")).toBeInTheDocument()
  })

  it("keeps macro and settings failures independently retryable", async () => {
    const user = userEvent.setup()
    mocks.listChatMacros
      .mockResolvedValueOnce({ ok: false, status: 503, error: "Macro catalog unavailable" })
      .mockResolvedValueOnce(macroListResponse([builtinMacro]))
    mocks.getChatMacroSettings
      .mockResolvedValueOnce({ ok: false, status: 503, error: "Settings unavailable" })
      .mockResolvedValueOnce(success({ settings: makeSettings() }))

    render(<ChatMacrosSettings />)

    expect(await screen.findByText("Macro catalog unavailable")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Retry macro list" }))
    expect(await screen.findByRole("button", { name: "Select /wrapup" })).toBeInTheDocument()

    await user.click(screen.getByRole("tab", { name: "Output profiles" }))
    expect(await screen.findByText("Settings unavailable")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Retry settings" }))
    expect(await screen.findByLabelText("Profile")).toBeInTheDocument()
  })

  it("keeps compact header actions available in narrow layouts", async () => {
    render(<ChatMacrosSettings />)

    const actions = await screen.findByTestId("chat-macro-header-actions")
    expect(actions).toHaveClass("flex-wrap")
    expect(screen.getByRole("button", { name: "New macro" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Import macro" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Refresh macros" })).toBeVisible()
  })

  it("ignores stale catalog refreshes and never updates after unmount", async () => {
    const user = userEvent.setup()
    const firstCatalog = deferred<ReturnType<typeof macroListResponse>>()
    const secondCatalog = deferred<ReturnType<typeof macroListResponse>>()
    const pendingSettings = deferred<ReturnType<typeof success<{ settings: ChatMacroSettings }>>>()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    mocks.listChatMacros
      .mockReturnValueOnce(firstCatalog.promise)
      .mockReturnValueOnce(secondCatalog.promise)
    mocks.getChatMacroSettings.mockReturnValueOnce(pendingSettings.promise)

    const { unmount } = render(<ChatMacrosSettings />)

    await waitFor(() => expect(mocks.listChatMacros).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole("button", { name: "Refresh macros" }))
    await waitFor(() => expect(mocks.listChatMacros).toHaveBeenCalledTimes(2))
    await act(async () => {
      secondCatalog.resolve(macroListResponse([userMacro]))
    })
    expect(await screen.findByRole("button", { name: "Select /research" })).toBeInTheDocument()

    await act(async () => {
      firstCatalog.resolve(macroListResponse([builtinMacro]))
    })
    expect(screen.queryByRole("button", { name: "Select /wrapup" })).not.toBeInTheDocument()

    unmount()
    await act(async () => {
      pendingSettings.resolve(success({ settings: makeSettings() }))
    })
    expect(errorSpy).not.toHaveBeenCalled()
    errorSpy.mockRestore()
  })
})
