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
  updateChatMacroOutputProfiles: vi.fn(),
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
  updateChatMacroOutputProfiles: mocks.updateChatMacroOutputProfiles,
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
    vi.resetAllMocks()
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
    mocks.updateChatMacroOutputProfiles.mockImplementation((output_profiles: ChatMacroSettings["output_profiles"]) =>
      Promise.resolve(success({ settings: makeSettings({ output_profiles }) }))
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

  it("preserves an edited imported draft across tab changes without replaying it", async () => {
    const user = userEvent.setup()
    render(<ChatMacrosSettings />)

    const upload = screen.getByLabelText("Import macro YAML file") as HTMLInputElement
    await user.upload(
      upload,
      new File(["name: imported"], "imported.yaml", { type: "text/yaml" })
    )
    const source = await screen.findByLabelText("Macro YAML")
    await user.type(source, "\nchanged: true")

    await user.click(screen.getByRole("tab", { name: "Output profiles" }))
    await user.click(screen.getByRole("tab", { name: "Macros" }))

    expect(await screen.findByLabelText("Name")).toHaveValue("imported")
    expect(screen.getByLabelText("Macro YAML")).toHaveValue(
      "name: imported\nchanged: true"
    )
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
    await user.click(screen.getByRole("button", { name: "Clone macro" }))
    expect(screen.getByLabelText("Clone macro name")).toHaveClass("bg-surface")
    expect(screen.queryByRole("button", { name: "Save macro" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Delete macro" })).not.toBeInTheDocument()
  })

  it("preserves the selected dirty draft when toggling another macro refreshes the catalog", async () => {
    const user = userEvent.setup()
    mocks.listChatMacros
      .mockResolvedValueOnce(macroListResponse([builtinMacro, userMacro]))
      .mockResolvedValueOnce(macroListResponse([
        { ...builtinMacro, enabled: false },
        { ...userMacro }
      ]))

    render(<ChatMacrosSettings />)

    await user.click(await screen.findByRole("button", { name: "Select /research" }))
    const source = await screen.findByLabelText("Macro YAML")
    await user.type(source, "\ncustom: dirty")
    await user.click(screen.getByRole("switch", { name: "Toggle /wrapup" }))

    await waitFor(() => expect(mocks.listChatMacros).toHaveBeenCalledTimes(2))
    expect(screen.getByRole("button", { name: "Select /research" })).toHaveAttribute(
      "aria-pressed",
      "true"
    )
    expect((screen.getByLabelText("Macro YAML") as HTMLTextAreaElement).value).toContain(
      "custom: dirty"
    )
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

  it.each(["success", "failure"])("preserves an unsaved output-profile heading during refresh and after %s", async (outcome) => {
    const user = userEvent.setup()
    const refresh = deferred<ReturnType<typeof success<{ settings: ChatMacroSettings }>> | { ok: false; status: number; error: string }>()
    render(<ChatMacrosSettings />)
    await user.click(screen.getByRole("tab", { name: "Output profiles" }))
    const heading = await screen.findByLabelText("Section heading 1")
    await user.type(heading, "Unsaved heading")
    mocks.getChatMacroSettings.mockReturnValueOnce(refresh.promise)

    await user.click(screen.getByRole("button", { name: "Refresh macros" }))

    expect(screen.getByText("Loading output profiles")).toBeVisible()
    expect(screen.getByLabelText("Section heading 1")).toBe(heading)
    expect(heading).toHaveValue("Unsaved heading")
    await act(async () => {
      refresh.resolve(outcome === "success"
        ? success({ settings: makeSettings() })
        : { ok: false, status: 503, error: "Settings refresh unavailable" })
    })

    expect(screen.queryByText("Loading output profiles")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Section heading 1")).toBe(heading)
    expect(heading).toHaveValue("Unsaved heading")
    expect(mocks.updateChatMacroOutputProfiles).not.toHaveBeenCalled()
    if (outcome === "failure") {
      expect(screen.getByText("Settings refresh unavailable")).toBeVisible()
      expect(screen.getByRole("button", { name: "Retry settings" })).toBeEnabled()
      await user.click(screen.getByRole("button", { name: "Retry settings" }))
      await waitFor(() => expect(screen.queryByText("Settings refresh unavailable")).not.toBeInTheDocument())
      expect(screen.getByLabelText("Section heading 1")).toHaveValue("Unsaved heading")
    }
  })

  it.each(["save", "delete"])("refreshes a completed %s without replacing a newer imported draft", async (operation) => {
    const user = userEvent.setup()
    const mutation = deferred<ReturnType<typeof success<ChatMacroDetail>>>()
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: true, macro: { name: "research" } }))
    if (operation === "save") mocks.updateChatMacro.mockReturnValueOnce(mutation.promise)
    else mocks.deleteChatMacro.mockReturnValueOnce(mutation.promise)
    render(<ChatMacrosSettings />)
    await user.click(await screen.findByRole("button", { name: "Select /research" }))
    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: operation === "save" ? "Save macro" : "Delete macro" }))
    await waitFor(() => expect(operation === "save" ? mocks.updateChatMacro : mocks.deleteChatMacro).toHaveBeenCalled())
    await user.upload(screen.getByLabelText("Import macro YAML file"), new File(["name: imported"], "imported.yaml", { type: "text/yaml" }))
    await waitFor(() => expect(screen.getByLabelText("Name")).toHaveValue("imported"))
    mocks.listChatMacros.mockResolvedValueOnce(macroListResponse(operation === "save" ? [builtinMacro, userMacro] : [builtinMacro]))
    await act(async () => { mutation.resolve(success(makeDetail(userMacro))) })
    await waitFor(() => expect(mocks.listChatMacros).toHaveBeenCalledTimes(2))
    expect(screen.getByLabelText("Name")).toHaveValue("imported")
    expect(screen.getByLabelText("Macro YAML")).toHaveValue("name: imported")
    if (operation === "delete") expect(screen.queryByRole("button", { name: "Select /research" })).not.toBeInTheDocument()
  })

  it("does not reselect a saved macro when its catalog refresh finishes after navigation", async () => {
    const user = userEvent.setup()
    const catalog = deferred<ReturnType<typeof macroListResponse>>()
    mocks.listChatMacros.mockResolvedValueOnce(macroListResponse([builtinMacro, userMacro])).mockReturnValueOnce(catalog.promise)
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: true, macro: { name: "research" } }))
    render(<ChatMacrosSettings />)
    await user.click(await screen.findByRole("button", { name: "Select /research" }))
    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Save macro" }))
    await waitFor(() => expect(mocks.listChatMacros).toHaveBeenCalledTimes(2))
    await user.click(screen.getByRole("button", { name: "New macro" }))
    await user.type(screen.getByLabelText("Name"), "newer")
    await act(async () => { catalog.resolve(macroListResponse([builtinMacro, userMacro])) })
    expect(screen.getByLabelText("Name")).toHaveValue("newer")
    expect(screen.getByRole("button", { name: "Select /research" })).toHaveAttribute("aria-pressed", "false")
  })

  it("refreshes a late clone without selecting it over a newer draft", async () => {
    const user = userEvent.setup()
    const clone = deferred<ReturnType<typeof success<ChatMacroDetail>>>()
    mocks.cloneChatMacro.mockReturnValueOnce(clone.promise)
    render(<ChatMacrosSettings />)
    await user.click(await screen.findByRole("button", { name: "Clone macro" }))
    await user.type(screen.getByLabelText("Clone macro name"), "research")
    await user.click(screen.getByRole("button", { name: "Clone /wrapup" }))
    await user.click(screen.getByRole("button", { name: "New macro" }))
    await user.type(screen.getByLabelText("Name"), "newer")
    await act(async () => { clone.resolve(success(makeDetail(userMacro))) })
    await waitFor(() => expect(mocks.listChatMacros).toHaveBeenCalledTimes(2))
    expect(screen.getByLabelText("Name")).toHaveValue("newer")
  })

  it("does not apply a pending file read after a newer selection", async () => {
    const user = userEvent.setup()
    const read = deferred<string>()
    const file = new File(["name: imported"], "imported.yaml", { type: "text/yaml" })
    Object.defineProperty(file, "text", { value: () => read.promise })
    render(<ChatMacrosSettings />)
    await screen.findByRole("button", { name: "Select /research" })
    await user.upload(screen.getByLabelText("Import macro YAML file"), file)
    await user.click(screen.getByRole("button", { name: "Select /research" }))
    await act(async () => { read.resolve("name: imported") })
    expect(screen.getByRole("button", { name: "Select /research" })).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByLabelText("Name")).toHaveValue("research")
  })

  it.each(["save", "delete"])("does not let a %s callback cancel a newer pending import", async (operation) => {
    const user = userEvent.setup()
    const mutation = deferred<ReturnType<typeof success<ChatMacroDetail>>>()
    const read = deferred<string>()
    const file = new File(["name: imported"], "imported.yaml", { type: "text/yaml" })
    Object.defineProperty(file, "text", { value: () => read.promise })
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: true, macro: { name: "research" } }))
    if (operation === "save") mocks.updateChatMacro.mockReturnValueOnce(mutation.promise)
    else mocks.deleteChatMacro.mockReturnValueOnce(mutation.promise)
    render(<ChatMacrosSettings />)
    await user.click(await screen.findByRole("button", { name: "Select /research" }))
    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: operation === "save" ? "Save macro" : "Delete macro" }))
    await user.upload(screen.getByLabelText("Import macro YAML file"), file)
    await act(async () => { mutation.resolve(success(makeDetail(userMacro))) })
    await act(async () => { read.resolve("name: imported") })
    expect(screen.getByLabelText("Name")).toHaveValue("imported")
    expect(screen.getByLabelText("Macro YAML")).toHaveValue("name: imported")
    expect(mocks.listChatMacros).toHaveBeenCalledTimes(2)
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
