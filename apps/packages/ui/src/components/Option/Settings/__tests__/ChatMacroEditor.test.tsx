import React from "react"
import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { ChatMacroDetail, ChatMacroSummary } from "@/services/chat-macros"
import { ChatMacroEditor } from "../ChatMacroEditor"
import { createBlankMacroDraft, serializeGuidedMacro } from "../chat-macro-editor-utils"

const mocks = vi.hoisted(() => ({
  createChatMacro: vi.fn(),
  getChatMacro: vi.fn(),
  updateChatMacro: vi.fn(),
  deleteChatMacro: vi.fn(),
  validateChatMacro: vi.fn(),
  confirmDanger: vi.fn(),
  downloadBlob: vi.fn()
}))

vi.mock("@/services/chat-macros", () => ({
  createChatMacro: mocks.createChatMacro,
  getChatMacro: mocks.getChatMacro,
  updateChatMacro: mocks.updateChatMacro,
  deleteChatMacro: mocks.deleteChatMacro,
  validateChatMacro: mocks.validateChatMacro
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mocks.confirmDanger
}))

vi.mock("@/utils/download-blob", () => ({
  downloadBlob: mocks.downloadBlob
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | { defaultValue?: string }) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue || key
    }
  })
}))

const rawSource = [
  "schema_version: 1",
  "name: research",
  "command: research",
  "description: Collect evidence",
  "custom: preserve-this-source"
].join("\n")

const makeSummary = (overrides: Partial<ChatMacroSummary> = {}): ChatMacroSummary => ({
  name: "research",
  command: "research",
  description: "Collect evidence",
  enabled: true,
  source: "user",
  immutable: false,
  digest: "digest",
  schema_version: 1,
  validation_status: "valid",
  validation_error: null,
  ...overrides
})

const makeDetail = (overrides: Partial<ChatMacroDetail> = {}): ChatMacroDetail => ({
  summary: makeSummary(),
  definition: {
    schema_version: 1,
    name: "research",
    command: "research",
    description: "Collect evidence",
    enabled: true,
    args: {},
    context: {},
    execution: {},
    steps: [],
    output_profile: "default",
    permissions: { tool_calls: [], skills: [] }
  },
  raw: rawSource,
  supporting_files: {},
  ...overrides
})

const success = <T,>(data: T) => ({ ok: true, status: 200, data })

const createGuidedRaw = (overrides: Partial<ReturnType<typeof createBlankMacroDraft>> = {}) => {
  const draft = createBlankMacroDraft()
  return serializeGuidedMacro({
    ...draft,
    name: "research",
    command: "research",
    description: "Collect evidence",
    ...overrides
  })
}

const fillNewMacro = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.type(screen.getByLabelText("Name"), "handoff")
  await user.type(screen.getByLabelText("Command"), "handoff")
}

const renderEditor = (props: Partial<React.ComponentProps<typeof ChatMacroEditor>> = {}) => {
  const allProps: React.ComponentProps<typeof ChatMacroEditor> = {
    selected: null,
    outputProfileNames: ["default", "brief"],
    onSaved: vi.fn(),
    onDeleted: vi.fn(),
    onCloneRequested: vi.fn(),
    ...props
  }
  return {
    ...render(<ChatMacroEditor {...allProps} />),
    props: allProps
  }
}

describe("ChatMacroEditor", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getChatMacro.mockResolvedValue(success(makeDetail()))
    mocks.validateChatMacro.mockResolvedValue(success({ valid: true, macro: { name: "handoff" } }))
    mocks.createChatMacro.mockResolvedValue(success(makeDetail()))
    mocks.updateChatMacro.mockResolvedValue(success(makeDetail()))
    mocks.deleteChatMacro.mockResolvedValue(success(undefined))
    mocks.confirmDanger.mockResolvedValue(true)
  })

  it("creates a guided macro only after server validation succeeds", async () => {
    const user = userEvent.setup()
    renderEditor()

    await fillNewMacro(user)
    await user.type(screen.getByLabelText("Branch prompt 1"), "List decisions")
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    await waitFor(() => expect(mocks.validateChatMacro).toHaveBeenCalled())
    expect(mocks.createChatMacro).toHaveBeenCalledWith(
      expect.objectContaining({ name: "handoff" })
    )
  })

  it("announces a server validation failure without persisting the draft", async () => {
    const user = userEvent.setup()
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: false, error: "Unknown profile" }))
    renderEditor()

    await user.click(screen.getByRole("button", { name: "YAML" }))
    await user.type(screen.getByLabelText("Macro YAML"), "name: invalid")
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Unknown profile")
    expect(mocks.createChatMacro).not.toHaveBeenCalled()
    expect(mocks.updateChatMacro).not.toHaveBeenCalled()
  })

  it("loads editable details as raw source while keeping the selected name read-only", async () => {
    renderEditor({ selected: makeSummary() })

    expect(await screen.findByLabelText("Name")).toHaveValue("research")
    expect(screen.getByLabelText("Name")).toHaveAttribute("readonly")
    expect(screen.getByLabelText("Macro YAML")).toHaveValue(rawSource)
    expect(mocks.getChatMacro).toHaveBeenCalledWith("research")
  })

  it("switches a guided draft to source mode without losing its generated YAML", async () => {
    const user = userEvent.setup()
    renderEditor()

    await user.type(screen.getByLabelText("Name"), "handoff")
    await user.type(screen.getByLabelText("Branch prompt 1"), "List decisions")
    await user.click(screen.getByRole("button", { name: "YAML" }))

    expect((screen.getByLabelText("Macro YAML") as HTMLTextAreaElement).value).toContain("name: handoff")
    expect((screen.getByLabelText("Macro YAML") as HTMLTextAreaElement).value).toContain("prompt: List decisions")
  })

  it("imports YAML into source mode without automatically persisting it", async () => {
    const user = userEvent.setup()
    const { container } = renderEditor()
    const upload = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(["name: imported"], "imported.yaml", { type: "text/yaml" })

    await user.upload(upload, file)

    expect(await screen.findByLabelText("Macro YAML")).toHaveValue("name: imported")
    expect(mocks.createChatMacro).not.toHaveBeenCalled()
    expect(mocks.updateChatMacro).not.toHaveBeenCalled()
  })

  it("applies an external import request as a fresh source draft without persisting", async () => {
    const { rerender, props } = renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    rerender(
      <ChatMacroEditor
        {...props}
        selected={null}
        importSource={{ requestId: 1, raw: "name: imported\ncommand: imported" }}
      />
    )

    expect(await screen.findByLabelText("Macro YAML")).toHaveValue("name: imported\ncommand: imported")
    expect(screen.getByLabelText("Name")).toHaveValue("imported")
    expect(screen.getByLabelText("Name")).not.toHaveAttribute("readonly")
    expect(mocks.createChatMacro).not.toHaveBeenCalled()
    expect(mocks.updateChatMacro).not.toHaveBeenCalled()
  })

  it("exports the exact server raw source as a YAML blob", async () => {
    const user = userEvent.setup()
    renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Download macro YAML" }))

    expect(mocks.downloadBlob).toHaveBeenCalledWith(expect.any(Blob), "research.yaml")
    const [blob] = mocks.downloadBlob.mock.calls[0] as [Blob, string]
    expect(await blob.text()).toBe(rawSource)
  })

  it("confirms deletion with cancel focus and refreshes only after success", async () => {
    const user = userEvent.setup()
    const onDeleted = vi.fn()
    renderEditor({ selected: makeSummary(), onDeleted })

    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Delete macro" }))

    expect(mocks.confirmDanger).toHaveBeenCalledWith(
      expect.objectContaining({ autoFocusButton: "cancel" })
    )
    await waitFor(() => expect(mocks.deleteChatMacro).toHaveBeenCalledWith("research"))
    expect(onDeleted).toHaveBeenCalledWith("research")
  })

  it("keeps built-ins read-only and routes cloning through the parent", async () => {
    const user = userEvent.setup()
    const onCloneRequested = vi.fn()
    const selected = makeSummary({ source: "builtin", immutable: true })
    renderEditor({ selected, onCloneRequested })

    await screen.findByLabelText("Macro YAML")
    expect(screen.queryByRole("button", { name: "Save macro" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Delete macro" })).not.toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Clone macro" }))
    expect(onCloneRequested).toHaveBeenCalledWith(selected)
  })

  it("does not let a stale detail request replace a newer selection", async () => {
    let resolveFirst: ((value: ReturnType<typeof success<ChatMacroDetail>>) => void) | undefined
    let resolveSecond: ((value: ReturnType<typeof success<ChatMacroDetail>>) => void) | undefined
    mocks.getChatMacro.mockImplementation((name: string) => new Promise((resolve) => {
      if (name === "first") resolveFirst = resolve
      else resolveSecond = resolve
    }))
    const { rerender, props } = renderEditor({ selected: makeSummary({ name: "first" }) })

    rerender(
      <ChatMacroEditor
        {...props}
        selected={makeSummary({ name: "second" })}
      />
    )

    await act(async () => {
      resolveSecond?.(success(makeDetail({ raw: "name: second" })))
    })
    expect(await screen.findByLabelText("Macro YAML")).toHaveValue("name: second")

    await act(async () => {
      resolveFirst?.(success(makeDetail({ raw: "name: first" })))
    })
    expect(screen.getByLabelText("Macro YAML")).toHaveValue("name: second")
  })

  it("clears the previous detail and gates data actions when a new detail load fails", async () => {
    mocks.getChatMacro
      .mockResolvedValueOnce(success(makeDetail({ raw: "name: first" })))
      .mockResolvedValueOnce({ ok: false, status: 503, error: "Detail unavailable" })
    const { rerender, props } = renderEditor({ selected: makeSummary({ name: "first" }) })

    expect(await screen.findByLabelText("Macro YAML")).toHaveValue("name: first")
    rerender(<ChatMacroEditor {...props} selected={makeSummary({ name: "second" })} />)

    expect(await screen.findByRole("alert")).toHaveTextContent("Detail unavailable")
    expect(screen.queryByLabelText("Macro YAML")).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save macro" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Download macro YAML" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Copy macro YAML" })).toBeDisabled()
  })

  it("preserves the selected name and blocks a validated imported YAML name mismatch", async () => {
    const user = userEvent.setup()
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: true, macro: { name: "other" } }))
    const { container } = renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    const upload = container.querySelector('input[type="file"]') as HTMLInputElement
    await user.upload(upload, new File(["name: other\ncustom: keep"], "other.yaml", { type: "text/yaml" }))

    expect(screen.getByLabelText("Name")).toHaveValue("research")
    expect(screen.getByLabelText("Macro YAML")).toHaveValue("name: other\ncustom: keep")
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Validated macro name must match")
    expect(mocks.updateChatMacro).not.toHaveBeenCalled()
  })

  it("keeps the selected name after importing guided YAML and switching to Guided", async () => {
    const user = userEvent.setup()
    const { container } = renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    const upload = container.querySelector('input[type="file"]') as HTMLInputElement
    await user.upload(
      upload,
      new File([createGuidedRaw({ name: "imported", command: "handoff" })], "imported.yaml", {
        type: "text/yaml"
      })
    )

    expect(screen.getByLabelText("Name")).toHaveValue("research")

    await user.click(screen.getByRole("button", { name: "Guided" }))

    expect(screen.getByLabelText("Name")).toHaveValue("research")
    expect(screen.getByLabelText("Command")).toHaveValue("handoff")
  })

  it("allows an existing user macro command to change in Guided mode", async () => {
    const user = userEvent.setup()
    mocks.getChatMacro.mockResolvedValueOnce(success(makeDetail({ raw: createGuidedRaw() })))
    renderEditor({ selected: makeSummary() })

    await waitFor(() => expect(screen.getByLabelText("Command")).toHaveValue("research"))
    expect(screen.getByLabelText("Command")).not.toHaveAttribute("readonly")
    await user.clear(screen.getByLabelText("Command"))
    await user.type(screen.getByLabelText("Command"), "handoff")
    await user.click(screen.getByRole("button", { name: "YAML" }))

    expect((screen.getByLabelText("Macro YAML") as HTMLTextAreaElement).value).toContain("command: handoff")
  })

  it("recovers from a rejected validation request without losing the create draft", async () => {
    const user = userEvent.setup()
    mocks.validateChatMacro.mockRejectedValueOnce(new Error("Validation unavailable"))
    renderEditor()

    await fillNewMacro(user)
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Validation unavailable")
    expect(screen.getByLabelText("Name")).toHaveValue("handoff")
    expect(screen.getByRole("button", { name: "Save macro" })).toBeEnabled()
  })

  it("recovers from a rejected create request", async () => {
    const user = userEvent.setup()
    mocks.createChatMacro.mockRejectedValueOnce(new Error("Create unavailable"))
    renderEditor()

    await fillNewMacro(user)
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Create unavailable")
    expect(screen.getByRole("button", { name: "Save macro" })).toBeEnabled()
  })

  it("recovers from a rejected update request", async () => {
    const user = userEvent.setup()
    mocks.validateChatMacro.mockResolvedValueOnce(success({ valid: true, macro: { name: "research" } }))
    mocks.updateChatMacro.mockRejectedValueOnce(new Error("Update unavailable"))
    renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Save macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Update unavailable")
    expect(screen.getByRole("button", { name: "Save macro" })).toBeEnabled()
  })

  it("recovers from a rejected delete confirmation", async () => {
    const user = userEvent.setup()
    mocks.confirmDanger.mockRejectedValueOnce(new Error("Confirmation unavailable"))
    renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Delete macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Confirmation unavailable")
    expect(mocks.deleteChatMacro).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Delete macro" })).toBeEnabled()
  })

  it("recovers from a rejected delete request", async () => {
    const user = userEvent.setup()
    mocks.deleteChatMacro.mockRejectedValueOnce(new Error("Delete unavailable"))
    renderEditor({ selected: makeSummary() })

    await screen.findByLabelText("Macro YAML")
    await user.click(screen.getByRole("button", { name: "Delete macro" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("Delete unavailable")
    expect(screen.getByRole("button", { name: "Delete macro" })).toBeEnabled()
  })
})
