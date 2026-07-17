import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Modal } from "antd"
import type { SkillResponse } from "@/types/skill"
import { SkillDrawer } from "../SkillDrawer"

const tldwClientMock = vi.hoisted(() => ({
  createSkill: vi.fn(),
  updateSkill: vi.fn(),
  getSkill: vi.fn()
}))

const notificationMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => notificationMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; [k: string]: unknown }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return (fallbackOrOptions.defaultValue || key).replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) => String(fallbackOrOptions[token] ?? "")
        )
      }
      return key
    }
  })
}))

const makeSkill = (overrides: Partial<SkillResponse> = {}): SkillResponse => ({
  id: "skill-1",
  name: "test-skill",
  description: "Skill description",
  argument_hint: "[arg]",
  disable_model_invocation: false,
  user_invocable: true,
  allowed_tools: ["Read", "Grep"],
  model: "gpt-4o-mini",
  context: "fork",
  content: "Body content",
  raw_content: "---\nname: test-skill\ncustom-key: true\n---\n\nBody content",
  supporting_files: { "notes.md": "hello" },
  directory_path: "/tmp/test-skill",
  created_at: "2026-07-14T00:00:00Z",
  last_modified: "2026-07-14T00:00:00Z",
  version: 3,
  ...overrides
})

const renderDrawer = (
  props: Partial<React.ComponentProps<typeof SkillDrawer>> = {}
) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <SkillDrawer
        open={props.open ?? true}
        skill={props.skill ?? null}
        duplicateFrom={props.duplicateFrom}
        draftScope={props.draftScope ?? "test-scope"}
        requestSignal={props.requestSignal}
        onClose={props.onClose ?? vi.fn()}
        onAfterClose={props.onAfterClose}
        onSaved={props.onSaved ?? vi.fn()}
      />
    </QueryClientProvider>
  )
}

describe("SkillDrawer authoring", () => {
  let confirmSpy: ReturnType<typeof vi.fn>
  let useModalSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.clearAllMocks()
    window.sessionStorage.clear()
    tldwClientMock.createSkill.mockResolvedValue({ name: "summarizer-skill" })
    tldwClientMock.updateSkill.mockResolvedValue({ name: "test-skill" })
    tldwClientMock.getSkill.mockResolvedValue(makeSkill({ version: 4 }))

    Object.defineProperty(window, "matchMedia", {
      configurable: true,
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })

    if (typeof globalThis.ResizeObserver === "undefined") {
      globalThis.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      } as unknown as typeof ResizeObserver
    }

    confirmSpy = vi.fn()
    useModalSpy = vi.spyOn(Modal, "useModal").mockReturnValue([
      { confirm: confirmSpy },
      null
    ] as unknown as ReturnType<typeof Modal.useModal>)
  })

  afterEach(() => {
    cleanup()
    useModalSpy.mockRestore()
  })

  it("opens new skills in guided mode with understandable fields", () => {
    renderDrawer()

    expect(screen.getByRole("dialog", { name: "New Skill: summarizer-skill" })).toBeInTheDocument()
    expect(screen.getByText("Start from template")).toBeInTheDocument()
    expect(screen.getByText("Condense source material into a short, useful answer.")).toBeInTheDocument()
    expect(screen.getByLabelText("Name")).toHaveValue("summarizer-skill")
    expect(screen.getByLabelText("Description")).toHaveValue(
      "Condense source material into a short, useful answer."
    )
    expect((screen.getByLabelText("Instructions") as HTMLTextAreaElement).value).toContain(
      "Summarize the following source material"
    )
    expect(screen.queryByLabelText("SKILL.md Content")).not.toBeInTheDocument()
  })

  it("switches templates without exposing or requiring YAML", () => {
    renderDrawer()

    fireEvent.click(screen.getByText("Explainer"))

    expect(screen.getByLabelText("Name")).toHaveValue("explainer-skill")
    expect((screen.getByLabelText("Instructions") as HTMLTextAreaElement).value).toContain(
      "Explain the following concept"
    )
    expect(confirmSpy).not.toHaveBeenCalled()
  })

  it("recovers the selected template with its draft values", () => {
    const first = renderDrawer()
    fireEvent.click(screen.getByText("Explainer"))
    first.unmount()

    renderDrawer()

    expect(screen.getByRole("radio", { name: "Explainer" })).toBeChecked()
    expect(screen.getByLabelText("Description")).toHaveValue(
      "Teach a concept step by step for a specific audience."
    )
  })

  it("does not recover an authoring draft from another identity scope", () => {
    const first = renderDrawer({ draftScope: "server-a:user-1" })
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Private draft for user one" }
    })
    first.unmount()

    renderDrawer({ draftScope: "server-a:user-2" })

    expect(screen.getByLabelText("Description")).toHaveValue(
      "Condense source material into a short, useful answer."
    )
    expect(
      screen.queryByText("Recovered your unsaved draft from this session.")
    ).not.toBeInTheDocument()
  })

  it("generates canonical source for advanced users", () => {
    renderDrawer()

    fireEvent.click(screen.getByText("Advanced source"))

    const editor = screen.getByLabelText("SKILL.md Content")
    expect((editor as HTMLTextAreaElement).value).toContain('name: "summarizer-skill"')
    expect((editor as HTMLTextAreaElement).value).toContain("$ARGUMENTS")
  })

  it("shows the canonical name in source mode and rejects a frontmatter mismatch", async () => {
    renderDrawer()
    fireEvent.click(screen.getByText("Advanced source"))

    const nameInput = screen.getByLabelText("Name")
    expect(nameInput).toBeVisible()
    fireEvent.change(nameInput, { target: { value: "canonical-skill" } })
    fireEvent.change(screen.getByLabelText("SKILL.md Content"), {
      target: {
        value: "---\nname: other-skill\ndescription: mismatch\n---\n\nBody"
      }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(
      await screen.findByText(
        'Frontmatter name "other-skill" must match canonical name "canonical-skill".'
      )
    ).toBeInTheDocument()
    expect(tldwClientMock.createSkill).not.toHaveBeenCalled()
  })

  it("creates a skill from guided fields", async () => {
    const onSaved = vi.fn()
    tldwClientMock.createSkill.mockResolvedValueOnce({ name: "research-helper" })
    renderDrawer({ onSaved })

    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "research-helper" }
    })
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Research a topic" }
    })
    fireEvent.change(screen.getByLabelText("Instructions"), {
      target: { value: "Research $ARGUMENTS." }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(tldwClientMock.createSkill).toHaveBeenCalledWith({
        name: "research-helper",
        content: expect.stringContaining("Research $ARGUMENTS.")
      })
    })
    expect(window.sessionStorage.length).toBe(0)
    expect(onSaved).toHaveBeenCalledWith("research-helper")
  })

  it("ignores a create result after its identity scope is aborted", async () => {
    const onSaved = vi.fn()
    const controller = new AbortController()
    let resolveCreate: ((result: { name: string }) => void) | undefined
    let createSignal: AbortSignal | undefined
    tldwClientMock.createSkill.mockImplementationOnce(
      (_payload: unknown, options?: { signal?: AbortSignal }) => {
        createSignal = options?.signal
        return new Promise((resolve) => {
          resolveCreate = resolve
        })
      }
    )
    renderDrawer({ onSaved, requestSignal: controller.signal })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(tldwClientMock.createSkill).toHaveBeenCalledTimes(1))

    controller.abort()
    await act(async () => {
      resolveCreate?.({ name: "summarizer-skill" })
      await Promise.resolve()
    })

    expect(createSignal).toBe(controller.signal)
    expect(onSaved).not.toHaveBeenCalled()
    expect(notificationMock.success).not.toHaveBeenCalled()
    expect(notificationMock.error).not.toHaveBeenCalled()
  })

  it("keeps invalid guided input local and actionable", async () => {
    renderDrawer()

    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "Bad Name" }
    })
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByText(/Name must start with a lowercase letter/)).toBeInTheDocument()
    expect(screen.getByText("Description is required.")).toBeInTheDocument()
    expect(tldwClientMock.createSkill).not.toHaveBeenCalled()
  })

  it("guards every dirty close path and discards only after confirmation", () => {
    const onClose = vi.fn()
    renderDrawer({ onClose })

    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Changed description" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    expect(onClose).not.toHaveBeenCalled()
    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Discard unsaved skill draft?",
        cancelText: "Keep editing",
        okText: "Discard draft"
      })
    )

    const config = confirmSpy.mock.calls[0][0] as { onOk?: () => void }
    act(() => config.onOk?.())
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(window.sessionStorage.length).toBe(0)
  })

  it("closes a pristine drawer without prompting", () => {
    const onClose = vi.fn()
    renderDrawer({ onClose })

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(confirmSpy).not.toHaveBeenCalled()
  })

  it("recovers an unsaved draft within the browser session", () => {
    const first = renderDrawer()
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Recovered description" }
    })
    first.unmount()

    renderDrawer()

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Recovered your unsaved draft from this session."
    )
    expect(screen.getByLabelText("Description")).toHaveValue("Recovered description")
  })

  it("requires conflict review when a recovered edit draft has an older base version", async () => {
    const first = renderDrawer({
      skill: makeSkill({ version: 3, raw_content: null })
    })
    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "Recovered local edit" }
    })
    first.unmount()

    renderDrawer({
      skill: makeSkill({ version: 4, raw_content: null })
    })

    expect(await screen.findByText(
      "This skill changed elsewhere. Review the latest version before choosing whether to overwrite it."
    )).toBeInTheDocument()
    expect(screen.getByLabelText("Description")).toHaveValue("Recovered local edit")
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    expect(tldwClientMock.updateSkill).not.toHaveBeenCalled()
  })

  it("opens custom frontmatter in source mode and saves it without data loss", async () => {
    renderDrawer({ skill: makeSkill() })

    expect(screen.getByRole("dialog", { name: "Edit Skill: test-skill" })).toBeInTheDocument()
    const source = screen.getByLabelText("SKILL.md Content") as HTMLTextAreaElement
    expect(source.value).toContain("custom-key: true")

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(tldwClientMock.updateSkill).toHaveBeenCalledWith(
        "test-skill",
        { content: makeSkill().raw_content },
        3
      )
    })
  })

  it("rejects a frontmatter name mismatch while editing advanced source", async () => {
    renderDrawer({ skill: makeSkill() })
    fireEvent.change(screen.getByLabelText("SKILL.md Content"), {
      target: {
        value: "---\n{name: other-skill, description: Mismatch}\n---\n\nBody"
      }
    })

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(
      await screen.findByText(
        'Frontmatter name "other-skill" must match canonical name "test-skill".'
      )
    ).toBeInTheDocument()
    expect(tldwClientMock.updateSkill).not.toHaveBeenCalled()
  })

  it("duplicates custom source and supporting files under the new name", async () => {
    renderDrawer({ duplicateFrom: makeSkill() })

    expect(screen.getByRole("dialog", { name: "New Skill: test-skill-copy" })).toBeInTheDocument()
    const source = screen.getByLabelText("SKILL.md Content") as HTMLTextAreaElement
    expect(source.value).toContain('name: "test-skill-copy"')
    expect(source.value).toContain("custom-key: true")
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(tldwClientMock.createSkill).toHaveBeenCalledWith({
        name: "test-skill-copy",
        content: expect.stringMatching(/name: "test-skill-copy"[\s\S]*custom-key: true/),
        supporting_files: { "notes.md": "hello" }
      })
    })
  })

  it("preserves the local draft and requires review before overwriting a newer version", async () => {
    tldwClientMock.updateSkill
      .mockRejectedValueOnce(Object.assign(new Error("HTTP 409"), { status: 409 }))
      .mockResolvedValueOnce({ name: "test-skill" })
    tldwClientMock.getSkill.mockResolvedValueOnce(makeSkill({
      version: 4,
      raw_content: "---\nname: test-skill\nremote-change: true\n---\n\nRemote body"
    }))
    renderDrawer({ skill: makeSkill({ raw_content: null }) })

    fireEvent.change(screen.getByLabelText("Description"), {
      target: { value: "My local changes" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This skill changed elsewhere. Review the latest version before choosing whether to overwrite it."
    )
    expect(screen.getByLabelText("Description")).toHaveValue("My local changes")
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Review latest" }))
    await waitFor(() => expect(tldwClientMock.getSkill).toHaveBeenCalledWith("test-skill"))
    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Overwrite the latest server version?",
        okText: "Keep draft and overwrite"
      })
    )
    const confirmConfig = confirmSpy.mock.calls.at(-1)?.[0] as {
      content?: React.ReactNode
      onOk?: () => void
    }
    render(<>{confirmConfig.content}</>)
    expect(screen.getByText(/remote-change: true/)).toBeInTheDocument()
    expect(screen.getByRole("region", { name: "Latest server source" })).toHaveAttribute(
      "tabindex",
      "0"
    )
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()

    act(() => confirmConfig.onOk?.())
    await waitFor(() => expect(screen.getByRole("button", { name: "Save" })).not.toBeDisabled())
    expect(screen.getByLabelText("Description")).toHaveValue("My local changes")

    const saveButton = screen.getByText("Save").closest("button")
    expect(saveButton).not.toBeNull()
    fireEvent.click(saveButton as HTMLButtonElement)
    await waitFor(() => {
      expect(tldwClientMock.updateSkill).toHaveBeenLastCalledWith(
        "test-skill",
        expect.any(Object),
        4
      )
    })
  })

  it("confirms before replacing manually edited source with guided fields", () => {
    renderDrawer()
    fireEvent.click(screen.getByText("Advanced source"))
    fireEvent.change(screen.getByLabelText("SKILL.md Content"), {
      target: { value: "custom source" }
    })

    fireEvent.click(screen.getByText("Guided"))

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Replace advanced source?" })
    )
    expect(screen.getByLabelText("SKILL.md Content")).toHaveValue("custom source")
  })
})
