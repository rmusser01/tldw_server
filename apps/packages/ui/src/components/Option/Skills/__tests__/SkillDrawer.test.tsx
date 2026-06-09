import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { Modal } from "antd"
import { SkillDrawer } from "../SkillDrawer"

const tldwClientMock = vi.hoisted(() => ({
  createSkill: vi.fn(),
  updateSkill: vi.fn()
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
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

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
        open
        skill={null}
        onClose={props.onClose ?? vi.fn()}
        onSaved={props.onSaved ?? vi.fn()}
      />
    </QueryClientProvider>
  )
}

const getContentEditor = () =>
  screen.getByLabelText("SKILL.md Content") as HTMLTextAreaElement

describe("SkillDrawer guided templates", () => {
  let confirmSpy: ReturnType<typeof vi.fn>
  let useModalSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.clearAllMocks()

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

  it("opens new skills with a beginner-friendly template draft", () => {
    renderDrawer()

    expect(screen.getByText("Start from template")).toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Summarizer" })).toBeChecked()

    const editor = getContentEditor()
    expect(editor.value).toContain('name: "summarizer-skill"')
    expect(editor.value).toContain("Summarize the following source material")
    expect(editor.value).toContain("$ARGUMENTS")
  })

  it("applies another template immediately while the generated draft is unchanged", () => {
    renderDrawer()

    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "concept-coach" }
    })
    fireEvent.click(screen.getByRole("radio", { name: "Explainer" }))

    const editor = getContentEditor()
    expect(editor.value).toContain('name: "concept-coach"')
    expect(editor.value).toContain("Explain the following concept")
    expect(confirmSpy).not.toHaveBeenCalled()
  })

  it("does not silently slugify invalid name input into the generated draft", () => {
    renderDrawer()

    const editor = getContentEditor()
    const originalDraft = editor.value

    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "Concept Coach" }
    })

    expect(editor.value).toBe(originalDraft)
    expect(editor.value).not.toContain('name: "concept-coach"')

    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "concept-coach" }
    })

    expect(editor.value).toContain('name: "concept-coach"')
  })

  it("confirms before replacing manually edited content with a different template", () => {
    renderDrawer()

    const editor = getContentEditor()
    fireEvent.change(editor, {
      target: { value: "custom draft" }
    })

    fireEvent.click(screen.getByRole("radio", { name: "Extractor" }))

    expect(confirmSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Replace draft with template?",
        content: "This will replace your current SKILL.md draft with the selected template."
      })
    )
    expect(editor.value).toBe("custom draft")

    const [dialogConfig] = confirmSpy.mock.calls[0] as [
      { onOk?: () => void | Promise<void> }
    ]
    act(() => {
      dialogConfig.onOk?.()
    })

    expect(getContentEditor().value).toContain("Extract structured information")
  })

  it("falls back to the validated form name when the API returns an invalid created name", async () => {
    const onSaved = vi.fn()
    tldwClientMock.createSkill.mockResolvedValueOnce({ name: "Created Skill" })

    renderDrawer({ onSaved })

    fireEvent.change(screen.getByLabelText("Name"), {
      target: { value: "created-skill" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(tldwClientMock.createSkill).toHaveBeenCalledWith(
        expect.objectContaining({ name: "created-skill" })
      )
    })
    await waitFor(() => {
      expect(onSaved).toHaveBeenCalledWith("created-skill")
    })
  })
})
