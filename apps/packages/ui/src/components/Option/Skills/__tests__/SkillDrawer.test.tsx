import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
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

const renderDrawer = () => {
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
        onClose={vi.fn()}
        onSaved={vi.fn()}
      />
    </QueryClientProvider>
  )
}

const getContentEditor = () =>
  screen.getByLabelText("SKILL.md Content") as HTMLTextAreaElement

describe("SkillDrawer guided templates", () => {
  let confirmSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.clearAllMocks()

    if (!window.matchMedia) {
      Object.defineProperty(window, "matchMedia", {
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
    }

    if (typeof globalThis.ResizeObserver === "undefined") {
      globalThis.ResizeObserver = class {
        observe() {}
        unobserve() {}
        disconnect() {}
      } as unknown as typeof ResizeObserver
    }

    confirmSpy = vi.spyOn(Modal, "confirm").mockImplementation((config) => {
      return {
        destroy: vi.fn(),
        update: vi.fn(),
        config
      } as unknown as ReturnType<typeof Modal.confirm>
    })
  })

  afterEach(() => {
    cleanup()
    confirmSpy.mockRestore()
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
      target: { value: "Concept Coach" }
    })
    fireEvent.click(screen.getByRole("radio", { name: "Explainer" }))

    const editor = getContentEditor()
    expect(editor.value).toContain('name: "concept-coach"')
    expect(editor.value).toContain("Explain the following concept")
    expect(Modal.confirm).not.toHaveBeenCalled()
  })

  it("confirms before replacing manually edited content with a different template", () => {
    renderDrawer()

    const editor = getContentEditor()
    fireEvent.change(editor, {
      target: { value: "custom draft" }
    })

    fireEvent.click(screen.getByRole("radio", { name: "Extractor" }))

    expect(Modal.confirm).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Replace draft with template?",
        content: "This will replace your current SKILL.md draft with the selected template."
      })
    )
    expect(editor.value).toBe("custom draft")

    const [dialogConfig] = confirmSpy.mock.calls[0]
    act(() => {
      dialogConfig.onOk?.()
    })

    expect(getContentEditor().value).toContain("Extract structured information")
  })
})
