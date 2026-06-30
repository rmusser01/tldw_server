import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useWorkflowsStore } from "@/store/workflows"
import { AnalyzeBookWorkflow } from "../steps/AnalyzeBookWorkflow"

const mocks = vi.hoisted(() => ({
  t: vi.fn(
    (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      values?: Record<string, string | number | undefined>
    ) => {
      const template =
        typeof fallbackOrOptions === "string"
          ? fallbackOrOptions
          : fallbackOrOptions?.defaultValue || _key
      if (!values) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token: string) => {
        const value = values[token]
        return value === undefined ? `{{${token}}}` : String(value)
      })
    }
  ),
}))

const book = {
  file: null,
  mediaId: null,
  title: "Design Systems Handbook",
  content: "Chapter 1\nA compact test fixture for workflow rendering.",
  fileType: "md",
}

const warningChapter = {
  id: "chapter-1",
  number: 1,
  title: "Too short",
  content: "Brief.",
  wordCount: 1,
  charCount: 6,
  status: "warning" as const,
  preview: "Brief.",
}

const cleanChapter = {
  id: "chapter-2",
  number: 2,
  title: "Ready",
  content: "This chapter has enough text for the review fixture.",
  wordCount: 10,
  charCount: 52,
  status: "clean" as const,
  preview: "This chapter has enough text for the review fixture.",
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.t,
  }),
}))

const setAnalyzeBookWorkflow = (
  currentStepIndex: number,
  data: Record<string, unknown>
) => {
  useWorkflowsStore.setState({
    activeWorkflow: {
      id: "wf-analyze-book-product-state-test",
      workflowId: "analyze-book",
      status: "active",
      currentStepIndex,
      startedAt: 1,
      data,
    },
    isWizardOpen: true,
    isProcessing: false,
    processingProgress: 0,
    processingMessage: "",
    error: null,
  })
}

describe("AnalyzeBookWorkflow product-state UI", () => {
  beforeEach(() => {
    mocks.t.mockClear()
  })

  afterEach(() => {
    cleanup()
    useWorkflowsStore.setState({
      activeWorkflow: null,
      isWizardOpen: false,
      isProcessing: false,
      processingProgress: 0,
      processingMessage: "",
      error: null,
    })
    vi.clearAllMocks()
  })

  it("renders selected-book success through the canonical design-system Alert", () => {
    setAnalyzeBookWorkflow(0, { book })

    const { container } = render(<AnalyzeBookWorkflow />)

    expect(screen.getByText("Book selected")).toBeInTheDocument()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
    expect(
      screen
        .getByText("Design Systems Handbook")
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })

  it("renders chapter warnings through the canonical design-system Alert", () => {
    setAnalyzeBookWorkflow(1, {
      book,
      chapters: [warningChapter, cleanChapter],
    })

    const { container } = render(<AnalyzeBookWorkflow />)

    expect(screen.getByText("1 chapters may need review")).toBeInTheDocument()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
    expect(
      screen
        .getByText(
          "Some chapters appear too short or too long. You can adjust the chapter pattern below."
        )
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })

  it("renders analysis-complete success through the canonical design-system Alert", () => {
    setAnalyzeBookWorkflow(4, {
      book,
      chapters: [cleanChapter],
      analysisConfig: {
        preset: "comprehensive",
        customPrompt: "",
        selectedModel: "default",
        scope: "whole",
      },
      analysis: {
        wholeBook: "A compact analysis result.",
        perChapter: {},
      },
    })

    const { container } = render(<AnalyzeBookWorkflow />)

    expect(screen.getByText("Analysis Complete")).toBeInTheDocument()
    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
    expect(
      screen
        .getByText(
          "Your book analysis is ready. You can edit, save, or export the results below."
        )
        .closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
  })
})
