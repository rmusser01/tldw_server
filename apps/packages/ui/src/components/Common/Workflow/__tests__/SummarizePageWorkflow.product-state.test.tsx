import React from "react"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useWorkflowsStore } from "@/store/workflows"
import { SummarizePageWorkflow } from "../steps/SummarizePageWorkflow"

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

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: mocks.t,
  }),
}))

describe("SummarizePageWorkflow product-state UI", () => {
  beforeEach(() => {
    const realSetTimeout = globalThis.setTimeout
    vi.spyOn(globalThis, "setTimeout").mockImplementation(
      (...args: Parameters<typeof setTimeout>) => {
        if (args[1] === 500) {
          return 0 as unknown as ReturnType<typeof setTimeout>
        }
        return realSetTimeout(...args)
      }
    )

    vi.stubGlobal("chrome", {
      tabs: {
        query: vi.fn().mockResolvedValue([
          {
            id: 42,
            title: "Example Page",
            url: "https://example.test/page",
          },
        ]),
      },
      scripting: {
        executeScript: vi.fn().mockResolvedValue([
          {
            result: {
              title: "Captured Article",
              content:
                "Page content selected for summarization and long enough to show a length preview.",
            },
          },
        ]),
      },
    })

    useWorkflowsStore.setState({
      activeWorkflow: {
        id: "wf-summarize-page-test",
        workflowId: "summarize-page",
        status: "active",
        currentStepIndex: 0,
        startedAt: 1,
        data: {},
      },
      isWizardOpen: true,
      isProcessing: false,
      processingProgress: 0,
      processingMessage: "",
      error: null,
    })
    mocks.t.mockClear()
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
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

  it("renders captured page success through the canonical design-system Alert", async () => {
    const { container } = render(<SummarizePageWorkflow />)

    await waitFor(() => {
      expect(screen.getByText("Page captured")).toBeInTheDocument()
    })

    expect(container.querySelectorAll('[data-ds-component="Alert"]')).toHaveLength(1)
    expect(
      screen.getByText("Captured Article").closest('[data-ds-component="Alert"]')
    ).toBeInTheDocument()
    expect(screen.getByText("https://example.test/page")).toBeInTheDocument()
    expect(screen.getByText(/^Content length: \d+ characters$/)).toBeInTheDocument()
  })
})
