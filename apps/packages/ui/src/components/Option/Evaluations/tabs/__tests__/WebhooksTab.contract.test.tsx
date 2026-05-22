// @vitest-environment jsdom

import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Modal } from "antd"
import { WebhooksTab } from "../WebhooksTab"

const deleteSpy = vi.fn()

const storeState = {
  webhookSecretText: null as string | null
}
const webhooksListState = {
  data: {
    ok: true,
    data: [
      {
        webhook_id: 17,
        url: "https://example.com/webhooks/evals",
        events: ["evaluation.completed"],
        status: "active",
        created_at: "2026-03-29T12:00:00Z"
      }
    ]
  },
  isLoading: false,
  isError: false,
  error: null as Error | null
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/store/evaluations", () => ({
  useEvaluationsStore: (selector: (state: typeof storeState) => unknown) =>
    selector(storeState)
}))

vi.mock("../../components", () => ({
  CopyButton: () => null
}))

vi.mock("../../hooks/useWebhooks", () => ({
  useWebhooksList: () => webhooksListState,
  useRegisterWebhook: () => ({
    mutateAsync: vi.fn(),
    isPending: false
  }),
  useDeleteWebhook: () => ({
    mutateAsync: deleteSpy,
    isPending: false
  }),
  webhookEventOptions: [
    { value: "evaluation.completed", label: "evaluation.completed" }
  ],
  defaultWebhookEvents: ["evaluation.completed"]
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("WebhooksTab backend contract", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
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
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    webhooksListState.data = {
      ok: true,
      data: [
        {
          webhook_id: 17,
          url: "https://example.com/webhooks/evals",
          events: ["evaluation.completed"],
          status: "active",
          created_at: "2026-03-29T12:00:00Z"
        }
      ]
    }
    webhooksListState.isLoading = false
    webhooksListState.isError = false
    webhooksListState.error = null
    vi.spyOn(Modal, "confirm").mockImplementation((config: any) => {
      void config?.onOk?.()
      return {
        destroy: vi.fn(),
        update: vi.fn()
      } as any
    })
  })

  it("renders bare-array webhook responses and unregisters by URL", async () => {
    render(<WebhooksTab />)

    expect(
      screen.getByText("https://example.com/webhooks/evals")
    ).toBeInTheDocument()
    expect(screen.getByText(/ID/i)).toHaveTextContent("17")
    expect(screen.getByText("Active")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Delete" }))

    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith("https://example.com/webhooks/evals")
    })
  })

  it("shows endpoint diagnostics while preserving the webhook registration form", () => {
    webhooksListState.data = { ok: false, error: "HTTP 503" } as any
    webhooksListState.isError = true
    webhooksListState.error = new Error("HTTP 503")

    render(<WebhooksTab />)

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByText("Unable to load webhooks")).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "/api/v1/evaluations/webhooks"
    )
    expect(screen.getByRole("button", { name: "Register webhook" })).toBeEnabled()
  })
})
