// @vitest-environment jsdom
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardTemplateValueModal } from "../FlashcardTemplateValueModal"
import { useFlashcardTemplatesQuery } from "../../hooks"

const messageApi = {
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
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
      return defaultValueOrOptions?.defaultValue ?? key
    }
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageApi
}))

vi.mock("../../hooks", () => ({
  useFlashcardTemplatesQuery: vi.fn()
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
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

describe("FlashcardTemplateValueModal", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders template load errors with the design-system alert", () => {
    vi.mocked(useFlashcardTemplatesQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Network offline")
    } as any)

    render(
      <FlashcardTemplateValueModal
        open
        onClose={vi.fn()}
        onApply={vi.fn()}
      />
    )

    const errorMessage = screen.getByText("Could not load templates.")
    expect(errorMessage.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    expect(errorMessage.closest('[data-ds-component="Alert"]')).toHaveTextContent("Network offline")
  })
})
