import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { useQuery } from "@tanstack/react-query"
import { ContentTypePicker } from "../ChatbooksPlaygroundPage"

const { useQueryMock, refetchMock } = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  refetchMock: vi.fn()
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock
}))

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

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const expectInsideDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  expect(node.closest('[data-ds-component="Alert"]')).toHaveAttribute(
    "data-ds-component",
    "Alert"
  )
}

describe("ContentTypePicker load error state", () => {
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
    vi.mocked(useQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error(
        "Request failed: 500 GET /api/v1/chatbooks/export request_id=cbk-123abc (/Users/macbook-dev/private/path.log)"
      ),
      refetch: refetchMock
    } as any)
  })

  it("shows sanitized load error text and exposes retry action", () => {
    render(
      <ContentTypePicker
        typeKey="conversation"
        label="Conversations"
        includeAll={false}
        onIncludeAllChange={vi.fn()}
        selectedIds={[]}
        onSelectionChange={vi.fn()}
        fetcher={vi.fn()}
      />
    )

    expect(screen.getByText("Unable to load items")).toBeInTheDocument()
    expectInsideDesignSystemAlert("Unable to load items")
    expect(screen.getByText(/GET \[server-endpoint\]/i)).toBeInTheDocument()
    expect(screen.getByText(/\[redacted-path\]/i)).toBeInTheDocument()
    expect(
      screen.getByText(/Check server logs with correlation ID: cbk-123abc\./i)
    ).toBeInTheDocument()
    expect(screen.queryByText("/api/v1/chatbooks/export")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(refetchMock).toHaveBeenCalledTimes(1)
  })
})
