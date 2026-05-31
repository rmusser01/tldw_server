import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import MediaTrashPage from "../MediaTrashPage"

const mocks = vi.hoisted(() => ({
  isOnline: true,
  demoEnabled: false,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  hasCompletedFirstRun: true,
  capsLoading: false,
  capabilities: { hasMedia: true },
  navigate: vi.fn(),
  refetch: vi.fn(),
  checkOnce: vi.fn()
}))

const interpolate = (template: string, values?: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (_, key) => String(values?.[key] ?? ""))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | {
            defaultValue?: string
            [k: string]: unknown
          }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      const template = fallbackOrOptions?.defaultValue || key
      return interpolate(
        template,
        fallbackOrOptions as Record<string, unknown> | undefined
      )
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mocks.navigate
}))

vi.mock("@tanstack/react-query", () => ({
  keepPreviousData: {},
  useQuery: () => ({
    data: {
      items: [],
      pagination: { page: 1, total_pages: 1, total_items: 0 }
    },
    isLoading: false,
    isFetching: false,
    isError: false,
    refetch: mocks.refetch
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => mocks.isOnline
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: mocks.capabilities,
    loading: mocks.capsLoading
  })
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: mocks.demoEnabled })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: mocks.uxState,
    hasCompletedFirstRun: mocks.hasCompletedFirstRun
  }),
  useConnectionActions: () => ({
    checkOnce: mocks.checkOnce
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => vi.fn().mockResolvedValue(true)
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn().mockResolvedValue({})
}))

vi.mock("@/services/tldw/path-utils", () => ({
  toAllowedPath: (path: string) => path
}))

vi.mock("@/components/Media/Pagination", () => ({
  Pagination: () => <div />
}))

describe("MediaTrashPage connection states", () => {
  beforeEach(() => {
    mocks.isOnline = true
    mocks.demoEnabled = false
    mocks.uxState = "connected_ok"
    mocks.hasCompletedFirstRun = true
    mocks.capsLoading = false
    mocks.capabilities = { hasMedia: true }
    mocks.navigate.mockReset()
    mocks.refetch.mockReset()
    mocks.checkOnce.mockReset()
  })

  it("shows credential guidance and opens settings when auth is missing", () => {
    mocks.isOnline = false
    mocks.uxState = "error_auth"

    render(<MediaTrashPage />)

    const title = screen.getByText("Add your credentials to use Media")
    expect(title).toBeInTheDocument()
    expect(
      title.closest('[data-ds-component="EmptyState"]')
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance and routes first-run users to setup", () => {
    mocks.isOnline = false
    mocks.uxState = "unconfigured"
    mocks.hasCompletedFirstRun = false

    render(<MediaTrashPage />)

    expect(screen.getByText("Finish setup to use Media")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(mocks.navigate).toHaveBeenCalledWith("/")
  })

  it("shows unreachable guidance with retry and diagnostics actions", () => {
    mocks.isOnline = false
    mocks.uxState = "error_unreachable"

    render(<MediaTrashPage />)

    expect(
      screen.getByText("Can't reach your tldw server right now")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }))
    expect(mocks.checkOnce).toHaveBeenCalled()

    fireEvent.click(
      screen.getByRole("button", { name: "Health & diagnostics" })
    )
    expect(mocks.navigate).toHaveBeenCalledWith("/settings/health")
  })
})
