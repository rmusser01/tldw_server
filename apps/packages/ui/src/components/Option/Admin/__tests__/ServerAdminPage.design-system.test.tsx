// @vitest-environment jsdom
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import ServerAdminPage from "../ServerAdminPage"

const apiMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getSystemStats: vi.fn(),
  listAdminUsers: vi.fn(),
  listAdminRoles: vi.fn(),
  getMediaIngestionBudgetDiagnostics: vi.fn(),
  updateAdminUser: vi.fn(),
  createAdminRole: vi.fn(),
  deleteAdminRole: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      if (maybeOptions?.defaultValue) return maybeOptions.defaultValue
      return _key
    }
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

vi.mock("../AdminAudioInstallerCard", () => ({
  AdminAudioInstallerCard: () => <div>Audio installer</div>
}))

const mockMatchMedia = () => {
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

const resolveBaseAdminCalls = () => {
  apiMock.getConfig.mockResolvedValue({
    serverUrl: "http://127.0.0.1:8000",
    authMode: "single-user"
  })
  apiMock.getSystemStats.mockResolvedValue({
    users: { total: 1, active: 1, admins: 1, verified: 1, new_last_30d: 0 },
    storage: {
      total_used_mb: 10,
      total_quota_mb: 100,
      average_used_mb: 10,
      max_used_mb: 10
    },
    sessions: { active: 1, unique_users: 1 }
  })
  apiMock.listAdminUsers.mockResolvedValue({
    users: [
      {
        id: 11,
        uuid: "user-11",
        username: "admin",
        email: "admin@example.com",
        role: "admin",
        is_active: true,
        is_verified: true,
        created_at: "2026-02-01T00:00:00Z",
        storage_quota_mb: 1024,
        storage_used_mb: 128
      }
    ],
    total: 1,
    page: 1,
    limit: 20,
    pages: 1
  })
  apiMock.listAdminRoles.mockResolvedValue([])
  apiMock.getMediaIngestionBudgetDiagnostics.mockResolvedValue({
    status: "ok",
    entity: "user:11",
    policy_id: "media.default",
    limits: {},
    usage: {},
    retry_after: null
  })
}

const expectDesignSystemAlertForTitle = async (titleText: string) => {
  const title = await screen.findByText(titleText)
  const alert = title.closest('[data-ds-component="Alert"]')

  expect(alert).not.toBeNull()
  const alertEl = alert as HTMLElement
  expect(alertEl).toHaveAttribute("role", "alert")
  return alertEl
}

describe("ServerAdminPage design-system states", () => {
  const docsUrl = "https://github.com/rmusser01/tldw_server#documentation--resources"

  beforeEach(() => {
    vi.clearAllMocks()
    mockMatchMedia()
    resolveBaseAdminCalls()
    vi.spyOn(window, "open").mockImplementation(() => null)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("uses PermissionNotice with an actionable recovery path for forbidden admin APIs", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce(
      Object.assign(new Error("Request failed: 403 Forbidden"), { status: 403 })
    )

    render(<ServerAdminPage />)

    expect(await screen.findByText("Permission denied")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Request access" }))

    expect(window.open).toHaveBeenCalledWith(
      docsUrl,
      "_blank",
      "noopener,noreferrer"
    )
  })

  it("uses a blocked callout with an actionable recovery path when admin APIs are missing", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce(
      Object.assign(new Error("Request failed: 404 Not Found"), { status: 404 })
    )

    render(<ServerAdminPage />)

    expect(await screen.findByText("Blocked")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Review server configuration" }))

    expect(window.open).toHaveBeenCalledWith(
      docsUrl,
      "_blank",
      "noopener,noreferrer"
    )
  })

  it("uses an Error state panel with a retry path for system stats failures", async () => {
    apiMock.getSystemStats
      .mockRejectedValueOnce(new Error("Request timed out"))
      .mockResolvedValueOnce({
        users: { total: 2, active: 2, admins: 1, verified: 2, new_last_30d: 1 },
        storage: {
          total_used_mb: 20,
          total_quota_mb: 200,
          average_used_mb: 10,
          max_used_mb: 15
        },
        sessions: { active: 2, unique_users: 2 }
      })

    render(<ServerAdminPage />)

    const errorLabel = await screen.findByText("Error")
    expect(errorLabel).toBeInTheDocument()

    const retryButton = within(
      errorLabel.closest("section") as HTMLElement
    ).getByRole("button", { name: "Retry" })
    fireEvent.click(retryButton)

    await waitFor(() => {
      expect(apiMock.getSystemStats).toHaveBeenCalledTimes(2)
    })
  })

  it("uses an Empty state panel when no media budget user can be selected", async () => {
    apiMock.listAdminUsers.mockResolvedValueOnce({
      users: [],
      total: 0,
      page: 1,
      limit: 20,
      pages: 0
    })

    render(<ServerAdminPage />)

    expect(await screen.findByText("Empty")).toBeInTheDocument()
    expect(
      screen.getByText("Select a user to inspect media ingestion limits and usage.")
    ).toBeInTheDocument()
  })

  it("renders user-load errors through the design-system Alert primitive", async () => {
    apiMock.listAdminUsers.mockRejectedValueOnce(new Error("Users exploded"))

    render(<ServerAdminPage />)

    const alert = await expectDesignSystemAlertForTitle("Unable to load users")
    expect(alert).toHaveTextContent("Users exploded")
  })

  it("renders role-load errors through the design-system Alert primitive", async () => {
    apiMock.listAdminRoles.mockRejectedValueOnce(new Error("Roles exploded"))

    render(<ServerAdminPage />)

    const alert = await expectDesignSystemAlertForTitle("Unable to load roles")
    expect(alert).toHaveTextContent("Roles exploded")
  })

  it("renders media budget diagnostic errors through the design-system Alert primitive", async () => {
    apiMock.getMediaIngestionBudgetDiagnostics.mockRejectedValueOnce(
      new Error("Budget exploded")
    )

    render(<ServerAdminPage />)

    const alert = await expectDesignSystemAlertForTitle(
      "Unable to load media ingestion budget diagnostics"
    )
    expect(alert).toHaveTextContent("Budget exploded")
  })
})
