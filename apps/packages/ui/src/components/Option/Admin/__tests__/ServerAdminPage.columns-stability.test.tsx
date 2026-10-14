// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query"
import ServerAdminPage from "../ServerAdminPage"
import { createAdminQueryClient } from "../AdminQueryProvider"

// Capture the columns array identity each time a Table renders, so the test
// can assert memoized columns survive unrelated state changes (C-S5).
const columnLog = vi.hoisted(() => ({
  users: [] as unknown[][],
  roles: [] as unknown[][]
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const { createElement } = await import("react")
  return {
    ...actual,
    Table: (props: {
      columns?: Array<{ key?: string }> & unknown
      [k: string]: unknown
    }) => {
      const columns = props.columns as Array<{ key?: string }> | undefined
      if (Array.isArray(columns) && columns.length > 0) {
        if (columns[0]?.key === "username") columnLog.users.push(columns)
        if (columns[0]?.key === "name") columnLog.roles.push(columns)
      }
      return createElement(actual.Table, props as never)
    }
  }
})

const apiMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getSystemStats: vi.fn(),
  listAdminUsers: vi.fn(),
  createAdminUser: vi.fn(),
  listAdminRoles: vi.fn(),
  getMediaIngestionBudgetDiagnostics: vi.fn(),
  updateAdminUser: vi.fn(),
  resetAdminUserPassword: vi.fn(),
  createAdminRole: vi.fn(),
  deleteAdminRole: vi.fn()
}))

// The page's memoized columns list `t` as a dependency, so the mock must
// hand back a STABLE t reference - a fresh closure per render would churn
// the memos on every render and defeat the identity assertions.
const stableT = vi.hoisted(
  () =>
    (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions?.defaultValue) return fallbackOrOptions.defaultValue
      if (maybeOptions?.defaultValue) return maybeOptions.defaultValue
      return key
    }
)

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: stableT })
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

const adminQueryClients: QueryClient[] = []
const renderPage = (ui: React.ReactElement) => {
  const client = createAdminQueryClient()
  adminQueryClients.push(client)
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  )
}

describe("ServerAdminPage memoized column identities (C-S5)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    columnLog.users.length = 0
    columnLog.roles.length = 0

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

    apiMock.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user"
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
    apiMock.listAdminRoles.mockResolvedValue([
      { id: 1, name: "analyst", description: "read-only", is_system: false }
    ])
    apiMock.getMediaIngestionBudgetDiagnostics.mockResolvedValue({
      status: "ok",
      entity: "user:11",
      policy_id: "media.default",
      limits: {},
      usage: {},
      retry_after: null
    })
    apiMock.resetAdminUserPassword.mockResolvedValue({
      user_id: 11,
      force_password_change: true,
      message: "ok"
    })
  })

  it(
    "keeps user and roles column identities stable across a resetPasswordResult state change",
    async () => {
    renderPage(<ServerAdminPage />)

    await screen.findByText("admin@example.com")
    await screen.findByText("analyst")

    // Trigger the password-reset flow: resetPasswordResult (and the
    // transient updatingUserId) flip while the tables sit rendered.
    const rowButton = (
      await screen.findAllByRole("button", { name: "Reset password" })
    )[0]
    fireEvent.click(rowButton)
    const confirmButtons = await screen.findAllByRole("button", {
      name: "Reset password"
    })
    fireEvent.click(confirmButtons[confirmButtons.length - 1])

    await screen.findByTestId("admin-reset-password-result")

    // updatingUserId churned (null -> 11 -> null) through the reset, so the
    // memo legitimately recomputed; snapshot the post-reset identity.
    expect(columnLog.users.length).toBeGreaterThan(0)
    expect(columnLog.roles.length).toBeGreaterThan(0)
    const usersColumns = columnLog.users[columnLog.users.length - 1]
    const rolesColumns = columnLog.roles[columnLog.roles.length - 1]
    columnLog.users.length = 0
    columnLog.roles.length = 0

    // Close the reveal modal: a pure resetPasswordResult state change. The
    // tables re-render (log refills) but the memoized column arrays must be
    // the same references.
    fireEvent.click(screen.getByRole("button", { name: "Done" }))

    expect(columnLog.users.length).toBeGreaterThan(0)
    expect(columnLog.roles.length).toBeGreaterThan(0)
    for (const columns of columnLog.users) {
      expect(columns).toBe(usersColumns)
    }
    for (const columns of columnLog.roles) {
      expect(columns).toBe(rolesColumns)
    }
    },
    20000
  )
})
