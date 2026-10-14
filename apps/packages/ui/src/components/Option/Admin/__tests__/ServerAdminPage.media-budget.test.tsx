import React from "react"
import { QueryClientProvider, type QueryClient } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import ServerAdminPage from "../ServerAdminPage"
import { createAdminQueryClient } from "../AdminQueryProvider"

/**
 * Render the page under a fresh admin query client so each test owns its
 * cache (B-S4): system stats now flow through react-query.
 */
const adminQueryClients: QueryClient[] = []
const renderPage = (ui: React.ReactElement) => {
  const client = createAdminQueryClient()
  adminQueryClients.push(client)
  return render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  adminQueryClients.splice(0).forEach((client) => client.clear())
})

const apiMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getSystemStats: vi.fn(),
  listAdminUsers: vi.fn(),
  listAdminRoles: vi.fn(),
  getMediaIngestionBudgetDiagnostics: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions
      }
      if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        typeof fallbackOrOptions.defaultValue === "string"
      ) {
        return fallbackOrOptions.defaultValue
      }
      return maybeOptions?.defaultValue || key
    }
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: apiMock
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
}))

/**
 * Diagnostics load only after an explicit user selection (B-S5): open the
 * media-budget user Select and pick the option with the given label.
 * (antd v6 keeps the mousedown handler on the `.ant-select` root — the old
 * `.ant-select-selector` inner node no longer exists.)
 */
const selectMediaBudgetUser = async (optionLabel: string) => {
  const mediaBudgetCard = (
    screen.getByText("Media ingestion budget").closest(".ant-card") as HTMLElement
  )
  const userSelect = (
    within(mediaBudgetCard).getByText("User").closest(".ant-select") as HTMLElement
  )
  fireEvent.mouseDown(userSelect)
  fireEvent.click(await screen.findByText(optionLabel))
}

describe("ServerAdminPage media budget diagnostics", () => {
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

    apiMock.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "single-user"
    })
    apiMock.getSystemStats.mockResolvedValue({
      users: { total: 1, active: 1, admins: 1, verified: 1, new_last_30d: 0 },
      storage: { total_used_mb: 10, total_quota_mb: 100, average_used_mb: 10, max_used_mb: 10 },
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
      limits: {
        jobs_max_concurrent: 4,
        ingestion_bytes_daily_cap: 4000
      },
      usage: {
        jobs_active: 1,
        jobs_remaining: 3,
        ingestion_bytes_daily_used: 1024,
        ingestion_bytes_daily_remaining: 2976
      },
      retry_after: 3832
    })
  })

  it("loads and renders media ingestion budget diagnostics for selected user", async () => {
    renderPage(<ServerAdminPage />)

    await selectMediaBudgetUser("admin (#11)")

    await waitFor(() => {
      expect(apiMock.getMediaIngestionBudgetDiagnostics).toHaveBeenCalledWith({
        userId: 11,
        policyId: "media.default"
      })
    })

    expect(await screen.findByText("Media ingestion budget")).toBeTruthy()
    expect(await screen.findByText("user:11")).toBeTruthy()
    expect((await screen.findAllByText("10 MiB")).length).toBeGreaterThanOrEqual(2)
    expect(await screen.findByText(/128 MiB\s*\/\s*1 GiB/)).toBeTruthy()
    expect(await screen.findByText("3.9 KiB")).toBeTruthy()
    expect(await screen.findByText("2.9 KiB")).toBeTruthy()
    expect(await screen.findByText("~1h 4m")).toBeTruthy()
  })

  it("test_no_diagnostics_preload_on_mount", async () => {
    renderPage(<ServerAdminPage />)

    // Users (and the rest of the reference data) finish loading; the budget
    // card must NOT fetch diagnostics until a user is explicitly selected.
    expect(await screen.findByText("admin@example.com")).toBeTruthy()
    await waitFor(() => {
      expect(apiMock.listAdminUsers).toHaveBeenCalledTimes(1)
    })

    expect(apiMock.getMediaIngestionBudgetDiagnostics).not.toHaveBeenCalled()
  })

  it("test_policy_input_debounced", async () => {
    renderPage(<ServerAdminPage />)

    // Explicit selection settles first (300ms debounce on the fetch effect).
    await selectMediaBudgetUser("admin (#11)")
    await waitFor(() => {
      expect(apiMock.getMediaIngestionBudgetDiagnostics).toHaveBeenCalledTimes(1)
    })

    vi.useFakeTimers()
    const policyInput = screen.getByPlaceholderText("Policy ID")
    for (const partial of ["a", "ab", "abc", "abcd"]) {
      fireEvent.change(policyInput, { target: { value: partial } })
      act(() => {
        vi.advanceTimersByTime(299)
      })
    }

    // Four keystrokes at 299ms each stay inside the 300ms debounce window:
    // only the selection fetch has happened so far.
    expect(apiMock.getMediaIngestionBudgetDiagnostics).toHaveBeenCalledTimes(1)

    act(() => {
      vi.advanceTimersByTime(1)
    })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(0)
    })

    // Crossing 300ms fires exactly one fetch for the settled policy value.
    expect(apiMock.getMediaIngestionBudgetDiagnostics).toHaveBeenCalledTimes(2)
    expect(apiMock.getMediaIngestionBudgetDiagnostics).toHaveBeenLastCalledWith({
      userId: 11,
      policyId: "abcd"
    })
  })

  it("formats oversized legacy storage values as bytes", async () => {
    apiMock.getSystemStats.mockResolvedValueOnce({
      users: { total: 1, active: 1, admins: 1, verified: 1, new_last_30d: 0 },
      storage: {
        total_used_mb: 2147483648,
        total_quota_mb: 4294967296,
        average_used_mb: 1073741824,
        max_used_mb: 2147483648
      },
      sessions: { active: 1, unique_users: 1 }
    })

    renderPage(<ServerAdminPage />)

    expect((await screen.findAllByText("2 GiB")).length).toBeGreaterThanOrEqual(2)
    expect((await screen.findAllByText("4 GiB")).length).toBeGreaterThanOrEqual(1)
    expect((await screen.findAllByText("1 GiB")).length).toBeGreaterThanOrEqual(1)
  })

  it("surfaces timeout messaging and retries system stats fetch", async () => {
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

    renderPage(<ServerAdminPage />)

    const timeoutMessage = await screen.findByText(
      "System statistics took longer than 10 seconds. Retry to try again."
    )
    expect(timeoutMessage).toBeTruthy()

    const retryButton = within(
      timeoutMessage.closest("section") as HTMLElement
    ).getByRole("button", { name: "Retry" })
    fireEvent.click(retryButton)

    await waitFor(() => {
      expect(apiMock.getSystemStats).toHaveBeenCalledTimes(2)
    })

    expect(apiMock.getSystemStats).toHaveBeenNthCalledWith(1, {
      timeoutMs: 10000
    })
    expect(apiMock.getSystemStats).toHaveBeenNthCalledWith(2, {
      timeoutMs: 10000
    })
  }, 15000)
})
