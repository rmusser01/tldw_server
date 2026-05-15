// @vitest-environment jsdom
import { describe, expect, it, vi, beforeEach } from "vitest"
import { render, screen, fireEvent } from "@testing-library/react"

// Mock the hook to control the return value
const mockQuota = {
  level: "ok" as "ok" | "warning" | "exceeded",
  ratio: 0,
  usedBytes: 0,
  budgetBytes: 5 * 1024 * 1024,
  availableBytes: 5 * 1024 * 1024,
  canWrite: () => true,
  refresh: vi.fn()
}

vi.mock("@/hooks/useStorageQuota", () => ({
  useStorageQuota: () => mockQuota
}))

import { StorageQuotaBanner } from "../StorageQuotaBanner"

describe("StorageQuotaBanner", () => {
  beforeEach(() => {
    sessionStorage.clear()
    mockQuota.level = "ok"
    mockQuota.ratio = 0
  })

  it("renders nothing when level is ok", () => {
    const { container } = render(<StorageQuotaBanner />)
    expect(container.innerHTML).toBe("")
  })

  it("renders warning banner at 80%+", () => {
    mockQuota.level = "warning"
    mockQuota.ratio = 0.82
    mockQuota.usedBytes = 4.1 * 1024 * 1024
    render(<StorageQuotaBanner />)
    const banner = screen.getByTestId("storage-quota-banner-warning")
    expect(banner).toBeInTheDocument()
    expect(banner).toHaveAttribute("data-ds-component", "Alert")
    expect(screen.getByText("Storage getting full")).toBeInTheDocument()
    expect(screen.getByText(/82%/)).toBeInTheDocument()
  })

  it("renders exceeded banner at 95%+", () => {
    mockQuota.level = "exceeded"
    mockQuota.ratio = 0.97
    mockQuota.usedBytes = 4.85 * 1024 * 1024
    render(<StorageQuotaBanner />)
    const banner = screen.getByTestId("storage-quota-banner-exceeded")
    expect(banner).toBeInTheDocument()
    expect(banner).toHaveAttribute("data-ds-component", "Alert")
    expect(screen.getByText("Storage nearly full")).toBeInTheDocument()
  })

  it("exceeded banner cannot be dismissed", () => {
    mockQuota.level = "exceeded"
    mockQuota.ratio = 0.97
    render(<StorageQuotaBanner />)
    // Alert with type=error and no closable prop should not have close button
    expect(screen.queryByRole("button", { name: /close/i })).not.toBeInTheDocument()
  })

  it("persists warning dismissal for the session", () => {
    mockQuota.level = "warning"
    mockQuota.ratio = 0.85
    mockQuota.usedBytes = 4.25e6
    mockQuota.budgetBytes = 5e6
    mockQuota.availableBytes = 0.75e6
    mockQuota.canWrite = () => true

    const { unmount } = render(<StorageQuotaBanner />)
    const closeBtn = screen.getByRole("button", { name: /close/i })
    fireEvent.click(closeBtn)

    // Banner should disappear
    expect(screen.queryByTestId("storage-quota-banner-warning")).toBeNull()
    unmount()

    // Re-render — should still be dismissed (sessionStorage persists)
    render(<StorageQuotaBanner />)
    expect(screen.queryByTestId("storage-quota-banner-warning")).toBeNull()
  })
})
