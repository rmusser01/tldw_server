import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StatusBanner } from "../StatusBanner"

describe("StatusBanner", () => {
  it("renders loading status through the design-system LoadingState primitive", () => {
    render(<StatusBanner state="loading" loading />)

    const loadingText = screen.getByText("Loading status...")

    expect(loadingText.closest('[data-ds-component="LoadingState"]')).toBeTruthy()
  })

  it("sanitizes user-facing admin error details", () => {
    render(
      <StatusBanner
        state="inactive"
        error="Request failed: 503 (GET /api/v1/admin/mlx/status) config=/Users/dev/.config/tldw/config.txt"
      />
    )

    expect(screen.getByText("Status Error")).toBeTruthy()
    const fullText = document.body.textContent || ""
    expect(fullText).toContain("[admin-endpoint]")
    expect(fullText).toContain("[redacted-path]")
    expect(fullText).not.toContain("/api/v1/admin/mlx/status")
    expect(fullText).not.toContain("/Users/dev/.config/tldw/config.txt")
  })

  it("renders errors through the design-system Alert primitive and preserves retry", async () => {
    const user = userEvent.setup()
    const onRefresh = vi.fn()

    render(
      <StatusBanner
        state="inactive"
        error="Failed to load status"
        onRefresh={onRefresh}
      />
    )

    const title = screen.getByText("Status Error")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).toBeTruthy()

    await user.click(screen.getByRole("button", { name: "Retry" }))

    expect(onRefresh).toHaveBeenCalledTimes(1)
  })
})
