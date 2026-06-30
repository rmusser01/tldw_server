import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import ConnectionProblemBanner from "../ConnectionProblemBanner"

describe("ConnectionProblemBanner", () => {
  it("renders connection recovery through the shared RecoveryCallout primitive", () => {
    const onPrimaryAction = vi.fn()
    const onSecondaryAction = vi.fn()
    const onRetry = vi.fn()

    render(
      <ConnectionProblemBanner
        badgeLabel="Not connected"
        title="Connect to use Notes"
        description="This view needs a connected server."
        examples={["Add your server URL.", "Check your API key."]}
        primaryActionLabel="Set up server"
        onPrimaryAction={onPrimaryAction}
        secondaryActionLabel="Health & diagnostics"
        onSecondaryAction={onSecondaryAction}
        retryActionLabel="Retry connection"
        onRetry={onRetry}
      />
    )

    const banner = screen.getByText("Connect to use Notes").closest("section")

    expect(banner).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByText("Not connected")).toBeInTheDocument()
    expect(screen.getByText("This view needs a connected server.")).toBeInTheDocument()
    expect(screen.getByText("Add your server URL.")).toBeInTheDocument()
    expect(screen.getByText("Check your API key.")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Set up server" }))
    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }))

    expect(onPrimaryAction).toHaveBeenCalledTimes(1)
    expect(onSecondaryAction).toHaveBeenCalledTimes(1)
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it("keeps the retry action disabled when retrying is unavailable", () => {
    render(
      <ConnectionProblemBanner
        title="Connect to use Review"
        primaryActionLabel="Open Settings"
        retryActionLabel="Retry connection"
        onRetry={vi.fn()}
        retryDisabled
      />
    )

    expect(screen.getByRole("button", { name: "Retry connection" })).toBeDisabled()
  })
})
