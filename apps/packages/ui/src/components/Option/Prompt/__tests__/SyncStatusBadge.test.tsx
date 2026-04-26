import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { SyncStatusBadge } from "../SyncStatusBadge"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallbackOrOptions?: any, maybeOptions?: any) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue ?? _key
      }
      if (maybeOptions && typeof maybeOptions === "object") {
        return maybeOptions.defaultValue ?? _key
      }
      return _key
    }
  })
}))

vi.mock("antd", () => ({
  Tag: ({ children }: any) => <span>{children}</span>,
  Tooltip: ({ children }: any) => <>{children}</>
}))

describe("SyncStatusBadge", () => {
  it("invokes onClick when rendered as compact interactive badge", async () => {
    const user = userEvent.setup()
    const onClick = vi.fn()

    render(
      <SyncStatusBadge
        syncStatus="conflict"
        sourceSystem="workspace"
        compact
        onClick={onClick}
      />
    )

    await user.click(screen.getByRole("button", { name: "Resolve conflict" }))
    expect(onClick).toHaveBeenCalledTimes(1)
  })

  it("renders non-interactive compact badge when onClick is not provided", () => {
    render(
      <SyncStatusBadge
        syncStatus="pending"
        sourceSystem="workspace"
        compact
      />
    )

    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("shows retry button when syncStatus is pending and onRetry is provided", async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()

    render(
      <SyncStatusBadge
        syncStatus="pending"
        sourceSystem="workspace"
        onRetry={onRetry}
      />
    )

    const retryButton = screen.getByTestId("sync-retry-button")
    expect(retryButton).toBeInTheDocument()

    await user.click(retryButton)
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it("does not show retry button when syncStatus is not pending", () => {
    render(
      <SyncStatusBadge
        syncStatus="synced"
        sourceSystem="workspace"
        onRetry={vi.fn()}
      />
    )

    expect(screen.queryByTestId("sync-retry-button")).not.toBeInTheDocument()
  })

  it("does not show retry button when onRetry is not provided", () => {
    render(
      <SyncStatusBadge
        syncStatus="pending"
        sourceSystem="workspace"
      />
    )

    expect(screen.queryByTestId("sync-retry-button")).not.toBeInTheDocument()
  })

  it("shows retry button in compact mode for pending status", async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()

    render(
      <SyncStatusBadge
        syncStatus="pending"
        sourceSystem="workspace"
        compact
        onRetry={onRetry}
      />
    )

    const retryButton = screen.getByTestId("sync-retry-button")
    expect(retryButton).toBeInTheDocument()

    await user.click(retryButton)
    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})
