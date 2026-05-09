import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { getDesignSystemState } from "@/design-system"
import { SyncStatusBadge } from "../SyncStatusBadge"
import type { PromptSyncStatus } from "@/db/dexie/types"

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

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(actual.getDesignSystemState)
  }
})

vi.mock("antd", () => ({
  Tag: ({ children, color }: any) => (
    <span data-testid="antd-tag" data-color={color}>
      {children}
    </span>
  ),
  Tooltip: ({ children }: any) => <>{children}</>
}))

const STATUS_CASES = [
  ["local", "Local", "empty", "secondary"],
  ["synced", "Synced", "ready", "success"],
  ["pending", "Pending", "degraded", "warning"],
  ["conflict", "Conflict", "blocked", "danger"]
] satisfies Array<[PromptSyncStatus, string, string, string]>

describe("SyncStatusBadge", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it.each(STATUS_CASES)(
    "renders %s through the canonical state registry",
    (syncStatus, label, stateKey, variant) => {
      const { container } = render(
        <SyncStatusBadge syncStatus={syncStatus} sourceSystem="workspace" />
      )

      const badge = container.querySelector('[data-ds-component="Badge"]')

      expect(screen.getByText(label)).toBeInTheDocument()
      expect(getDesignSystemState).toHaveBeenCalledWith(stateKey)
      expect(badge).toHaveAttribute("data-ds-size", "sm")
      expect(badge).toHaveAttribute("data-ds-variant", variant)
      expect(screen.queryByTestId("antd-tag")).not.toBeInTheDocument()
    }
  )

  it("falls back to the canonical empty state for missing sync status", () => {
    const { container } = render(<SyncStatusBadge sourceSystem="workspace" />)
    const badge = container.querySelector('[data-ds-component="Badge"]')

    expect(screen.getByText("Local")).toBeInTheDocument()
    expect(getDesignSystemState).toHaveBeenCalledWith("empty")
    expect(badge).toHaveAttribute("data-ds-variant", "secondary")
  })

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
