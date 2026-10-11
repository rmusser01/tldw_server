import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  BackendUnavailableModalGate
} from "@web/components/layout/BackendUnavailableModalGate"

describe("BackendUnavailableModalGate", () => {
  it("shows the backend-unreachable modal when no fatal recovery takeover is active", async () => {
    render(
      <BackendUnavailableModalGate
        backendUnavailableDetail={{
          method: "GET",
          path: "/api/v1/llm/models/metadata",
          message: "Failed to fetch",
          source: "direct",
          timestamp: Date.now()
        }}
        fatalBackendRecoveryActive={false}
        isChecking={false}
        onClose={vi.fn()}
        onOpenHealth={vi.fn()}
        onRetry={vi.fn()}
        t={(key: string, fallback?: string) => fallback ?? key}
      />
    )

    expect(
      await screen.findByText("Can't reach your tldw server")
    ).toBeInTheDocument()
  })

  it("preserves backend-unreachable detail and recovery actions in a non-blocking inline alert", () => {
    const onClose = vi.fn()
    const onOpenHealth = vi.fn()
    const onRetry = vi.fn()
    render(
      <BackendUnavailableModalGate
        backendUnavailableDetail={{
          method: "GET",
          path: "/api/v1/llm/models/metadata",
          message: "Failed to fetch",
          source: "direct",
          timestamp: Date.now()
        }}
        fatalBackendRecoveryActive={false}
        isChecking={false}
        onClose={onClose}
        onOpenHealth={onOpenHealth}
        onRetry={onRetry}
        presentation="inline"
        t={(key: string, fallback?: string) => fallback ?? key}
      />
    )

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(
      screen.getByRole("status", { name: "Can't reach your tldw server" })
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Dismiss" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument()
    expect(
      screen.getByText("Failed to fetch (GET /api/v1/llm/models/metadata)")
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(onRetry).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    expect(onOpenHealth).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it.each(["modal", "inline"] as const)("suppresses %s presentation while a fatal backend recovery takeover is active", (presentation) => {
    render(
      <BackendUnavailableModalGate
        backendUnavailableDetail={{
          method: "GET",
          path: "/api/v1/llm/models/metadata",
          message: "Failed to fetch",
          source: "direct",
          timestamp: Date.now()
        }}
        fatalBackendRecoveryActive
        isChecking={false}
        onClose={vi.fn()}
        onOpenHealth={vi.fn()}
        onRetry={vi.fn()}
        presentation={presentation}
        t={(key: string, fallback?: string) => fallback ?? key}
      />
    )

    expect(
      screen.queryByText("Can't reach your tldw server")
    ).not.toBeInTheDocument()
  })

  it.each(["modal", "inline"] as const)("consumes stale %s detail when fatal recovery takes over", (presentation) => {
    const onConsumeHiddenDetail = vi.fn()

    render(
      <BackendUnavailableModalGate
        backendUnavailableDetail={{
          method: "GET",
          path: "/api/v1/llm/models/metadata",
          message: "Failed to fetch",
          source: "direct",
          timestamp: Date.now()
        }}
        fatalBackendRecoveryActive
        isChecking={false}
        onClose={vi.fn()}
        onOpenHealth={vi.fn()}
        onRetry={vi.fn()}
        onConsumeHiddenDetail={onConsumeHiddenDetail}
        presentation={presentation}
        t={(key: string, fallback?: string) => fallback ?? key}
      />
    )

    expect(onConsumeHiddenDetail).toHaveBeenCalledTimes(1)
  })
})
