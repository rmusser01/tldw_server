import React from "react"
import { render, screen } from "@testing-library/react"
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

  it("can render backend-unreachable detail as a non-blocking inline alert", () => {
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
  })

  it("suppresses the modal while a fatal backend recovery takeover is active", () => {
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
        t={(key: string, fallback?: string) => fallback ?? key}
      />
    )

    expect(
      screen.queryByText("Can't reach your tldw server")
    ).not.toBeInTheDocument()
  })

  it("consumes stale modal detail when fatal recovery takes over", () => {
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
        t={(key: string, fallback?: string) => fallback ?? key}
      />
    )

    expect(onConsumeHiddenDetail).toHaveBeenCalledTimes(1)
  })
})
