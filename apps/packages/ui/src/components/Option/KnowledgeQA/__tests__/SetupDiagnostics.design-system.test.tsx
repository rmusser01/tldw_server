// @vitest-environment jsdom

import { render, screen, within } from "@testing-library/react"
import type { ComponentProps } from "react"
import { describe, expect, it, vi } from "vitest"

const registryLabels = vi.hoisted(() => ({
  blocked: "Registry Blocked",
  ready: "Registry Ready",
}))

vi.mock("@/design-system", () => ({
  BLOCKED_STATE_LABEL: registryLabels.blocked,
  READY_STATE_LABEL: registryLabels.ready,
}))

import { KnowledgeQASetupDiagnostics } from "../SetupDiagnostics"

const noop = vi.fn()

const renderDiagnostics = (
  overrides: Partial<ComponentProps<typeof KnowledgeQASetupDiagnostics>> = {}
) =>
  render(
    <KnowledgeQASetupDiagnostics
      connection={{
        serverUrl: "http://127.0.0.1:8000",
        configStep: "health",
        errorKind: "none",
        lastError: null,
        lastStatusCode: null,
        isChecking: false,
      }}
      uxState="connected_ok"
      retryCountdownSeconds={30}
      onOpenSetup={noop}
      onOpenSettings={noop}
      onOpenDiagnostics={noop}
      onRetryConnection={noop}
      onRetrySearch={noop}
      onRetrySync={noop}
      {...overrides}
    />
  )

describe("KnowledgeQASetupDiagnostics design-system labels", () => {
  it("uses the design-system registry label for complete diagnostics", () => {
    renderDiagnostics()

    expect(
      within(screen.getByTestId("knowledge-setup-check-server-url")).getByText(
        registryLabels.ready
      )
    ).toBeInTheDocument()
    expect(
      within(screen.getByTestId("knowledge-setup-check-backend")).getByText(
        registryLabels.ready
      )
    ).toBeInTheDocument()
  })

  it("uses the design-system registry label for blocked diagnostics", () => {
    renderDiagnostics({
      connection: {
        serverUrl: "http://127.0.0.1:8000",
        configStep: "health",
        errorKind: "unreachable",
        lastError:
          "Absolute URL requests are blocked unless the request origin is explicitly allowlisted.",
        lastStatusCode: 400,
        isChecking: false,
      },
      uxState: "error_unreachable",
      extensionFailureState: "api_allowlist_blocked",
    })

    expect(
      within(screen.getByTestId("knowledge-setup-check-browser-access")).getByText(
        registryLabels.blocked
      )
    ).toBeInTheDocument()
    expect(
      within(screen.getByTestId("knowledge-setup-check-backend")).getByText(
        registryLabels.blocked
      )
    ).toBeInTheDocument()
  })
})
