// @vitest-environment jsdom
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import {
  ACPPlaygroundRecovery,
  normalizeACPHealthSnapshot,
  shouldShowAcpPlaygroundRecovery,
  type ACPHealthSnapshot
} from "../ACPPlaygroundRecovery"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

describe("ACPPlaygroundRecovery", () => {
  it("does not render recovery while ACP health is loading, healthy, or degraded", () => {
    expect(
      shouldShowAcpPlaygroundRecovery({
        healthData: null,
        isHealthLoading: true
      })
    ).toBe(false)
    expect(
      shouldShowAcpPlaygroundRecovery({
        healthData: { overall: "healthy" },
        isHealthLoading: false
      })
    ).toBe(false)
    expect(
      shouldShowAcpPlaygroundRecovery({
        healthData: { overall: "degraded" },
        isHealthLoading: false
      })
    ).toBe(false)
  })

  it("normalizes non-OK ACP health responses as unavailable", () => {
    expect(
      normalizeACPHealthSnapshot(
        { overall: "healthy", detail: "ACP runner disabled" },
        { overall: "unavailable", status: 503 }
      )
    ).toEqual(
      expect.objectContaining({
        overall: "unavailable",
        status: 503,
        rawMessage: "ACP runner disabled"
      })
    )
  })

  it("renders shared recovery with diagnostics and retry when ACP health is unavailable", async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()
    const healthData: ACPHealthSnapshot = {
      overall: "unavailable",
      status: 503,
      rawMessage: "ACP runner disabled"
    }

    render(
      <ACPPlaygroundRecovery
        healthData={healthData}
        isHealthLoading={false}
        onRetry={onRetry}
        serverUrl="http://127.0.0.1:8000"
      />
    )

    const recovery = screen.getByTestId("acp-playground-capability-recovery")

    expect(recovery).toHaveAttribute("data-ds-component", "RecoveryCallout")
    expect(
      screen.getByRole("heading", {
        name: "ACP Playground is unavailable on this server"
      })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "The connected server does not advertise ACP session orchestration."
      )
    ).toBeInTheDocument()
    const diagnostics = within(recovery).getByLabelText("Diagnostics")
    expect(diagnostics).toHaveTextContent("GET")
    expect(diagnostics).toHaveTextContent("[server-endpoint]")
    expect(diagnostics).toHaveTextContent("[server-url]")
    expect(diagnostics).toHaveTextContent("503")
    expect(diagnostics).toHaveTextContent("ACP runner disabled")
    expect(diagnostics).not.toHaveTextContent("/api/v1/acp/health")
    expect(diagnostics).not.toHaveTextContent("http://127.0.0.1:8000")

    await user.click(screen.getByRole("button", { name: "Try again" }))

    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})
