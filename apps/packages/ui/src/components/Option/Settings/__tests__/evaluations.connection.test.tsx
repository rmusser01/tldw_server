import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { EvaluationsSettings } from "../evaluations"

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node =
    typeof text === "string"
      ? screen.getByText(text, { exact: false })
      : screen.getByText(text)

  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

const connectionState = {
  online: true,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  navigate: vi.fn()
}

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({
    data: {
      defaultEvalType: "response_quality",
      defaultTargetModel: "gpt-4o-mini",
      defaultRunConfig: "",
      defaultDatasetId: null,
      defaultSpecByType: {}
    },
    isLoading: false
  }),
  useMutation: (options?: { onSuccess?: (value: unknown) => void }) => ({
    mutateAsync: vi.fn(async () => {
      const value = {
        ok: true,
        data: {
          tier: "free",
          usage: { evaluations_today: 1 },
          limits: { evaluations_per_day: 10 }
        }
      }
      options?.onSuccess?.(value)
      return value
    }),
    isPending: false
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => connectionState.navigate
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      return fallbackOrOptions?.defaultValue ?? _key
    }
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => connectionState.online
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => ({
    uxState: connectionState.uxState,
    hasCompletedFirstRun: true
  })
}))

vi.mock("@/services/evaluations", () => ({
  getRateLimits: vi.fn()
}))

vi.mock("@/services/evaluations-settings", () => ({
  getEvaluationDefaults: vi.fn(),
  setEvaluationDefaults: vi.fn(),
  setDefaultSpecForType: vi.fn()
}))

describe("EvaluationsSettings connection warning", () => {
  beforeEach(() => {
    connectionState.online = true
    connectionState.uxState = "connected_ok"
    connectionState.navigate.mockReset()
  })

  it("shows credential guidance when auth is missing", () => {
    connectionState.online = false
    connectionState.uxState = "error_auth"

    render(<EvaluationsSettings />)

    expect(
      screen.getByText("Add your credentials to test Evaluations.")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Add your credentials to test Evaluations.")

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(connectionState.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance when setup is incomplete", () => {
    connectionState.online = false
    connectionState.uxState = "unconfigured"

    render(<EvaluationsSettings />)

    expect(
      screen.getByText("Finish setup to test Evaluations.")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Finish setup to test Evaluations.")

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(connectionState.navigate).toHaveBeenCalledWith("/")
  })

  it("shows unreachable guidance when the server is unreachable", () => {
    connectionState.online = false
    connectionState.uxState = "error_unreachable"

    render(<EvaluationsSettings />)

    expect(
      screen.getByText("Can't reach your tldw server right now.")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Can't reach your tldw server right now.")

    fireEvent.click(screen.getByRole("button", { name: "Health & diagnostics" }))
    expect(connectionState.navigate).toHaveBeenCalledWith("/settings/health")
  })

  it("shows generic offline guidance through the design-system alert", () => {
    connectionState.online = false
    connectionState.uxState = "connected_ok"

    render(<EvaluationsSettings />)

    expect(
      screen.getByText("Connect to your tldw server to test Evaluations.")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Connect to your tldw server to test Evaluations.")
  })

  it("renders API test results through the design-system alert", async () => {
    render(<EvaluationsSettings />)

    fireEvent.click(
      screen.getByRole("button", { name: "Test Evaluations API" })
    )

    expect(
      await screen.findByText("Evaluations API reachable")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Evaluations API reachable")
    expectDesignSystemAlert("free")
  })

  it("suppresses the warning while connection checks are still testing", () => {
    connectionState.online = false
    connectionState.uxState = "testing"

    render(<EvaluationsSettings />)

    expect(
      screen.queryByText("Add your credentials to test Evaluations.")
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText("Connect to your tldw server to test Evaluations.")
    ).not.toBeInTheDocument()
  })
})
