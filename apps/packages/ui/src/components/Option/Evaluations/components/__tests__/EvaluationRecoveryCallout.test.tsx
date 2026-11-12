import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import {
  EvaluationRecoveryCallout,
  getEvaluationRecoveryDetail
} from "../EvaluationRecoveryCallout"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) =>
      key === "evaluations:recoveryDefaultMessage"
        ? "Translated recovery default"
        : key === "common:diagnostics"
          ? "Diagnostics"
        : options?.defaultValue || key
  })
}))

describe("EvaluationRecoveryCallout", () => {
  it("renders the default recovery message through the evaluations i18n hook", () => {
    const { container } = render(
      <EvaluationRecoveryCallout
        title="Unable to load evaluations"
        endpoint="/api/v1/evaluations"
      />
    )

    expect(screen.getByText("Translated recovery default")).toBeInTheDocument()
    expect(
      screen.getByText("/api/v1/evaluations")
    ).toBeInTheDocument()
    expect(
      container.querySelector('[data-ds-component="RecoveryCallout"]')
    ).toBeInTheDocument()
  })

  it("classifies forbidden responses with the shared permission denied state", () => {
    const { container } = render(
      <EvaluationRecoveryCallout
        title="Unable to load evaluations"
        message="Use an admin account to view evaluations."
        endpoint="/api/v1/evaluations"
        response={{
          ok: false,
          status: 403,
          error: { detail: "Missing evals:read scope" }
        }}
      />
    )
    const recovery = container.querySelector(
      '[data-ds-component="RecoveryCallout"]'
    )

    expect(recovery).toBeInTheDocument()
    expect(screen.getByText("Permission denied")).toBeInTheDocument()
    expect(screen.getByText("Unable to load evaluations")).toBeInTheDocument()
    expect(
      screen.getByText("Use an admin account to view evaluations.")
    ).toBeInTheDocument()
    const diagnostics = screen.getByLabelText("Diagnostics")
    expect(diagnostics).toHaveTextContent("/api/v1/evaluations")
    expect(diagnostics).toHaveTextContent(
      "HTTP 403: Missing evals:read scope"
    )
  })

  it.each([
    [
      "dataset response",
      {
        response: {
          ok: false,
          status: 403,
          error: { detail: "Permission denied: missing evals.read" }
        }
      }
    ],
    [
      "recipe error",
      { error: new Error("HTTP 403: Permission denied: missing evals.read") }
    ]
  ] as const)(
    "explains account access for a forbidden %s without suggesting connection repair",
    (_label, props) => {
      render(
        <EvaluationRecoveryCallout
          title="Unable to load evaluations"
          endpoint="/api/v1/evaluations"
          {...props}
        />
      )

      expect(screen.getByText("Permission denied")).toBeInTheDocument()
      expect(
        screen.getByText(/account with access to Evaluations API/i)
      ).toBeInTheDocument()
      expect(
        screen.queryByText("Translated recovery default")
      ).not.toBeInTheDocument()
    }
  )

  it.each([
    [401, "Sign in required"],
    [404, "Unavailable"],
    [500, "Error"]
  ] as const)("preserves HTTP %s from query error response metadata", (status, label) => {
    const error = Object.assign(new Error("Request rejected"), {
      resp: { ok: false, status, error: "Request rejected" }
    })
    render(
      <EvaluationRecoveryCallout
        title="Unable to load recipes"
        endpoint="/api/v1/evaluations/recipes"
        error={error}
      />
    )

    expect(screen.getByText(label)).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      `HTTP ${status}: Request rejected`
    )
    expect(
      screen.queryByText(/account with access to Evaluations API/i)
    ).not.toBeInTheDocument()
  })

  it("keeps the explicit response authoritative over query error metadata", () => {
    const error = Object.assign(new Error("Permission denied: missing evals.read"), {
      resp: { ok: false, status: 403, error: "Permission denied: missing evals.read" }
    })
    render(
      <EvaluationRecoveryCallout
        title="Unable to load recipes"
        endpoint="/api/v1/evaluations/recipes"
        error={error}
        response={{ ok: false, status: 500, error: "Current server failure" }}
      />
    )

    expect(screen.getByText("Error")).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "HTTP 500: Current server failure"
    )
    expect(screen.queryByText("Permission denied")).not.toBeInTheDocument()
  })

  it("extracts FastAPI detail fields from backend error payloads", () => {
    expect(
      getEvaluationRecoveryDetail(undefined, {
        ok: false,
        status: 422,
        error: { detail: "Dataset rows are invalid" }
      })
    ).toBe("HTTP 422: Dataset rows are invalid")

    expect(
      getEvaluationRecoveryDetail(undefined, {
        ok: false,
        status: 422,
        error: { detail: [{ msg: "Field required" }] }
      })
    ).toBe("HTTP 422: Field required")
  })

  it("does not duplicate HTTP status prefixes from the request layer", () => {
    expect(
      getEvaluationRecoveryDetail(undefined, {
        ok: false,
        status: 404,
        error: "HTTP 404"
      })
    ).toBe("HTTP 404")
  })
})
