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
