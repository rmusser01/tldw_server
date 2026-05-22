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
