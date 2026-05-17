import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import * as LocaleDiagnostics from "../LocaleJsonDiagnostics"

describe("LocaleJsonDiagnostics design-system contract", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("renders locale JSON parse errors through the design-system Alert primitive", () => {
    expect(LocaleDiagnostics.LocaleJsonDiagnosticsPanel).toBeTypeOf("function")

    render(
      <LocaleDiagnostics.LocaleJsonDiagnosticsPanel
        issues={[
          {
            path: "../../assets/locale/en/common.json",
            message: "Expected property name or '}' in JSON at position 42",
            line: 3,
            column: 12,
          },
        ]}
      />
    )

    const title = screen.getByText("Locale JSON errors detected")
    const alert = title.closest('[data-ds-component="Alert"]')

    expect(alert).toBeInTheDocument()
    expect(alert).toHaveAttribute("role", "alert")
    expect(screen.getByText("../../assets/locale/en/common.json")).toBeInTheDocument()
    expect(screen.getByText(/line 3, col 12/)).toBeInTheDocument()
    expect(screen.getByText(/Expected property name/)).toBeInTheDocument()
  })

  it("keeps issue keys unique when one locale file reports multiple parse errors", () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)

    render(
      <LocaleDiagnostics.LocaleJsonDiagnosticsPanel
        issues={[
          {
            path: "../../assets/locale/en/common.json",
            message: "Expected property name or '}' in JSON at position 42",
            line: 3,
            column: 12,
          },
          {
            path: "../../assets/locale/en/common.json",
            message: "Expected ',' or '}' after property value in JSON at position 48",
            line: 3,
            column: 18,
          },
        ]}
      />
    )

    expect(
      consoleError.mock.calls.some((call) =>
        call.some(
          (value) =>
            typeof value === "string" &&
            value.includes("Encountered two children with the same key")
        )
      )
    ).toBe(false)
  })
})
