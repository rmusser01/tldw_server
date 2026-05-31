import { render, screen, within } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { ConfigurationErrorScreen } from "../ConfigurationErrorScreen"

describe("ConfigurationErrorScreen", () => {
  it("renders loopback API configuration errors with setup state diagnostics", () => {
    render(
      <ConfigurationErrorScreen
        issue={{
          kind: "loopback_api_not_browser_reachable",
          apiOrigin: "http://127.0.0.1:8000",
          pageOrigin: "http://192.168.1.20:3000"
        }}
      />
    )

    expect(screen.getByText("Setup required")).toBeInTheDocument()
    expect(
      screen.getByRole("heading", {
        name: "WebUI networking configuration error"
      })
    ).toBeInTheDocument()
    const diagnostics = screen.getByLabelText("Diagnostics")
    expect(diagnostics).toBeInTheDocument()
    expect(within(diagnostics).getByText("API origin")).toBeInTheDocument()
    expect(
      within(diagnostics).getByText("http://127.0.0.1:8000")
    ).toBeInTheDocument()
    expect(within(diagnostics).getByText("Page origin")).toBeInTheDocument()
    expect(
      within(diagnostics).getByText("http://192.168.1.20:3000")
    ).toBeInTheDocument()
  })

  it("returns null for unknown issue kinds", () => {
    const { container } = render(
      <ConfigurationErrorScreen
        issue={
          { kind: "unknown_issue" } as Parameters<
            typeof ConfigurationErrorScreen
          >[0]["issue"]
        }
      />
    )

    expect(container).toBeEmptyDOMElement()
  })
})
