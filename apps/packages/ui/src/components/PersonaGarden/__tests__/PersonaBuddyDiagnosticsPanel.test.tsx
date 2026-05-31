import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"

import { PersonaBuddyDiagnosticsPanel } from "../PersonaBuddyDiagnosticsPanel"

describe("PersonaBuddyDiagnosticsPanel", () => {
  it("renders compact diagnostics rows", () => {
    render(
      <PersonaBuddyDiagnosticsPanel
        diagnostics={{
          state: "degraded",
          title: "Persona Buddy degraded",
          message: "Visual pack needs attention.",
          rows: [
            { label: "Persona", value: "Ada", state: "healthy" },
            { label: "Visual pack", value: "Missing manifest", state: "degraded" }
          ]
        }}
      />
    )

    expect(screen.getByTestId("persona-buddy-diagnostics")).toBeInTheDocument()
    expect(screen.getByText("Persona Buddy degraded")).toBeInTheDocument()
    expect(screen.getByText("Visual pack")).toBeInTheDocument()
    expect(screen.getByText("Missing manifest")).toBeInTheDocument()
  })

  it("maps recovering diagnostics to the design-system retrying state", () => {
    render(
      <PersonaBuddyDiagnosticsPanel
        diagnostics={{
          state: "recovering",
          title: "Persona Buddy recovering",
          message: "Live session is reconnecting.",
          rows: [{ label: "Live session", value: "Reconnecting", state: "recovering" }]
        }}
      />
    )

    expect(screen.getByText("Retrying")).toBeInTheDocument()
    expect(screen.getByText("Live session")).toBeInTheDocument()
  })
})
