import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { InspectorRail } from "../InspectorRail"

describe("InspectorRail", () => {
  it("shows real scope and staged source state", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={2}
        stagedSourceTitles={["Operator Notes", "Research Clip"]}
        selectedModelLabel="gpt-test"
        selectedPersonaLabel="Analyst"
        backendAvailable
        streaming={false}
      />
    )

    expect(screen.getByText("Default workspace")).toBeInTheDocument()
    expect(screen.getByText("Operator Notes")).toBeInTheDocument()
    expect(screen.getByText("gpt-test")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
  })

  it("labels inactive v1 panels honestly", () => {
    render(
      <InspectorRail
        scopeLabel="No workspace"
        stagedSourceCount={0}
        stagedSourceTitles={[]}
        selectedModelLabel="No model selected"
        selectedPersonaLabel={null}
        backendAvailable={false}
        streaming={false}
      />
    )

    expect(screen.getByText("Not configured")).toBeInTheDocument()
    expect(screen.getByText("No active task")).toBeInTheDocument()
    expect(screen.getByText("Server unavailable")).toBeInTheDocument()
  })
})
