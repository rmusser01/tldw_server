import { render, screen } from "@testing-library/react"
import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { describe, expect, it, vi } from "vitest"
import { InspectorRail } from "../InspectorRail"

vi.mock("@/design-system", async (importActual) => {
  const actual = await importActual<typeof import("@/design-system")>()

  return {
    ...actual,
    getDesignSystemState: vi.fn(
      (key: Parameters<typeof actual.getDesignSystemState>[0]) => {
        const state = actual.getDesignSystemState(key)

        return {
          ...state,
          label: key === "ready" ? "Ready via registry" : state.label
        }
      }
    )
  }
})

describe("InspectorRail", () => {
  it("shows real scope and staged source state", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={2}
        stagedSources={[
          { sourceId: "source-1", title: "Operator Notes" },
          { sourceId: "source-2", title: "Research Clip" }
        ]}
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
    expect(screen.getByText("Ready via registry")).toBeInTheDocument()
  })

  it("labels inactive v1 panels honestly", () => {
    render(
      <InspectorRail
        scopeLabel="No workspace"
        stagedSourceCount={0}
        stagedSources={[]}
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

  it("uses stable source metadata for duplicate staged source titles", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={2}
        stagedSources={[
          { sourceId: "source-1", title: "Meeting Notes" },
          { sourceId: "source-2", title: "Meeting Notes" }
        ]}
        selectedModelLabel="gpt-test"
        selectedPersonaLabel={null}
        backendAvailable
        streaming={false}
      />
    )

    expect(screen.getAllByText("Meeting Notes")).toHaveLength(2)
  })

  it("keeps the inspector API structured without title-only fallbacks", () => {
    const source = readFileSync(resolve(__dirname, "../InspectorRail.tsx"), "utf8")

    expect(source).not.toContain("stagedSourceTitles")
  })
})
