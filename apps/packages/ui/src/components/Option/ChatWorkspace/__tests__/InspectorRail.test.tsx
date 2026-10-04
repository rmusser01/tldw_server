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
  it("requires a model for idle readiness while leaving persona optional", () => {
    const props = {
      scopeLabel: "Workspace",
      stagedSourceCount: 0,
      stagedSources: [],
      selectedPersonaLabel: null,
      assistantSource: "none" as const,
      backendAvailable: true,
      workspaceReady: true,
      streaming: false
    }
    const { rerender } = render(
      <InspectorRail {...props} selectedModelLabel="No model selected" hasModelSelected={false} />
    )
    expect(screen.getByText("Select a model")).toBeInTheDocument()
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
    expect(screen.getByText("Choose a model before sending.")).toBeInTheDocument()
    rerender(<InspectorRail {...props} selectedModelLabel="Selected model" hasModelSelected />)
    expect(screen.getByText("Ready via registry")).toBeInTheDocument()
    expect(screen.queryByText("Choose a model before sending.")).not.toBeInTheDocument()
  })

  it.each([
    [{ connectionMode: "demo" }, "Demo mode - not live"],
    [{ connectionMode: "bypass" }, "Offline bypass - not verified"],
    [{ backendAvailable: false }, "Server unavailable"],
    [{ workspaceReady: false }, "Loading workspace context"],
    [{ historyLoading: true }, "Loading chat history"],
    [{ historyLoadError: "Unavailable" }, "Chat history unavailable"],
    [{ streaming: true }, "Streaming"],
    [{ sending: true }, "Sending"],
    [{ sendError: "Failed" }, "Send failed"]
  ] as const)("retains runtime precedence with no model for %j", (runtime, label) => {
    render(
      <InspectorRail
        scopeLabel="Workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="No model selected"
        hasModelSelected={false}
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable
        workspaceReady
        streaming={false}
        {...runtime}
      />
    )
    expect(screen.getByText(label)).toBeInTheDocument()
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
  })

  it.each([
    [{ connectionMode: "demo" }, "Demo mode - not live"],
    [{ connectionMode: "bypass" }, "Offline bypass - not verified"],
    [{ historyLoading: true }, "Loading chat history"],
    [{ historyLoadError: "History unavailable" }, "Chat history unavailable"],
    [{ sending: true }, "Sending"]
  ] as const)("does not report Ready for %j", (runtime, label) => {
    render(
      <InspectorRail
        scopeLabel="Workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="global-model"
        hasModelSelected
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable
        workspaceReady
        streaming={false}
        {...runtime}
      />
    )
    expect(screen.getByText(label)).toBeInTheDocument()
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
  })

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
        hasModelSelected
        selectedPersonaLabel="Analyst"
        assistantSource="explicit"
        backendAvailable
        workspaceReady
        streaming={false}
      />
    )

    expect(screen.getByText("Default workspace")).toBeInTheDocument()
    expect(screen.getByText("Operator Notes")).toBeInTheDocument()
    expect(screen.getByText("gpt-test")).toBeInTheDocument()
    expect(screen.getByText("Analyst")).toBeInTheDocument()
    expect(screen.getByText("Explicit persona")).toBeInTheDocument()
    expect(screen.getByText("Ready via registry")).toBeInTheDocument()
  })

  it("explains backend recovery without inactive placeholder panels", () => {
    render(
      <InspectorRail
        scopeLabel="No workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="No model selected"
        hasModelSelected={false}
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable={false}
        streaming={false}
        workspaceReady
      />
    )

    expect(screen.getByText("Server unavailable")).toBeInTheDocument()
    expect(
      screen.getByText("Reconnect to the server before sending workspace chat.")
    ).toBeInTheDocument()
    expect(screen.queryByText("Approvals")).not.toBeInTheDocument()
    expect(screen.queryByText("Task Progress")).not.toBeInTheDocument()
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
        hasModelSelected
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable
        workspaceReady
        streaming={false}
      />
    )

    expect(screen.getAllByText("Meeting Notes")).toHaveLength(2)
  })

  it("keeps the inspector API structured without title-only fallbacks", () => {
    const source = readFileSync(resolve(__dirname, "../InspectorRail.tsx"), "utf8")

    expect(source).not.toContain("stagedSourceTitles")
  })

  it("labels an inherited workspace persona separately from explicit selection", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="gpt-test"
        hasModelSelected
        selectedPersonaLabel="Workspace Analyst"
        assistantSource="workspace"
        backendAvailable
        workspaceReady
        streaming={false}
      />
    )

    expect(screen.getByText("Workspace Analyst")).toBeInTheDocument()
    expect(screen.getByText("Inherited from workspace")).toBeInTheDocument()
  })

  it("shows unavailable workspace defaults without duplicating persona content", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="gpt-test"
        hasModelSelected
        selectedPersonaLabel={null}
        assistantSource="unavailable"
        workspaceAssistantDegradedReason="persona_deleted"
        backendAvailable
        streaming={false}
        workspaceReady
      />
    )

    expect(screen.getByText("Workspace default unavailable")).toBeInTheDocument()
    expect(screen.getByText("Persona deleted")).toBeInTheDocument()
    expect(screen.queryByText("No persona selected")).not.toBeInTheDocument()
  })

  it("shows workspace hydration as not ready even while the server is connected", () => {
    render(
      <InspectorRail
        scopeLabel="Workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="gpt-test"
        hasModelSelected
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable
        streaming={false}
        workspaceReady={false}
      />
    )

    expect(screen.getByText("Loading workspace context")).toBeInTheDocument()
    expect(
      screen.getByText("Wait for workspace identity before sending.")
    ).toBeInTheDocument()
    expect(screen.queryByText("Ready via registry")).not.toBeInTheDocument()
  })

  it("surfaces send failures and missing model recovery in the inspector", () => {
    render(
      <InspectorRail
        scopeLabel="Default workspace"
        stagedSourceCount={0}
        stagedSources={[]}
        selectedModelLabel="No model selected"
        hasModelSelected={false}
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable
        streaming={false}
        workspaceReady
        sendError="Send failed"
      />
    )

    expect(screen.getByText("Send failed")).toBeInTheDocument()
    expect(screen.getByText("Choose a model before sending.")).toBeInTheDocument()
    expect(
      screen.getByText("Persona is optional; workspace defaults apply when available.")
    ).toBeInTheDocument()
  })
})
