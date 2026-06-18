import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"

import { useACPSessionsStore } from "@/store/acp-sessions"
import { ACPWorkspacePanel } from "../ACPWorkspacePanel"

const workspaceStoreMock = vi.hoisted(() => ({
  workspaceId: "workspace-alpha"
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: null
  })
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: { workspaceId: string | null }) => unknown
  ) => selector({ workspaceId: workspaceStoreMock.workspaceId })
}))

describe("ACPWorkspacePanel canonical Workspace handoff", () => {
  beforeEach(() => {
    useACPSessionsStore.getState().reset()
    workspaceStoreMock.workspaceId = "workspace-alpha"
  })

  it("links no-session users to the canonical Workspaces manager", () => {
    render(<ACPWorkspacePanel />)

    expect(
      screen.getByText(/Select a session to open the workspace terminal/i)
    ).toBeTruthy()
    expect(
      screen.getByText(/Use Workspaces to create or attach the project root/i)
    ).toBeTruthy()
    expect(
      screen.getByRole("link", { name: /manage canonical Workspaces/i })
    ).toHaveAttribute("href", "#/workspaces")
  })

  it("shows when the ACP session workspace matches the active server workspace", () => {
    const store = useACPSessionsStore.getState()
    const sessionId = store.createSession({
      cwd: "/workspace/alpha",
      name: "Alpha Session",
      workspaceId: "workspace-alpha"
    })
    store.updateSessionMetadata(sessionId, {
      sshWsUrl: "/api/v1/acp/sessions/alpha/ssh"
    })

    render(<ACPWorkspacePanel />)

    expect(screen.getByText("Session Workspace")).toBeInTheDocument()
    expect(
      screen.getByText(/Aligned with active Workspace/i)
    ).toBeInTheDocument()
    expect(screen.getByText("workspace-alpha")).toBeInTheDocument()
  })

  it("shows mismatch recovery when the ACP session workspace differs from the active workspace", () => {
    const store = useACPSessionsStore.getState()
    const sessionId = store.createSession({
      cwd: "/workspace/beta",
      name: "Beta Session",
      workspaceId: "workspace-beta"
    })
    store.updateSessionMetadata(sessionId, {
      sshWsUrl: "/api/v1/acp/sessions/beta/ssh"
    })

    render(<ACPWorkspacePanel />)

    expect(screen.getByText("Workspace mismatch")).toBeInTheDocument()
    expect(screen.getByText(/workspace-beta/)).toBeInTheDocument()
    expect(screen.getByText(/workspace-alpha/)).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Open Workspaces" })
    ).toHaveAttribute("href", "#/workspaces")
  })

  it("does not duplicate the active workspace id when the ACP session has no workspace id", () => {
    const store = useACPSessionsStore.getState()
    const sessionId = store.createSession({
      cwd: "/workspace/alpha",
      name: "Unattached Session"
    })
    store.updateSessionMetadata(sessionId, {
      sshWsUrl: "/api/v1/acp/sessions/unattached/ssh"
    })

    render(<ACPWorkspacePanel />)

    expect(screen.getByText("Active Workspace only")).toBeInTheDocument()
    expect(screen.getAllByText("workspace-alpha")).toHaveLength(1)
  })
})
