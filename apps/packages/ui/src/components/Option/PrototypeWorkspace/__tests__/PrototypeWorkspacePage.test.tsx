import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import { PrototypeWorkspacePage } from "../PrototypeWorkspacePage"

const searchParamsState = vi.hoisted(() => ({
  value: new URLSearchParams()
}))

const locationState = vi.hoisted(() => ({
  value: null as unknown
}))

const hookState = vi.hoisted(() => ({
  usePrototypeWorkspace: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useLocation: () => ({ state: locationState.value }),
    useSearchParams: () => [searchParamsState.value, vi.fn()]
  }
})

vi.mock("@/hooks/usePrototypeWorkspaces", () => ({
  usePrototypeWorkspace: (...args: unknown[]) =>
    hookState.usePrototypeWorkspace(...args)
}))

vi.mock("../PrototypeWorkspaceOwnerView", () => ({
  PrototypeWorkspaceOwnerView: ({
    prototypeWorkspaceId
  }: {
    prototypeWorkspaceId?: string | null
  }) => (
    <div data-testid="prototype-workspace-owner-view">
      Owner:{prototypeWorkspaceId ?? "none"}
    </div>
  )
}))

vi.mock("../PrototypeWorkspaceSessionView", () => ({
  PrototypeWorkspaceSessionView: ({
    prototypeWorkspaceId,
    sessionToken,
    shareToken,
    initialPassword
  }: {
    prototypeWorkspaceId?: string | null
    sessionToken?: string | null
    shareToken?: string | null
    initialPassword?: string | null
  }) => (
    <div data-testid="prototype-workspace-session-view">
      Workspace:{prototypeWorkspaceId ?? "none"} Session:{sessionToken ?? "none"}{" "}
      Share:{shareToken ?? "none"} Password:{initialPassword ?? "none"}
    </div>
  )
}))

describe("PrototypeWorkspacePage", () => {
  beforeEach(() => {
    searchParamsState.value = new URLSearchParams()
    locationState.value = null
    usePrototypeWorkspaceStore.getState().reset()
    hookState.usePrototypeWorkspace.mockReturnValue({
      data: null
    })
  })

  it("renders the owner workspace view when the workspace detail resolves the viewer as owner", () => {
    searchParamsState.value = new URLSearchParams({
      workspace: "pw_1"
    })
    hookState.usePrototypeWorkspace.mockReturnValue({
      data: {
        id: "pw_1",
        viewer_role: "owner",
        sessions: [],
        snapshots: []
      }
    })

    render(<PrototypeWorkspacePage />)

    expect(
      screen.getByTestId("prototype-workspace-owner-view")
    ).toHaveTextContent("Owner:pw_1")
    expect(
      screen.queryByTestId("prototype-workspace-session-view")
    ).not.toBeInTheDocument()
  })

  it("renders the collaborator session view when workspace detail resolves a non-owner viewer", () => {
    searchParamsState.value = new URLSearchParams({
      workspace: "pw_1"
    })
    hookState.usePrototypeWorkspace.mockReturnValue({
      data: {
        id: "pw_1",
        viewer_role: "internal_collaborator",
        sessions: [],
        snapshots: []
      }
    })

    render(<PrototypeWorkspacePage />)

    expect(
      screen.getByTestId("prototype-workspace-session-view")
    ).toHaveTextContent("Session:none Share:none")
    expect(
      screen.queryByTestId("prototype-workspace-owner-view")
    ).not.toBeInTheDocument()
  })

  it("renders the collaborator session view when the page is entered with a prototype session token", () => {
    searchParamsState.value = new URLSearchParams({
      session_token: "session-token-1",
      share_token: "share-token-1"
    })

    render(<PrototypeWorkspacePage />)

    expect(
      screen.getByTestId("prototype-workspace-session-view")
    ).toHaveTextContent("Session:session-token-1 Share:share-token-1")
    expect(
      screen.queryByTestId("prototype-workspace-owner-view")
    ).not.toBeInTheDocument()
  })

  it("does not pass a stale active workspace id to token-only collaborator entries", () => {
    usePrototypeWorkspaceStore.getState().setActiveWorkspaceId("pw_stale_owner")
    searchParamsState.value = new URLSearchParams({
      share_token: "share-token-1"
    })

    render(<PrototypeWorkspacePage />)

    expect(hookState.usePrototypeWorkspace).toHaveBeenCalledWith(null)
    expect(
      screen.getByTestId("prototype-workspace-session-view")
    ).toHaveTextContent("Workspace:none Session:none Share:share-token-1")
  })

  it("passes prototype share passwords from navigation state into collaborator entry", () => {
    searchParamsState.value = new URLSearchParams({
      share_token: "share-token-1"
    })
    locationState.value = {
      prototypeSharePassword: "demo-pass"
    }

    render(<PrototypeWorkspacePage />)

    expect(
      screen.getByTestId("prototype-workspace-session-view")
    ).toHaveTextContent("Password:demo-pass")
  })
})
