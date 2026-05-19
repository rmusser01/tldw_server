import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import {
  getPrototypeContractErrorDetail,
  getPrototypeContractState,
  type PrototypeContractErrorDetail
} from "@/test-utils/prototype-contract-fixtures"
import { PrototypeWorkspaceSessionView } from "../PrototypeWorkspaceSessionView"

const hookState = vi.hoisted(() => ({
  usePrototypePrivateLinkExchange: vi.fn(),
  usePrototypeWorkspace: vi.fn(),
  useCreateCollaboratorBranchSession: vi.fn(),
  useCreatePromotionRequest: vi.fn()
}))

const routerState = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => routerState.navigate
  }
})

vi.mock("@/hooks/useSharing", () => ({
  usePrototypePrivateLinkExchange: (...args: unknown[]) =>
    hookState.usePrototypePrivateLinkExchange(...args)
}))

vi.mock("@/hooks/usePrototypeWorkspaces", () => ({
  usePrototypeWorkspace: (...args: unknown[]) =>
    hookState.usePrototypeWorkspace(...args),
  useCreateCollaboratorBranchSession: (...args: unknown[]) =>
    hookState.useCreateCollaboratorBranchSession(...args),
  useCreatePromotionRequest: (...args: unknown[]) =>
    hookState.useCreatePromotionRequest(...args)
}))

const prototypeError = (
  detail: PrototypeContractErrorDetail
) => {
  const error = new Error(detail.message ?? detail.category)
  return Object.assign(error, {
    status: 403,
    detail
  })
}

describe("PrototypeWorkspaceSessionView", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    usePrototypeWorkspaceStore.getState().reset()
    hookState.usePrototypeWorkspace.mockReturnValue({ data: null })
    hookState.usePrototypePrivateLinkExchange.mockReturnValue({
      isPending: false,
      error: null,
      mutateAsync: vi.fn()
    })
    hookState.useCreateCollaboratorBranchSession.mockReturnValue({
      data: null,
      isPending: false,
      error: null,
      variables: undefined,
      mutateAsync: vi.fn()
    })
    hookState.useCreatePromotionRequest.mockReturnValue({
      isPending: false,
      error: null,
      mutateAsync: vi.fn()
    })
  })

  it("does not load stale owner workspace or stale collaborator session for a new share-token entry", () => {
    const store = usePrototypeWorkspaceStore.getState()
    store.setActiveWorkspaceId("pw_stale_owner")
    store.setCollaboratorEntry({
      collaboratorSessionId: "pss_old",
      collaboratorSessionToken: "old-session-token",
      collaboratorShareToken: "old-share-token",
      sharedActorId: "psa_old"
    })

    render(<PrototypeWorkspaceSessionView shareToken="new-share-token" />)

    expect(hookState.usePrototypeWorkspace).toHaveBeenCalledWith(null)
    expect(screen.queryByText("Workspace: pw_stale_owner")).not.toBeInTheDocument()
    expect(screen.queryByText("Session: pss_old")).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Start collaborator session" })
    ).toBeDisabled()
  })

  it("removes token-bearing route state after a collaborator branch session starts", async () => {
    const createSession = vi.fn().mockResolvedValue({
      job_id: "job-1",
      job_type: "branch_session_bootstrap",
      status: "queued",
      message: "queued",
      prototype_workspace_id: "pw_collab",
      prototype_session_id: "pss_collab",
      actor_type: "external_collaborator",
      shared_actor_id: "psa_collab"
    })
    hookState.useCreateCollaboratorBranchSession.mockReturnValue({
      data: null,
      isPending: false,
      error: null,
      variables: undefined,
      mutateAsync: createSession
    })

    render(
      <PrototypeWorkspaceSessionView
        shareToken="share-token-1"
        sessionToken="session-token-1"
      />
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Start collaborator session" })
    )

    await waitFor(() => {
      expect(createSession).toHaveBeenCalledWith({
        session_token: "session-token-1"
      })
    })
    expect(routerState.navigate).toHaveBeenCalledWith(
      "/prototype-workspaces?workspace=pw_collab",
      { replace: true }
    )
  })

  it("does not reuse mutation data from a previous route token", () => {
    const store = usePrototypeWorkspaceStore.getState()
    store.setCollaboratorEntry({
      collaboratorSessionId: "pss_stale_store",
      collaboratorSessionToken: "old-session-token",
      collaboratorShareToken: "same-share-token",
      sharedActorId: "psa_stale"
    })
    hookState.useCreateCollaboratorBranchSession.mockReturnValue({
      data: {
        job_id: "job-old",
        job_type: "branch_session_bootstrap",
        status: "queued",
        message: "queued",
        prototype_workspace_id: "pw_previous",
        prototype_session_id: "pss_previous",
        actor_type: "external_collaborator",
        shared_actor_id: "psa_previous"
      },
      isPending: false,
      error: null,
      variables: { session_token: "old-session-token" },
      mutateAsync: vi.fn()
    })

    render(
      <PrototypeWorkspaceSessionView
        shareToken="same-share-token"
        sessionToken="new-session-token"
      />
    )

    expect(hookState.usePrototypeWorkspace).toHaveBeenCalledWith(null)
    expect(screen.queryByText("Workspace: pw_previous")).not.toBeInTheDocument()
    expect(screen.queryByText("Session: pss_previous")).not.toBeInTheDocument()
    expect(
      screen.queryByText("Session: pss_stale_store")
    ).not.toBeInTheDocument()
  })

  it("does not show stale mutation errors from a previous route token", () => {
    hookState.useCreateCollaboratorBranchSession.mockReturnValue({
      data: null,
      isPending: false,
      error: prototypeError(getPrototypeContractErrorDetail("bootstrap_failed")),
      variables: { session_token: "old-session-token" },
      mutateAsync: vi.fn()
    })

    render(<PrototypeWorkspaceSessionView sessionToken="new-session-token" />)

    expect(
      screen.queryByTestId("prototype-entry-error-state")
    ).not.toBeInTheDocument()
  })

  it("maps frozen link exchange errors into collaborator entry route states", () => {
    const invalidLink = getPrototypeContractState("invalid_link")
    const invalidLinkDetail = invalidLink.mockResponse.detail
    hookState.usePrototypePrivateLinkExchange.mockReturnValue({
      isPending: false,
      error: prototypeError(invalidLinkDetail),
      variables: { token: "share-token-1" },
      mutateAsync: vi.fn()
    })

    render(<PrototypeWorkspaceSessionView shareToken="share-token-1" />)

    expect(screen.getByTestId("prototype-entry-error-state")).toHaveTextContent(
      invalidLink.frontendStateBucket
    )
    expect(screen.getByTestId("prototype-entry-error-state")).toHaveTextContent(
      invalidLinkDetail.message
    )
    expect(screen.queryByText("Retry is available")).not.toBeInTheDocument()
  })

  it("uses retryability from structured session errors for setup failures", () => {
    const bootstrapFailedDetail =
      getPrototypeContractErrorDetail("bootstrap_failed")
    hookState.useCreateCollaboratorBranchSession.mockReturnValue({
      data: null,
      isPending: false,
      error: prototypeError(bootstrapFailedDetail),
      variables: { session_token: "session-token-1" },
      mutateAsync: vi.fn()
    })

    render(<PrototypeWorkspaceSessionView sessionToken="session-token-1" />)

    expect(screen.getByTestId("prototype-entry-error-state")).toHaveTextContent(
      "Setup failed"
    )
    expect(screen.getByTestId("prototype-entry-error-state")).toHaveTextContent(
      "Retry is available"
    )
  })
})
