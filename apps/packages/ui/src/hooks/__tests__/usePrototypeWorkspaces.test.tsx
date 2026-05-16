import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

const fetchWithTldwAuthMock = vi.hoisted(() => vi.fn())
const getTldwServerURLMock = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/auth-fetch", () => ({
  fetchWithTldwAuth: fetchWithTldwAuthMock
}))

vi.mock("@/services/tldw-server", () => ({
  getTldwServerURL: getTldwServerURLMock
}))

import {
  prototypeWorkspaceQueryKeys,
  usePrototypeWorkspace,
  useCreateCollaboratorBranchSession,
  useCreateOwnerBranchSession,
  useCreatePromotionRequest,
  useCreatePrototypeWorkspace
} from "@/hooks/usePrototypeWorkspaces"

const buildWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

const buildWrapperWithClient = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
  return { queryClient, wrapper }
}

describe("usePrototypeWorkspaces", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getTldwServerURLMock.mockResolvedValue("http://127.0.0.1:8000")
  })

  it("exposes stable query keys for prototype workspace collections and mutations", () => {
    expect(prototypeWorkspaceQueryKeys.all()).toEqual(["prototype-workspaces"])
    expect(prototypeWorkspaceQueryKeys.workspaces()).toEqual([
      "prototype-workspaces",
      "workspaces"
    ])
    expect(prototypeWorkspaceQueryKeys.workspace("pw_1")).toEqual([
      "prototype-workspaces",
      "workspaces",
      "detail",
      "pw_1"
    ])
    expect(prototypeWorkspaceQueryKeys.sessions("pw_1")).toEqual([
      "prototype-workspaces",
      "sessions",
      "pw_1"
    ])
    expect(prototypeWorkspaceQueryKeys.promotions("pw_1")).toEqual([
      "prototype-workspaces",
      "promotions",
      "pw_1"
    ])
  })

  it("creates a prototype workspace through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(JSON.stringify({ id: "pw_1", title: "Sales dashboard" }), {
        status: 201,
        headers: { "Content-Type": "application/json" }
      })
    )

    const { result } = renderHook(() => useCreatePrototypeWorkspace(), {
      wrapper: buildWrapper()
    })

    await act(async () => {
      await result.current.mutateAsync({
        title: "Sales dashboard",
        creation_source: "prompt",
        prompt: "Build a B2B dashboard"
      })
    })

    await waitFor(() => {
      expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/prototype-workspaces",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Content-Type": "application/json"
          }),
          body: JSON.stringify({
            title: "Sales dashboard",
            creation_source: "prompt",
            prompt: "Build a B2B dashboard"
          })
        })
      )
    })
  })

  it("loads prototype workspace detail through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "pw_1",
          title: "Sales dashboard",
          owner_user_id: 1,
          creation_source: "prompt",
          canonical_snapshot_id: "psnap_seed_1",
          last_known_good_snapshot_id: "psnap_seed_1",
          canonical_preview_status: "ready",
          publish_validation_status: "passed",
          preview_policy: {},
          share_policy: {},
          runtime_policy: {},
          designated_promoter_ids: [],
          created_at: "2026-04-19T00:00:00Z",
          updated_at: "2026-04-19T00:00:00Z",
          is_archived: false,
          viewer_role: "owner",
          sessions: [],
          snapshots: []
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => usePrototypeWorkspace("pw_1"), {
      wrapper: buildWrapper()
    })

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/prototype-workspaces/pw_1"
    )
  })

  it("uses a semantic null detail key while the workspace id is unavailable", () => {
    const { queryClient, wrapper } = buildWrapperWithClient()

    renderHook(() => usePrototypeWorkspace(null), { wrapper })

    expect(fetchWithTldwAuthMock).not.toHaveBeenCalled()
    expect(queryClient.getQueryCache().getAll().map((query) => query.queryKey)).toContainEqual([
      "prototype-workspaces",
      "workspaces",
      "detail",
      null
    ])
    expect(queryClient.getQueryCache().find({
      queryKey: ["prototype-workspaces", "workspaces", "detail", "unknown"]
    })).toBeUndefined()
  })

  it("creates an owner branch session through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: "job-1",
          job_type: "branch_session_bootstrap",
          prototype_workspace_id: "pw_1",
          prototype_session_id: "pss_1",
          actor_type: "owner"
        }),
        {
          status: 202,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => useCreateOwnerBranchSession("pw_1"), {
      wrapper: buildWrapper()
    })

    await act(async () => {
      await result.current.mutateAsync({
        request_nonce: "owner-nonce-1"
      })
    })

    await waitFor(() => {
      expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/prototype-workspaces/pw_1/sessions",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Content-Type": "application/json"
          }),
          body: JSON.stringify({
            request_nonce: "owner-nonce-1"
          })
        })
      )
    })
  })

  it("creates a collaborator branch session through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          job_id: "job-2",
          job_type: "branch_session_bootstrap",
          prototype_workspace_id: "pw_1",
          prototype_session_id: "pss_2",
          actor_type: "external_collaborator",
          shared_actor_id: "psa_1"
        }),
        {
          status: 202,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => useCreateCollaboratorBranchSession(), {
      wrapper: buildWrapper()
    })

    await act(async () => {
      await result.current.mutateAsync({
        session_token: "session-token-1",
        request_nonce: "collab-nonce-1"
      })
    })

    await waitFor(() => {
      expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/prototype-sessions",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Content-Type": "application/json"
          }),
          body: JSON.stringify({
            session_token: "session-token-1",
            request_nonce: "collab-nonce-1"
          })
        })
      )
    })
  })

  it("preserves structured collaborator session errors for setup-failed UI states", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: {
            category: "bootstrap_failed",
            frontend_state: "setup_failed",
            message: "Prototype branch session could not be created",
            retryable: true
          }
        }),
        {
          status: 503,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => useCreateCollaboratorBranchSession(), {
      wrapper: buildWrapper()
    })

    await expect(
      result.current.mutateAsync({
        session_token: "session-token-1"
      })
    ).rejects.toMatchObject({
      status: 503,
      detail: {
        category: "bootstrap_failed",
        frontend_state: "setup_failed",
        retryable: true,
        message: "Prototype branch session could not be created"
      },
      message: "Prototype branch session could not be created"
    })
  })

  it("creates a prototype promotion request through the authenticated tldw fetch helper", async () => {
    fetchWithTldwAuthMock.mockResolvedValue(
      new Response(
        JSON.stringify({
          id: "pr_1",
          prototype_workspace_id: "pw_1",
          prototype_session_id: "pss_1",
          candidate_snapshot_id: "psnap_1",
          status: "pending"
        }),
        {
          status: 201,
          headers: { "Content-Type": "application/json" }
        }
      )
    )

    const { result } = renderHook(() => useCreatePromotionRequest(), {
      wrapper: buildWrapper()
    })

    await act(async () => {
      await result.current.mutateAsync({
        prototype_workspace_id: "pw_1",
        prototype_session_id: "pss_1",
        candidate_snapshot_id: "psnap_1",
        session_token: "session-token-1",
        request_reason: "Ready for review"
      })
    })

    await waitFor(() => {
      expect(fetchWithTldwAuthMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/prototype-promotions",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Content-Type": "application/json"
          }),
          body: JSON.stringify({
            prototype_workspace_id: "pw_1",
            prototype_session_id: "pss_1",
            candidate_snapshot_id: "psnap_1",
            session_token: "session-token-1",
            request_reason: "Ready for review"
          })
        })
      )
    })
  })
})
