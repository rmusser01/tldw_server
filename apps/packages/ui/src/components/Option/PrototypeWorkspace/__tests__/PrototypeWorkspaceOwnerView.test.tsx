import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PrototypeWorkspaceOwnerView } from "../PrototypeWorkspaceOwnerView"
import type { PrototypeWorkspaceDetail } from "@/types/prototype-workspace"

const hookState = vi.hoisted(() => ({
  useCreatePrototypeWorkspace: vi.fn(),
  useCreateOwnerBranchSession: vi.fn(),
  useReviewPrototypePromotionRequest: vi.fn(),
  useCreateToken: vi.fn()
}))

vi.mock("@/hooks/usePrototypeWorkspaces", () => ({
  useCreatePrototypeWorkspace: (...args: unknown[]) =>
    hookState.useCreatePrototypeWorkspace(...args),
  useCreateOwnerBranchSession: (...args: unknown[]) =>
    hookState.useCreateOwnerBranchSession(...args),
  useReviewPrototypePromotionRequest: (...args: unknown[]) =>
    hookState.useReviewPrototypePromotionRequest(...args)
}))

vi.mock("@/hooks/useSharing", () => ({
  useCreateToken: (...args: unknown[]) => hookState.useCreateToken(...args)
}))

const buildWorkspace = (
  overrides: Partial<PrototypeWorkspaceDetail> = {}
): PrototypeWorkspaceDetail => ({
  id: "pw_owner_review",
  owner_user_id: 1,
  title: "Owner review prototype",
  creation_source: "prompt",
  canonical_snapshot_id: "psnap_canonical",
  last_known_good_snapshot_id: "psnap_canonical",
  canonical_preview_status: "ready",
  publish_validation_status: "validated",
  preview_policy: {},
  share_policy: {},
  runtime_policy: {},
  designated_promoter_ids: [],
  created_at: "2026-05-22T00:00:00Z",
  updated_at: "2026-05-22T00:00:00Z",
  is_archived: false,
  viewer_role: "owner",
  sessions: [
    {
      id: "pss_collab",
      prototype_workspace_id: "pw_owner_review",
      base_snapshot_id: "psnap_canonical",
      actor_shared_actor_id: "psa_1",
      actor_type: "external_collaborator",
      share_link_id: 1,
      runtime_status: "running",
      preview_status: "ready",
      created_at: "2026-05-22T00:00:00Z",
      updated_at: "2026-05-22T00:00:00Z",
      is_revoked: false
    }
  ],
  snapshots: [
    {
      snapshot_id: "psnap_candidate",
      prototype_workspace_id: "pw_owner_review",
      parent_snapshot_id: "psnap_canonical",
      created_from_session_id: "pss_collab",
      author_shared_actor_id: "psa_1",
      storage_ref: "prototype://candidate",
      diff_summary: {},
      prompt_summary: "Candidate revision",
      preview_health: {},
      created_at: "2026-05-22T00:00:00Z",
      is_canonical: false,
      is_last_known_good: false
    },
    {
      snapshot_id: "psnap_canonical",
      prototype_workspace_id: "pw_owner_review",
      diff_summary: {},
      preview_health: {},
      created_at: "2026-05-21T00:00:00Z",
      is_canonical: true,
      is_last_known_good: true
    }
  ],
  promotion_requests: [
    {
      id: "ppr_pending",
      prototype_workspace_id: "pw_owner_review",
      prototype_session_id: "pss_collab",
      candidate_snapshot_id: "psnap_candidate",
      requested_by_shared_actor_id: "psa_1",
      requested_by_user_id: null,
      status: "pending",
      reviewed_by_user_id: null,
      review_notes: null,
      created_at: "2026-05-22T00:00:00Z",
      updated_at: "2026-05-22T00:00:00Z"
    }
  ],
  ...overrides
})

describe("PrototypeWorkspaceOwnerView", () => {
  const reviewMutateAsync = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    hookState.useCreatePrototypeWorkspace.mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn()
    })
    hookState.useCreateOwnerBranchSession.mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn()
    })
    hookState.useCreateToken.mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn()
    })
    reviewMutateAsync.mockResolvedValue({
      status: "promoted",
      prototype_workspace_id: "pw_owner_review",
      candidate_snapshot_id: "psnap_candidate",
      canonical_snapshot_id: "psnap_candidate",
      preview_handle: "pph_promoted",
      details: {}
    })
    hookState.useReviewPrototypePromotionRequest.mockReturnValue({
      isPending: false,
      mutateAsync: reviewMutateAsync,
      data: null,
      error: null
    })
  })

  it("renders pending promotion requests with explicit owner review actions", async () => {
    const user = userEvent.setup()
    render(
      <PrototypeWorkspaceOwnerView
        prototypeWorkspaceId="pw_owner_review"
        workspace={buildWorkspace()}
      />
    )

    const request = screen.getByTestId("prototype-promotion-request-ppr_pending")
    expect(request).toHaveTextContent("Pending owner review")
    expect(request).toHaveTextContent("Candidate psnap_candidate")
    expect(request).toHaveTextContent("Session pss_collab")
    expect(request).toHaveTextContent("Shared actor psa_1")

    const approve = within(request).getByRole("button", {
      name: "Approve promotion ppr_pending"
    })
    const reject = within(request).getByRole("button", {
      name: "Reject promotion ppr_pending"
    })
    expect(approve).toBeEnabled()
    expect(reject).toBeEnabled()

    await user.click(approve)

    expect(reviewMutateAsync).toHaveBeenCalledWith({
      promotion_request_id: "ppr_pending",
      prototype_workspace_id: "pw_owner_review",
      decision: "approve",
      review_baseline_snapshot_id: "psnap_canonical"
    })
  })

  it("renders terminal promotion states distinctly and disables review actions", () => {
    render(
      <PrototypeWorkspaceOwnerView
        prototypeWorkspaceId="pw_owner_review"
        workspace={buildWorkspace({
          promotion_requests: [
            {
              id: "ppr_stale",
              prototype_workspace_id: "pw_owner_review",
              prototype_session_id: "pss_collab",
              candidate_snapshot_id: "psnap_candidate",
              requested_by_user_id: null,
              requested_by_shared_actor_id: "psa_1",
              status: "stale",
              reviewed_by_user_id: 1,
              review_notes: "Candidate is stale",
              created_at: "2026-05-22T00:00:00Z",
              updated_at: "2026-05-22T00:00:00Z"
            },
            {
              id: "ppr_rejected",
              prototype_workspace_id: "pw_owner_review",
              prototype_session_id: "pss_collab",
              candidate_snapshot_id: "psnap_candidate",
              requested_by_user_id: null,
              requested_by_shared_actor_id: "psa_1",
              status: "rejected",
              reviewed_by_user_id: 1,
              review_notes: "Validation failed",
              created_at: "2026-05-22T00:00:00Z",
              updated_at: "2026-05-22T00:00:00Z"
            },
            {
              id: "ppr_promoted",
              prototype_workspace_id: "pw_owner_review",
              prototype_session_id: "pss_collab",
              candidate_snapshot_id: "psnap_candidate",
              requested_by_user_id: null,
              requested_by_shared_actor_id: "psa_1",
              status: "promoted",
              reviewed_by_user_id: 1,
              review_notes: null,
              created_at: "2026-05-22T00:00:00Z",
              updated_at: "2026-05-22T00:00:00Z"
            }
          ]
        })}
      />
    )

    const stale = screen.getByTestId("prototype-promotion-request-ppr_stale")
    const rejected = screen.getByTestId("prototype-promotion-request-ppr_rejected")
    const promoted = screen.getByTestId("prototype-promotion-request-ppr_promoted")

    expect(stale).toHaveTextContent("Stale candidate")
    expect(rejected).toHaveTextContent("Rejected")
    expect(rejected).toHaveTextContent("Validation failed")
    expect(promoted).toHaveTextContent("Promoted")

    for (const request of [stale, rejected, promoted]) {
      expect(
        within(request).getByRole("button", { name: /Approve promotion/ })
      ).toBeDisabled()
      expect(
        within(request).getByRole("button", { name: /Reject promotion/ })
      ).toBeDisabled()
    }
  })

  it("surfaces validation-failed review results separately from branch runtime state", () => {
    hookState.useReviewPrototypePromotionRequest.mockReturnValue({
      isPending: false,
      mutateAsync: reviewMutateAsync,
      data: {
        status: "failed",
        failure_code: "publish_validation_failed",
        prototype_workspace_id: "pw_owner_review",
        candidate_snapshot_id: "psnap_candidate",
        canonical_snapshot_id: "psnap_canonical",
        preview_handle: null,
        details: { reason: "Validator rejected the candidate" }
      },
      error: null
    })

    render(
      <PrototypeWorkspaceOwnerView
        prototypeWorkspaceId="pw_owner_review"
        workspace={buildWorkspace()}
      />
    )

    const result = screen.getByTestId("prototype-promotion-review-result")
    expect(result).toHaveTextContent("Promotion failed")
    expect(result).toHaveTextContent("publish_validation_failed")
    expect(result).toHaveTextContent("Validator rejected the candidate")
  })

  it("separates branch runtime and preview status and marks revoked sessions not actionable", () => {
    render(
      <PrototypeWorkspaceOwnerView
        prototypeWorkspaceId="pw_owner_review"
        workspace={buildWorkspace({
          sessions: [
            {
              id: "pss_revoked",
              prototype_workspace_id: "pw_owner_review",
              base_snapshot_id: "psnap_canonical",
              actor_shared_actor_id: "psa_1",
              actor_type: "external_collaborator",
              share_link_id: 1,
              runtime_status: "revoked",
              preview_status: "revoked",
              created_at: "2026-05-22T00:00:00Z",
              updated_at: "2026-05-22T00:00:00Z",
              revoked_at: "2026-05-22T01:00:00Z",
              is_revoked: true
            }
          ]
        })}
      />
    )

    const session = screen.getByTestId("prototype-branch-session-pss_revoked")
    expect(session).toHaveTextContent("Runtime revoked")
    expect(session).toHaveTextContent("Preview revoked")
    expect(session).toHaveTextContent("Not actionable")
  })

  it("disables pending promotion review when the branch session is revoked", () => {
    render(
      <PrototypeWorkspaceOwnerView
        prototypeWorkspaceId="pw_owner_review"
        workspace={buildWorkspace({
          sessions: [
            {
              id: "pss_collab",
              prototype_workspace_id: "pw_owner_review",
              base_snapshot_id: "psnap_canonical",
              actor_shared_actor_id: "psa_1",
              actor_type: "external_collaborator",
              share_link_id: 1,
              runtime_status: "revoked",
              preview_status: "revoked",
              created_at: "2026-05-22T00:00:00Z",
              updated_at: "2026-05-22T00:00:00Z",
              revoked_at: "2026-05-22T01:00:00Z",
              is_revoked: true
            }
          ]
        })}
      />
    )

    const request = screen.getByTestId("prototype-promotion-request-ppr_pending")
    expect(
      within(request).getByRole("button", {
        name: "Approve promotion ppr_pending"
      })
    ).toBeDisabled()
    expect(
      within(request).getByRole("button", {
        name: "Reject promotion ppr_pending"
      })
    ).toBeDisabled()
  })
})
