import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { message } from "antd"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"

import { FamilyGuardrailsWizard } from "../FamilyGuardrailsWizard"

const {
  useCanonicalConnectionConfigMock,
  createHouseholdDraftMock,
  updateHouseholdDraftMock,
  addHouseholdMemberDraftMock,
  saveRelationshipDraftMock,
  saveGuardrailPlanDraftMock,
  listHouseholdDraftsMock,
  getHouseholdDraftSnapshotMock,
  getHouseholdInviteTrackerMock,
  provisionHouseholdMemberInviteMock,
  resendHouseholdMemberInviteMock,
  reissueHouseholdMemberInviteMock,
  clipboardWriteTextMock,
  resendPendingInvitesMock
} = vi.hoisted(() => ({
  useCanonicalConnectionConfigMock: vi.fn(),
  createHouseholdDraftMock: vi.fn(),
  updateHouseholdDraftMock: vi.fn(),
  addHouseholdMemberDraftMock: vi.fn(),
  saveRelationshipDraftMock: vi.fn(),
  saveGuardrailPlanDraftMock: vi.fn(),
  listHouseholdDraftsMock: vi.fn(),
  getHouseholdDraftSnapshotMock: vi.fn(),
  getHouseholdInviteTrackerMock: vi.fn(),
  provisionHouseholdMemberInviteMock: vi.fn(),
  resendHouseholdMemberInviteMock: vi.fn(),
  reissueHouseholdMemberInviteMock: vi.fn(),
  clipboardWriteTextMock: vi.fn(),
  resendPendingInvitesMock: vi.fn()
}))

vi.mock("@/services/family-wizard", () => ({
  createHouseholdDraft: createHouseholdDraftMock,
  updateHouseholdDraft: updateHouseholdDraftMock,
  addHouseholdMemberDraft: addHouseholdMemberDraftMock,
  saveRelationshipDraft: saveRelationshipDraftMock,
  saveGuardrailPlanDraft: saveGuardrailPlanDraftMock,
  listHouseholdDrafts: listHouseholdDraftsMock,
  getHouseholdDraftSnapshot: getHouseholdDraftSnapshotMock,
  getHouseholdInviteTracker: getHouseholdInviteTrackerMock,
  provisionHouseholdMemberInvite: provisionHouseholdMemberInviteMock,
  resendHouseholdMemberInvite: resendHouseholdMemberInviteMock,
  reissueHouseholdMemberInvite: reissueHouseholdMemberInviteMock,
  resendPendingInvites: resendPendingInvitesMock
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: (...args: unknown[]) =>
    useCanonicalConnectionConfigMock(...args)
}))

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const startNewHousehold = async () => {
  await waitFor(() => {
    expect(screen.getByRole("heading", { name: "Household Basics" })).toBeInTheDocument()
  })
}

const expectDesignSystemAlert = (text: string | RegExp) => {
  const node =
    typeof text === "string"
      ? screen.getByText(text, { exact: false })
      : screen.getByText(text)

  expect(node.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
}

const trackerItem = (overrides: Record<string, unknown> = {}) => ({
  member_draft_id: "member-dependent-1",
  display_name: "Alex",
  account_mode: "existing_account",
  dependent_user_id: "alex-kid",
  relationship_draft_id: "relationship-draft-1",
  relationship_status: "pending",
  plan_draft_id: "plan-draft-1",
  plan_status: "queued",
  invite_id: "invite-1",
  invite_status: "sent",
  invite_delivery_channel: "email",
  invite_delivery_target: "alex@example.com",
  invite_last_sent_at: "2026-01-01T00:00:00Z",
  invite_accepted_at: null,
  invite_expires_at: "2026-01-08T00:00:00Z",
  blocker_codes: ["invite_pending_acceptance", "plan_waiting_for_acceptance"],
  available_actions: ["resend_invite"],
  ...overrides
})

describe("FamilyGuardrailsWizard", () => {
  const originalMatchMedia = window.matchMedia
  const originalClipboard = Object.getOwnPropertyDescriptor(window.navigator, "clipboard")

  beforeAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn()
      }))
    })
    Object.defineProperty(window.navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: clipboardWriteTextMock
      }
    })
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
    if (originalClipboard) {
      Object.defineProperty(window.navigator, "clipboard", originalClipboard)
      return
    }
    delete (window.navigator as Navigator & { clipboard?: Clipboard }).clipboard
  })

  beforeEach(() => {
    fetchMock.mockReset()
    useCanonicalConnectionConfigMock.mockReset()
    createHouseholdDraftMock.mockReset()
    updateHouseholdDraftMock.mockReset()
    addHouseholdMemberDraftMock.mockReset()
    saveRelationshipDraftMock.mockReset()
    saveGuardrailPlanDraftMock.mockReset()
    listHouseholdDraftsMock.mockReset()
    getHouseholdDraftSnapshotMock.mockReset()
    getHouseholdInviteTrackerMock.mockReset()
    provisionHouseholdMemberInviteMock.mockReset()
    resendHouseholdMemberInviteMock.mockReset()
    reissueHouseholdMemberInviteMock.mockReset()
    resendPendingInvitesMock.mockReset()
    clipboardWriteTextMock.mockReset()
    useCanonicalConnectionConfigMock.mockReturnValue({
      config: {
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "test-key"
      },
      loading: false
    })
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        paths: {
          "/api/v1/guardian/wizard/drafts": {}
        }
      })
    })
    listHouseholdDraftsMock.mockResolvedValue([])
    clipboardWriteTextMock.mockResolvedValue(undefined)
    resendPendingInvitesMock.mockResolvedValue({
      household_draft_id: "draft-1",
      resent_count: 1,
      skipped_count: 0,
      resent_user_ids: ["child-1"],
      skipped_user_ids: [],
      resent_member_draft_ids: ["member-dependent-1"],
      skipped_member_draft_ids: []
    })
    createHouseholdDraftMock.mockResolvedValue({
      id: "draft-1",
      owner_user_id: "guardian-1",
      name: "My Household",
      mode: "family",
      status: "draft",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z"
    })
    updateHouseholdDraftMock.mockResolvedValue({
      id: "draft-1",
      owner_user_id: "guardian-1",
      name: "My Household",
      mode: "family",
      status: "draft",
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z"
    })
    addHouseholdMemberDraftMock.mockImplementation(
      async (_draftId: string, body: { role: string; display_name: string; user_id?: string; email?: string }) => ({
        id: `${body.role}-${body.user_id ?? body.display_name}`,
        household_draft_id: "draft-1",
        role: body.role,
        display_name: body.display_name,
        user_id: body.user_id ?? body.display_name.toLowerCase().replace(/\s+/g, "-"),
        email: body.email ?? null,
        invite_required: body.role === "dependent",
        metadata: {},
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      })
    )
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      status: "invites_pending",
      active_count: 1,
      pending_count: 1,
      failed_count: 0,
      items: []
    })
  })

  it("shows resume and edit actions for saved household drafts", async () => {
    listHouseholdDraftsMock.mockResolvedValue([
      {
        id: "draft-saved-1",
        owner_user_id: "guardian-1",
        name: "Saved Home",
        mode: "family",
        status: "draft",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-02T00:00:00Z"
      }
    ])
    render(<FamilyGuardrailsWizard />)

    await waitFor(() => {
      expect(screen.getByText("Resume saved household")).toBeInTheDocument()
    })
    expect(screen.getByRole("button", { name: "Resume latest draft" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Edit household" })).toBeInTheDocument()
    expect(screen.getByText("Saved Home")).toBeInTheDocument()
  })

  it("skips saved draft loading when the server does not expose wizard draft endpoints", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        paths: {}
      })
    })

    render(<FamilyGuardrailsWizard />)

    expect(
      await screen.findByText("Saved household drafts unavailable")
    ).toBeInTheDocument()
    expectDesignSystemAlert("Saved household drafts unavailable")
    expectDesignSystemAlert(
      "This server does not expose family wizard draft endpoints yet. Start a new household to continue."
    )
    expect(listHouseholdDraftsMock).not.toHaveBeenCalled()
  })

  it("renders compact step labels with current and next step context", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()

    expect(screen.getByText("Basics")).toBeInTheDocument()
    expect(screen.getByText("Tracker")).toBeInTheDocument()
    expect(screen.getByText("Review")).toBeInTheDocument()
    expect(screen.getByText("Step 1 of 8")).toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Household Basics" })).toBeInTheDocument()
    expect(screen.getByText("Next: Guardians")).toBeInTheDocument()
  })

  it("updates step context cues as the wizard advances", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByText("Step 2 of 8")).toBeInTheDocument()
      expect(screen.getByRole("heading", { name: "Add Guardians" })).toBeInTheDocument()
      expect(screen.getByText("Next: Dependents")).toBeInTheDocument()
    })
  })

  it("uses caregiver step labels in institutional mode", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    fireEvent.click(screen.getByRole("radio", { name: "Caregiver/Institutional" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByText("Step 2 of 8")).toBeInTheDocument()
      expect(screen.getByRole("heading", { name: "Add Caregivers" })).toBeInTheDocument()
      expect(screen.getByText("Caregivers")).toBeInTheDocument()
      expect(screen.getByText("Next: Dependents")).toBeInTheDocument()
    })
  })

  it("uses preset-only household setup without a separate mode selector", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    expect(screen.getByText("Household Preset")).toBeInTheDocument()
    expect(screen.queryByText("Household Mode")).not.toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Caregiver/Institutional" })).toBeInTheDocument()
  })

  it("uses caregiver terminology in intro copy after caregiver preset selection", async () => {
    render(<FamilyGuardrailsWizard />)

    expect(
      screen.getByText(
        "Template-first setup for guardians, dependents, moderation templates, and acceptance tracking."
      )
    ).toBeInTheDocument()

    await startNewHousehold()
    fireEvent.click(screen.getByRole("radio", { name: "Caregiver/Institutional" }))

    await waitFor(() => {
      expect(
        screen.getByText(
          "Template-first setup for caregivers, dependents, moderation templates, and acceptance tracking."
        )
      ).toBeInTheDocument()
    })
  })

  it("renders a sticky action footer with mobile-friendly wrapping controls", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    const footer = screen.getByTestId("wizard-action-footer")
    const controls = screen.getByTestId("wizard-action-controls")
    const continueButton = screen.getByRole("button", { name: /Save & Continue/i })

    expect(footer).toHaveStyle({ position: "sticky", bottom: "0px" })
    expect(controls).toHaveStyle({ display: "flex", flexWrap: "wrap" })
    expect(continueButton).toHaveStyle({ flex: "1 1 220px" })
  })

  it("anchors action footer to the bottom of the wizard shell", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    const shell = screen.getByTestId("wizard-shell")
    const footer = screen.getByTestId("wizard-action-footer")

    expect(shell).toHaveStyle({ minHeight: "100%" })
    expect(footer).toHaveStyle({ marginTop: "auto" })
  })

  it("shows explicit accessible labels for card-entry guardian and dependent fields", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByLabelText("Guardian 1 display name")).toBeInTheDocument()
      expect(screen.getByLabelText("Guardian 1 user ID")).toBeInTheDocument()
      expect(screen.getByLabelText("Guardian 1 email")).toBeInTheDocument()
      expectDesignSystemAlert("Add every guardian who can manage alerts and safety settings.")
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByLabelText("Dependent 1 display name")).toBeInTheDocument()
      expect(screen.getByLabelText("Dependent 1 user ID")).toBeInTheDocument()
      expect(screen.getByLabelText("Dependent 1 email")).toBeInTheDocument()
      expectDesignSystemAlert(
        "Choose whether each dependent already has an account or needs a new invite."
      )
    })
  })

  it("shows account user ID guidance in guardian and dependent setup steps", async () => {
    render(<FamilyGuardrailsWizard />)

    await startNewHousehold()
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText(
          "Use each guardian's existing account user ID (the one used to sign in)."
        )
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText(
          "Existing accounts need a sign-in user ID. Invite-new dependents can be provisioned by email or via a guardian copy link in the tracker."
        )
      ).toBeInTheDocument()
    })
  })

  it("resumes latest draft snapshot for returning guardians", async () => {
    listHouseholdDraftsMock.mockResolvedValue([
      {
        id: "draft-resume-1",
        owner_user_id: "guardian-1",
        name: "Resume Home",
        mode: "family",
        status: "draft",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      }
    ])
    getHouseholdDraftSnapshotMock.mockResolvedValue({
      id: "draft-resume-1",
      household: {
        id: "draft-resume-1",
        owner_user_id: "guardian-1",
        name: "Resume Home",
        mode: "family",
        status: "draft",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      },
      members: [
        {
          id: "member-guardian-1",
          household_draft_id: "draft-resume-1",
          role: "guardian",
          display_name: "Primary Guardian",
          user_id: "guardian-primary",
          email: null,
          invite_required: false,
          account_mode: "existing_account",
          provisioning_status: "not_started",
          metadata: {},
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        },
        {
          id: "member-dependent-1",
          household_draft_id: "draft-resume-1",
          role: "dependent",
          display_name: "Alex",
          user_id: "alex-kid",
          email: null,
          invite_required: true,
          account_mode: "existing_account",
          provisioning_status: "not_started",
          metadata: {},
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }
      ],
      relationships: [
        {
          id: "relationship-draft-1",
          household_draft_id: "draft-resume-1",
          guardian_member_draft_id: "member-guardian-1",
          dependent_member_draft_id: "member-dependent-1",
          relationship_type: "parent",
          dependent_visible: true,
          status: "pending",
          relationship_id: "relationship-1",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }
      ],
      plans: []
    })

    render(<FamilyGuardrailsWizard />)

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Resume latest draft" })).toBeEnabled()
    })
    fireEvent.click(screen.getByRole("button", { name: "Resume latest draft" }))

    await waitFor(() => {
      expect(getHouseholdDraftSnapshotMock).toHaveBeenCalledWith("draft-resume-1")
      expect(
        screen.getByText("Apply a template first, then customize if needed.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /^Back$/i }))

    await waitFor(() => {
      expect(
        screen.getByText(
          "Choose whether each dependent already has an account or needs a new invite."
        )
      ).toBeInTheDocument()
      expect(screen.getByDisplayValue("alex-kid")).toBeInTheDocument()
    })
  })

  it("shows mixed activation statuses for pending and active dependents", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 1,
      pending_count: 1,
      failed_count: 0,
      items: [
        trackerItem(),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Sam",
          dependent_user_id: "sam-kid",
          relationship_status: "active",
          plan_status: "active",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "invites_pending",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(screen.getAllByText("Queued until acceptance").length).toBeGreaterThan(0)
      expect(screen.getAllByText("Active").length).toBeGreaterThan(0)
      expect(screen.getByText("1 dependent is waiting on invite acceptance.")).toBeInTheDocument()
      expectDesignSystemAlert("1 dependent is waiting on invite acceptance.")
      expect(
        screen.getByText("Guardrails for pending dependents stay queued until acceptance.")
      ).toBeInTheDocument()
      expectDesignSystemAlert("Guardrails for pending dependents stay queued until acceptance.")
      expect(screen.getByText("Waiting on acceptance")).toBeInTheDocument()
    })
  })

  it("shows all-active tracker guidance when no dependents are pending", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 2,
      pending_count: 0,
      failed_count: 0,
      items: [
        trackerItem({
          member_draft_id: "member-dependent-1",
          display_name: "Child One",
          dependent_user_id: "child-1",
          relationship_status: "active",
          plan_status: "active",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        }),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Child Two",
          dependent_user_id: "child-2",
          relationship_status: "active",
          plan_status: "active",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "active",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(screen.getByText("All dependent guardrails are active.")).toBeInTheDocument()
      expect(
        screen.getByText("No pending invite acceptances remain for this household.")
      ).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "Resend Pending Invites" })
      ).toBeDisabled()
      expect(
        screen.queryByRole("button", { name: "Copy pending invite reminder" })
      ).not.toBeInTheDocument()
    })
  })

  it("creates a missing invite from the tracker step", async () => {
    provisionHouseholdMemberInviteMock.mockResolvedValue({
      id: "invite-1",
      household_draft_id: "draft-1",
      member_draft_id: "member-dependent-1",
      status: "ready",
      delivery_channel: "email",
      delivery_target: "alex@example.com",
      invite_token: "token-1",
      resend_count: 0,
      last_sent_at: null,
      accepted_at: null,
      expires_at: "2026-01-08T00:00:00Z",
      revoked_at: null,
      failure_reason: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z"
    })
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 0,
      pending_count: 1,
      failed_count: 0,
      items: [
        trackerItem({
          account_mode: "invite_new",
          dependent_user_id: null,
          invite_id: null,
          invite_status: "not_started",
          invite_delivery_channel: "guardian_copy",
          invite_delivery_target: null,
          invite_last_sent_at: null,
          blocker_codes: ["invite_not_provisioned", "account_not_accepted", "plan_waiting_for_acceptance"],
          available_actions: ["provision_invite"]
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "invites_pending",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    const createInviteButton = await screen.findByRole("button", {
      name: "Create Invite for Alex"
    })
    fireEvent.click(createInviteButton)

    await waitFor(() => {
      expect(provisionHouseholdMemberInviteMock).toHaveBeenCalledWith(
        "draft-1",
        "member-dependent-1"
      )
    })
  })

  it("resends pending invites from tracker step", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 1,
      pending_count: 1,
      failed_count: 0,
      items: [
        trackerItem(),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Sam",
          dependent_user_id: "sam-kid",
          relationship_status: "active",
          plan_status: "active",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "invites_pending",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Resend Pending Invites" })
      ).toBeEnabled()
    })

    fireEvent.click(screen.getByRole("button", { name: "Resend Pending Invites" }))

    await waitFor(() => {
      expect(resendPendingInvitesMock).toHaveBeenCalledTimes(1)
      expect(resendPendingInvitesMock).toHaveBeenCalledWith("draft-1", {
        dependent_user_ids: [],
        member_draft_ids: ["member-dependent-1"]
      })
    })
  })

  it("offers per-dependent resend action for pending tracker rows", async () => {
    resendHouseholdMemberInviteMock.mockResolvedValue({
      id: "invite-1",
      household_draft_id: "draft-1",
      member_draft_id: "member-dependent-1",
      status: "sent",
      delivery_channel: "email",
      delivery_target: "alex@example.com",
      invite_token: "token-1",
      resend_count: 1,
      last_sent_at: "2026-01-02T00:00:00Z",
      accepted_at: null,
      expires_at: "2026-01-08T00:00:00Z",
      revoked_at: null,
      failure_reason: null,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-02T00:00:00Z"
    })
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 1,
      pending_count: 1,
      failed_count: 0,
      items: [
        trackerItem(),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Sam",
          dependent_user_id: "sam-kid",
          relationship_status: "active",
          plan_status: "active",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "invites_pending",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    const resendRowButton = await screen.findByRole("button", {
      name: "Resend Invite for Alex"
    })
    fireEvent.click(resendRowButton)

    await waitFor(() => {
      expect(resendHouseholdMemberInviteMock).toHaveBeenCalledWith("draft-1", "invite-1")
    })
  })

  it("offers per-dependent template review action for failed tracker rows", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 1,
      pending_count: 0,
      failed_count: 1,
      items: [
        trackerItem({
          relationship_status: "active",
          plan_status: "failed",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "needs_attention",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    const reviewTemplateButton = await screen.findByRole("button", {
      name: "Review Template for Alex"
    })
    fireEvent.click(reviewTemplateButton)

    await waitFor(() => {
      expect(
        screen.getByText("Apply a template first, then customize if needed.")
      ).toBeInTheDocument()
      expect(
        screen.getByText("Reviewing template for Alex.")
      ).toBeInTheDocument()
      expectDesignSystemAlert("Reviewing template for Alex.")
      expectDesignSystemAlert(
        "Adjust template or advanced overrides, then continue to re-check activation status."
      )
    })
  })

  it("shows blocker fallback and fix-mapping action for declined tracker rows", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 0,
      pending_count: 0,
      failed_count: 1,
      items: [
        trackerItem({
          relationship_status: "declined",
          plan_status: "queued",
          blocker_codes: ["plan_waiting_for_acceptance"],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "needs_attention",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(screen.getByText("Plan queued for acceptance")).toBeInTheDocument()
    })
    const fixMappingButton = await screen.findByRole("button", {
      name: "Fix Mapping for Alex"
    })
    fireEvent.click(fixMappingButton)
    await waitFor(() => {
      expect(
        screen.getByText(
          "Choose whether each dependent already has an account or needs a new invite."
        )
      ).toBeInTheDocument()
      expect(screen.getByText("Fixing mapping for Alex.")).toBeInTheDocument()
    })
  })

  it("shows blocker tags for queued invite rows", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 0,
      pending_count: 1,
      failed_count: 1,
      items: [
        trackerItem(),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Child Two",
          dependent_user_id: "child-2",
          relationship_status: "active",
          plan_status: "failed",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={6}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "needs_attention",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(screen.getAllByText("Queued until acceptance").length).toBeGreaterThan(0)
      expect(screen.getByText("Waiting on acceptance")).toBeInTheDocument()
      expect(screen.getByText("Plan queued for acceptance")).toBeInTheDocument()
    })
  })

  it("shows pending acceptance warning on review step", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 1,
      pending_count: 1,
      failed_count: 0,
      items: [
        trackerItem(),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Child Two",
          dependent_user_id: "child-2",
          relationship_status: "active",
          plan_status: "active",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={7}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "invites_pending",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(
        screen.getByText("Setup is saved, and 1 dependent is still waiting on acceptance.")
      ).toBeInTheDocument()
      expectDesignSystemAlert("Setup is saved, and 1 dependent is still waiting on acceptance.")
      expect(
        screen.getByText("Pending dependents activate guardrails automatically after invite acceptance.")
      ).toBeInTheDocument()
      expectDesignSystemAlert(
        "Pending dependents activate guardrails automatically after invite acceptance."
      )
      expect(
        screen.getByRole("button", { name: "Finish Setup (Invites Pending)" })
      ).toBeInTheDocument()
    })
  })

  it("keeps default finish label on review step when all dependents are active", async () => {
    getHouseholdInviteTrackerMock.mockResolvedValue({
      household_draft_id: "draft-1",
      active_count: 2,
      pending_count: 0,
      failed_count: 0,
      items: [
        trackerItem({
          member_draft_id: "member-dependent-1",
          display_name: "Child One",
          dependent_user_id: "child-1",
          relationship_status: "active",
          plan_status: "active",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        }),
        trackerItem({
          member_draft_id: "member-dependent-2",
          display_name: "Child Two",
          dependent_user_id: "child-2",
          relationship_status: "active",
          plan_status: "active",
          invite_id: "invite-2",
          invite_status: "accepted",
          blocker_codes: [],
          available_actions: []
        })
      ]
    })

    render(
      <FamilyGuardrailsWizard
        initialStep={7}
        initialDraft={{
          id: "draft-1",
          owner_user_id: "guardian-1",
          name: "Home",
          mode: "family",
          status: "active",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z"
        }}
      />
    )

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Finish Setup" })).toBeInTheDocument()
    })
  })

  it("supports preset selection and seeds two guardians", async () => {
    render(<FamilyGuardrailsWizard />)

    expect(screen.getByText("Household Preset")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("radio", { name: "Two Guardians (shared household)" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })
    expect(screen.getAllByPlaceholderText(/Guardian \d+ display name/i)).toHaveLength(2)
  })

  it("uses quick dependent count from household setup when opening dependent step", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "3" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })
    expect(screen.getAllByPlaceholderText(/Child \d+ display name/i)).toHaveLength(3)
  })

  it("requires complete guardian rows before allowing continue", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    const continueButton = screen.getByRole("button", { name: /Save & Continue/i })
    expect(continueButton).toBeEnabled()

    fireEvent.click(screen.getByRole("button", { name: /Add Guardian/i }))
    expect(continueButton).toBeEnabled()
    fireEvent.click(continueButton)
    expect(screen.getByText("Display name is required.")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Guardian 2 display name")).toHaveFocus()

    fireEvent.change(screen.getByPlaceholderText("Guardian 2 display name"), {
      target: { value: "Backup Guardian" }
    })
    fireEvent.click(continueButton)

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })
  })

  it("blocks continue when guardians use duplicate user IDs", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Guardian/i }))
    fireEvent.change(screen.getByPlaceholderText("Guardian 2 display name"), {
      target: { value: "Backup Guardian" }
    })
    fireEvent.change(screen.getByLabelText("Guardian 2 user ID"), {
      target: { value: "guardian-primary" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getAllByText("Guardian user IDs must be unique before continuing.").length
      ).toBeGreaterThan(0)
    })
    expect(createHouseholdDraftMock).not.toHaveBeenCalled()
    expect(
      screen.queryByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
    ).not.toBeInTheDocument()
  })

  it("does not show toast error for guardian duplicate user IDs", async () => {
    const messageErrorSpy = vi.spyOn(message, "error")
    try {
      render(<FamilyGuardrailsWizard />)

      fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
      await waitFor(() => {
        expect(
          screen.getByText("Add every guardian who can manage alerts and safety settings.")
        ).toBeInTheDocument()
      })

      fireEvent.click(screen.getByRole("button", { name: /Add Guardian/i }))
      fireEvent.change(screen.getByPlaceholderText("Guardian 2 display name"), {
        target: { value: "Backup Guardian" }
      })
      fireEvent.change(screen.getByLabelText("Guardian 2 user ID"), {
        target: { value: "guardian-primary" }
      })

      messageErrorSpy.mockClear()
      fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

      await waitFor(() => {
        expect(
          screen.getAllByText("Guardian user IDs must be unique before continuing.").length
        ).toBeGreaterThan(0)
      })
      expect(messageErrorSpy).not.toHaveBeenCalled()
    } finally {
      messageErrorSpy.mockRestore()
    }
  })

  it("shows concrete duplicate guardian user IDs inline", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Guardian/i }))
    fireEvent.change(screen.getByPlaceholderText("Guardian 2 display name"), {
      target: { value: "Backup Guardian" }
    })
    fireEvent.change(screen.getByLabelText("Guardian 2 user ID"), {
      target: { value: "guardian-primary" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Duplicate guardian user IDs: guardian-primary")
      ).toBeInTheDocument()
    })
  })

  it("uses caregiver terminology in institutional guardian validation guidance", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("radio", { name: "Caregiver/Institutional" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Add every caregiver who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    const primaryCaregiverUserId = (screen.getByLabelText("Caregiver 1 user ID") as HTMLInputElement)
      .value
    fireEvent.click(screen.getByRole("button", { name: /Add Caregiver/i }))
    fireEvent.change(screen.getByPlaceholderText("Caregiver 2 display name"), {
      target: { value: "Backup Caregiver" }
    })
    fireEvent.change(screen.getByLabelText("Caregiver 2 user ID"), {
      target: { value: primaryCaregiverUserId }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getAllByText("Caregiver user IDs must be unique before continuing.").length
      ).toBeGreaterThan(0)
      expect(
        screen.getByText("Duplicate caregiver user IDs: caregiver-primary")
      ).toBeInTheDocument()
    })
  })

  it("uses caregiver terminology in institutional mapping and alert copy", async () => {
    saveRelationshipDraftMock.mockImplementation(
      async (
        _draftId: string,
        body: {
          guardian_member_draft_id: string
          dependent_member_draft_id: string
          relationship_type: string
          dependent_visible: boolean
        }
      ) => ({
        id: `relationship-${body.dependent_member_draft_id}`,
        household_draft_id: "draft-1",
        guardian_member_draft_id: body.guardian_member_draft_id,
        dependent_member_draft_id: body.dependent_member_draft_id,
        relationship_type: body.relationship_type,
        dependent_visible: body.dependent_visible,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      })
    )
    saveGuardrailPlanDraftMock.mockImplementation(
      async (
        _draftId: string,
        body: {
          dependent_user_id: string
          relationship_draft_id: string
          template_id: string
          overrides: { action: string; notify_context: string }
        }
      ) => ({
        id: `plan-${body.dependent_user_id}`,
        household_draft_id: "draft-1",
        dependent_user_id: body.dependent_user_id,
        relationship_draft_id: body.relationship_draft_id,
        template_id: body.template_id,
        overrides: body.overrides,
        status: "queued",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      })
    )

    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("radio", { name: "Caregiver/Institutional" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Add every caregiver who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Caregiver/i }))
    fireEvent.change(screen.getByPlaceholderText("Caregiver 2 display name"), {
      target: { value: "Backup Caregiver" }
    })
    fireEvent.change(screen.getByLabelText("Caregiver 2 user ID"), {
      target: { value: "caregiver-secondary" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText(
          "Create or link dependent accounts here. User IDs are required for invitation and acceptance."
        )
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "alex-kid" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "sam-kid" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText(
          "Map each dependent to a caregiver. For shared households, dependents can be mapped to different caregivers."
        )
      ).toBeInTheDocument()
      expect(
        screen.getByText(
          "Confirm which caregiver manages each dependent before templates are activated."
        )
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(screen.getByText("Apply a template first, then customize if needed.")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Choose how caregivers receive moderation context when alerts trigger.")
      ).toBeInTheDocument()
      expect(
        screen.getByText(
          "Choose the default moderation context caregivers should receive when alerts trigger."
        )
      ).toBeInTheDocument()
    })
  })

  it(
    "uses caregiver terminology for institutional dependent collision guidance and review summary",
    async () => {
    saveRelationshipDraftMock.mockImplementation(
      async (
        _draftId: string,
        body: {
          guardian_member_draft_id: string
          dependent_member_draft_id: string
          relationship_type: string
          dependent_visible: boolean
        }
      ) => ({
        id: `relationship-${body.dependent_member_draft_id}`,
        household_draft_id: "draft-1",
        guardian_member_draft_id: body.guardian_member_draft_id,
        dependent_member_draft_id: body.dependent_member_draft_id,
        relationship_type: body.relationship_type,
        dependent_visible: body.dependent_visible,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      })
    )
    saveGuardrailPlanDraftMock.mockImplementation(
      async (
        _draftId: string,
        body: {
          dependent_user_id: string
          relationship_draft_id: string
          template_id: string
          overrides: { action: string; notify_context: string }
        }
      ) => ({
        id: `plan-${body.dependent_user_id}`,
        household_draft_id: "draft-1",
        dependent_user_id: body.dependent_user_id,
        relationship_draft_id: body.relationship_draft_id,
        template_id: body.template_id,
        overrides: body.overrides,
        status: "queued",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z"
      })
    )

    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("radio", { name: "Caregiver/Institutional" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Add every caregiver who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText(
          "Create or link dependent accounts here. User IDs are required for invitation and acceptance."
        )
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "caregiver-primary" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "sam-kid" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Dependent user IDs must be unique and cannot match caregiver user IDs.")
      ).toBeInTheDocument()
      expect(
        screen.getByText("Dependent user IDs already used by caregivers: caregiver-primary")
      ).toBeInTheDocument()
      expect(screen.getByText("User ID cannot match a caregiver.")).toBeInTheDocument()
    })

    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "alex-kid" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(screen.getByText("Apply a template first, then customize if needed.")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Choose how caregivers receive moderation context when alerts trigger.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Invite + Acceptance Tracker" })).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Review + Activate" })).toBeInTheDocument()
      expect(screen.getByText("Caregivers:")).toBeInTheDocument()
    })
    },
    15000
  )

  it("requires complete dependent rows before allowing continue", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    const continueButton = screen.getByRole("button", { name: /Save & Continue/i })
    expect(continueButton).toBeEnabled()
    fireEvent.click(continueButton)
    expect(screen.getByPlaceholderText("Child 1 display name")).toHaveFocus()
    expect(screen.getAllByText("Display name is required.").length).toBeGreaterThan(0)

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.click(continueButton)
    expect(screen.getByPlaceholderText("Child 2 display name")).toHaveFocus()

    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.click(continueButton)

    await waitFor(() => {
      expect(
        screen.getByText("Apply a template first, then customize if needed.")
      ).toBeInTheDocument()
    })
  })

  it("blocks continue when dependents use duplicate user IDs", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "kid-shared" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "kid-shared" }
    })

    const dependentCallsBefore = addHouseholdMemberDraftMock.mock.calls.filter(
      (call): call is [string, { role: string }] => call[1]?.role === "dependent"
    ).length
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getAllByText("Dependent user IDs must be unique and cannot match guardian user IDs.").length
      ).toBeGreaterThan(0)
    })
    const dependentCallsAfter = addHouseholdMemberDraftMock.mock.calls.filter(
      (call): call is [string, { role: string }] => call[1]?.role === "dependent"
    ).length
    expect(dependentCallsAfter).toBe(dependentCallsBefore)
    expect(screen.queryByText("Apply a template first, then customize if needed.")).not.toBeInTheDocument()
  })

  it("does not show toast error for dependent duplicate user IDs", async () => {
    const messageErrorSpy = vi.spyOn(message, "error")
    try {
      render(<FamilyGuardrailsWizard />)

      fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
      fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

      await waitFor(() => {
        expect(
          screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
        ).toBeInTheDocument()
      })

      fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
        target: { value: "Alex" }
      })
      fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
        target: { value: "Sam" }
      })
      fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
        target: { value: "kid-shared" }
      })
      fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
        target: { value: "kid-shared" }
      })

      messageErrorSpy.mockClear()
      fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

      await waitFor(() => {
        expect(
          screen.getAllByText("Dependent user IDs must be unique and cannot match guardian user IDs.").length
        ).toBeGreaterThan(0)
      })
      expect(messageErrorSpy).not.toHaveBeenCalled()
    } finally {
      messageErrorSpy.mockRestore()
    }
  })

  it("shows concrete duplicate and guardian-collision dependent user IDs inline", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "3" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 3 display name"), {
      target: { value: "Riley" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "kid-shared" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "kid-shared" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[2], {
      target: { value: "guardian-primary" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Duplicate dependent user IDs: kid-shared")
      ).toBeInTheDocument()
      expect(
        screen.getByText("Dependent user IDs already used by guardians: guardian-primary")
      ).toBeInTheDocument()
    })
  })

  it("blocks continue when a dependent user ID matches a guardian user ID", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "guardian-primary" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "sam-kid" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getAllByText("Dependent user IDs must be unique and cannot match guardian user IDs.").length
      ).toBeGreaterThan(0)
    })
    expect(screen.queryByText("Apply a template first, then customize if needed.")).not.toBeInTheDocument()
  })

  it("keeps relationship mapping step for two-guardian households", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("radio", { name: "Two Guardians (shared household)" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "alex-kid" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "sam-kid" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Map each dependent to a guardian. For shared households, dependents can be mapped to different guardians.")
      ).toBeInTheDocument()
    })
  })

  it("returns to dependents when going back from templates in single-guardian flow", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "alex-kid" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "sam-kid" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByText("Apply a template first, then customize if needed.")).toBeInTheDocument()
    })
    expect(
      screen.getByText(
        "Relationship mapping was auto-applied to Primary Guardian for all dependents."
      )
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Back/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })
  })

  it("returns to relationship mapping when going back from templates in two-guardian flow", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("radio", { name: "Two Guardians (shared household)" }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[0], {
      target: { value: "alex-kid" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getAllByPlaceholderText("Child account user ID")[1], {
      target: { value: "sam-kid" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Map each dependent to a guardian. For shared households, dependents can be mapped to different guardians.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(screen.getByText("Apply a template first, then customize if needed.")).toBeInTheDocument()
    })
    expect(
      screen.queryByText(
        "Relationship mapping was auto-applied to Primary Guardian for all dependents."
      )
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Back/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Map each dependent to a guardian. For shared households, dependents can be mapped to different guardians.")
      ).toBeInTheDocument()
    })
  })

  it("auto-fills missing guardian user IDs from display names", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Guardian/i }))
    fireEvent.change(screen.getByPlaceholderText("Guardian 2 display name"), {
      target: { value: "Backup Guardian" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Auto-fill missing user IDs/i }))
    expect(screen.getByDisplayValue("backup-guardian")).toBeInTheDocument()
  })

  it("auto-fills missing guardian user IDs when continuing", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Add Guardian/i }))
    fireEvent.change(screen.getByPlaceholderText("Guardian 2 display name"), {
      target: { value: "Backup Guardian" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    expect(addHouseholdMemberDraftMock).toHaveBeenCalledWith(
      "draft-1",
      expect.objectContaining({
        role: "guardian",
        display_name: "Backup Guardian",
        user_id: "backup-guardian"
      })
    )
  })

  it("auto-fills missing dependent user IDs with unique values", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Auto-fill missing user IDs/i }))
    expect(screen.getByDisplayValue("sam")).toBeInTheDocument()
    expect(screen.getByDisplayValue("sam-2")).toBeInTheDocument()
  })

  it("auto-fills missing dependent user IDs when continuing", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("Child 1 display name"), {
      target: { value: "Sam" }
    })
    fireEvent.change(screen.getByPlaceholderText("Child 2 display name"), {
      target: { value: "Sam" }
    })

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByText("Apply a template first, then customize if needed.")).toBeInTheDocument()
    })

    const dependentCalls = addHouseholdMemberDraftMock.mock.calls.filter(
      (_call): _call is [string, { role: string; display_name: string; user_id: string }] =>
        _call[1]?.role === "dependent"
    )
    expect(dependentCalls).toHaveLength(2)
    expect(dependentCalls[0][1].user_id).toBe("sam")
    expect(dependentCalls[1][1].user_id).toBe("sam-2")
  })

  it("supports bulk dependent entry and conversion back to card mode", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Bulk entry/i }))
    fireEvent.change(
      screen.getByPlaceholderText("One per line: Display Name | user_id | email(optional)"),
      {
        target: {
          value: "Ana|ana-kid\nBen|ben-kid\nCleo|cleo-kid|cleo@example.com"
        }
      }
    )
    fireEvent.click(screen.getByRole("button", { name: /Apply bulk entries/i }))

    expect(screen.getByText("3 entries ready")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: /Card entry/i }))
    expect(screen.getAllByPlaceholderText(/Child \d+ display name/i)).toHaveLength(3)
    expect(screen.getByDisplayValue("Ana")).toBeInTheDocument()
  })

  it("auto-applies guardian bulk input when continuing without explicit apply", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Add every guardian who can manage alerts and safety settings.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Bulk entry/i }))
    fireEvent.change(
      screen.getByPlaceholderText("One per line: Display Name | user_id | email(optional)"),
      {
        target: {
          value: "Primary Guardian|guardian-primary\nBackup Guardian||backup@example.com"
        }
      }
    )

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    const guardianCalls = addHouseholdMemberDraftMock.mock.calls.filter(
      (_call): _call is [string, { role: string; display_name: string; user_id: string; email?: string }] =>
        _call[1]?.role === "guardian"
    )
    expect(guardianCalls).toHaveLength(2)
    expect(guardianCalls[0][1]).toMatchObject({
      display_name: "Primary Guardian",
      user_id: "guardian-primary"
    })
    expect(guardianCalls[1][1]).toMatchObject({
      display_name: "Backup Guardian",
      user_id: "backup-guardian",
      email: "backup@example.com"
    })
  })

  it("auto-applies dependent bulk input when continuing without explicit apply", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Bulk entry/i }))
    fireEvent.change(
      screen.getByPlaceholderText("One per line: Display Name | user_id | email(optional)"),
      {
        target: {
          value: "Ana|ana-kid\nBen||ben@example.com"
        }
      }
    )

    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(screen.getByText("Apply a template first, then customize if needed.")).toBeInTheDocument()
    })

    const dependentCalls = addHouseholdMemberDraftMock.mock.calls.filter(
      (_call): _call is [string, { role: string; display_name: string; user_id: string; email?: string }] =>
        _call[1]?.role === "dependent"
    )
    expect(dependentCalls).toHaveLength(2)
    expect(dependentCalls[0][1]).toMatchObject({
      display_name: "Ana",
      user_id: "ana-kid"
    })
    expect(dependentCalls[1][1]).toMatchObject({
      display_name: "Ben",
      user_id: "ben",
      email: "ben@example.com"
    })
  })

  it("uses compact table mode for larger dependent households", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "5" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    expect(screen.getByRole("button", { name: /Table entry/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Table entry/i }))

    fireEvent.change(screen.getByLabelText("Dependent 1 display name"), {
      target: { value: "Alex" }
    })
    fireEvent.change(screen.getByLabelText("Dependent 1 user ID"), {
      target: { value: "alex-kid" }
    })

    expect(screen.getByDisplayValue("Alex")).toBeInTheDocument()
    expect(screen.getByDisplayValue("alex-kid")).toBeInTheDocument()
  })

  it("supports bulk remove actions in dependent table mode", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "5" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Table entry/i }))
    fireEvent.click(screen.getByRole("button", { name: /Select all/i }))
    fireEvent.click(screen.getByRole("button", { name: /Remove selected/i }))

    expect(screen.getByLabelText("Dependent 1 display name")).toBeInTheDocument()
    expect(screen.getByText("Selected: 0")).toBeInTheDocument()
  })

  it("supports assigning templates to selected dependents in table mode", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "5" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Table entry/i }))
    fireEvent.click(screen.getByRole("button", { name: /Select all/i }))
    fireEvent.click(screen.getByRole("button", { name: /Apply \"Teen Balanced\" to selected/i }))

    expect(screen.getByText("Applied Teen Balanced template to 5 selected dependents.")).toBeInTheDocument()
  })

  it("supports keyboard select-all shortcut in dependent table mode", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "5" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Table entry/i }))
    expect(screen.getByText("Selected: 0")).toBeInTheDocument()

    fireEvent.keyDown(window, { key: "a", ctrlKey: true })
    expect(screen.getByText("Selected: 5")).toBeInTheDocument()

    fireEvent.keyDown(window, { key: "a", metaKey: true })
    expect(screen.getByText("Selected: 5")).toBeInTheDocument()
  })

  it("shows keyboard shortcut hints in dependent table mode", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "5" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Table entry/i }))
    expect(
      screen.getByText("Shortcuts: Ctrl/Cmd+A select all, Delete removes selected.")
    ).toBeInTheDocument()
  })

  it("uses delete shortcut for table bulk removal without triggering while typing", async () => {
    render(<FamilyGuardrailsWizard />)

    fireEvent.change(screen.getByLabelText("Dependents to set up"), {
      target: { value: "5" }
    })
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))
    fireEvent.click(screen.getByRole("button", { name: /Save & Continue/i }))

    await waitFor(() => {
      expect(
        screen.getByText("Create or link dependent accounts here. User IDs are required for invitation and acceptance.")
      ).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: /Table entry/i }))
    fireEvent.keyDown(window, { key: "a", ctrlKey: true })
    expect(screen.getByText("Selected: 5")).toBeInTheDocument()

    const firstDependentInput = screen.getByLabelText("Dependent 1 display name")
    fireEvent.focus(firstDependentInput)
    fireEvent.keyDown(firstDependentInput, { key: "Delete" })
    expect(screen.getByText("Selected: 5")).toBeInTheDocument()
    expect(screen.getByLabelText("Dependent 5 display name")).toBeInTheDocument()

    fireEvent.blur(firstDependentInput)
    fireEvent.keyDown(window, { key: "Delete" })
    expect(screen.getByText("Removed 5 selected dependents.")).toBeInTheDocument()
    expect(screen.getByText("Selected: 0")).toBeInTheDocument()
    expect(screen.queryByLabelText("Dependent 5 display name")).not.toBeInTheDocument()
  })
})
