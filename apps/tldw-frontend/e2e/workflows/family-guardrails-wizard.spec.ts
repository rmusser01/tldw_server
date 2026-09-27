import { assertNoCriticalErrors, expect, test } from "../utils/fixtures"
import { seedAuth } from "../utils/helpers"
import type { Page } from "@playwright/test"

type HouseholdStatus = "draft" | "invites_pending" | "needs_attention" | "active"
type AccountMode = "existing_account" | "invite_new"
type InviteStatus = "not_started" | "ready" | "sent" | "accepted" | "expired" | "failed"
type RelationshipStatus = "pending" | "pending_provisioning" | "active" | "declined" | "revoked"
type PlanStatus = "queued" | "active" | "failed"

type DraftRecord = {
  id: string
  owner_user_id: string
  name: string
  mode: "family" | "institutional"
  status: HouseholdStatus
  created_at: string
  updated_at: string
}

type SnapshotRecord = {
  household: DraftRecord
  members: Array<Record<string, unknown>>
  relationships: Array<Record<string, unknown>>
  plans: Array<Record<string, unknown>>
}

type TrackerItem = {
  member_draft_id: string
  display_name: string
  account_mode: AccountMode
  dependent_user_id: string | null
  relationship_draft_id: string | null
  relationship_status: RelationshipStatus | null
  plan_draft_id: string | null
  plan_status: PlanStatus | null
  invite_id: string | null
  invite_status: InviteStatus
  invite_delivery_channel: string | null
  invite_delivery_target: string | null
  invite_last_sent_at: string | null
  invite_accepted_at: string | null
  invite_expires_at: string | null
  blocker_codes: string[]
  available_actions: string[]
}

type TrackerRecord = {
  household_draft_id: string
  active_count: number
  pending_count: number
  failed_count: number
  items: TrackerItem[]
}

type RequestLog = {
  createDraftBodies: Array<Record<string, unknown>>
  updateDraftBodies: Array<Record<string, unknown>>
  memberBodies: Array<Record<string, unknown>>
  relationshipBodies: Array<Record<string, unknown>>
  planBodies: Array<Record<string, unknown>>
  bulkResendBodies: Array<Record<string, unknown>>
  provisionMemberIds: string[]
  resendInviteIds: string[]
  reissueInviteIds: string[]
}

const NOW = "2026-01-01T00:00:00Z"
const LATER = "2026-01-02T00:00:00Z"
const EXPIRY = "2026-01-08T00:00:00Z"

const buildDraft = (overrides: Partial<DraftRecord> = {}): DraftRecord => ({
  id: "draft-e2e-1",
  owner_user_id: "guardian-e2e",
  name: "My Household",
  mode: "family",
  status: "draft",
  created_at: NOW,
  updated_at: NOW,
  ...overrides
})

const buildSnapshot = ({
  draft = buildDraft(),
  dependentId = "member-dependent-1",
  dependentName = "Alex",
  dependentUserId = "alex-kid",
  dependentEmail = null,
  accountMode = "existing_account",
  relationshipStatus = "pending",
  planStatus = "queued"
}: {
  draft?: DraftRecord
  dependentId?: string
  dependentName?: string
  dependentUserId?: string | null
  dependentEmail?: string | null
  accountMode?: AccountMode
  relationshipStatus?: RelationshipStatus | null
  planStatus?: PlanStatus | null
} = {}): SnapshotRecord => ({
  household: draft,
  members: [
    {
      id: "member-guardian-1",
      household_draft_id: draft.id,
      role: "guardian",
      display_name: "Primary Guardian",
      user_id: "guardian-primary",
      email: null,
      invite_required: false,
      account_mode: "existing_account",
      provisioning_status: "not_started",
      metadata: {},
      created_at: NOW,
      updated_at: NOW
    },
    {
      id: dependentId,
      household_draft_id: draft.id,
      role: "dependent",
      display_name: dependentName,
      user_id: dependentUserId,
      email: dependentEmail,
      invite_required: true,
      account_mode: accountMode,
      provisioning_status: dependentUserId ? "not_started" : "invite_ready",
      metadata: {},
      created_at: NOW,
      updated_at: NOW
    }
  ],
  relationships:
    relationshipStatus == null
      ? []
      : [
          {
            id: "relationship-draft-1",
            household_draft_id: draft.id,
            guardian_member_draft_id: "member-guardian-1",
            dependent_member_draft_id: dependentId,
            relationship_type: "parent",
            dependent_visible: true,
            status: relationshipStatus,
            relationship_id: relationshipStatus === "active" ? "relationship-1" : null,
            created_at: NOW,
            updated_at: NOW
          }
        ],
  plans:
    planStatus == null
      ? []
      : [
          {
            id: "plan-draft-1",
            household_draft_id: draft.id,
            dependent_member_draft_id: dependentId,
            dependent_user_id: dependentUserId,
            relationship_draft_id: "relationship-draft-1",
            template_id: "default-child-safe",
            overrides: {},
            status: planStatus,
            materialized_policy_id: planStatus === "active" ? "policy-1" : null,
            failure_reason: null,
            created_at: NOW,
            updated_at: NOW
          }
        ]
})

const buildTrackerItem = (overrides: Partial<TrackerItem> = {}): TrackerItem => ({
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
  invite_last_sent_at: NOW,
  invite_accepted_at: null,
  invite_expires_at: EXPIRY,
  blocker_codes: ["invite_pending_acceptance", "plan_waiting_for_acceptance"],
  available_actions: ["resend_invite"],
  ...overrides
})

const setWizardClientConfig = async (page: Page) => {
  await page.addInitScript(() => {
    localStorage.setItem(
      "tldwConfig",
      JSON.stringify({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "single-user",
        apiKey: "e2e-family-guardrails-key"
      })
    )
  })
}

const assertWizardDiagnostics = async (
  diagnostics: {
    requestFailures: Array<{ url: string; errorText: string }>
  }
) => {
  await assertNoCriticalErrors(diagnostics as Parameters<typeof assertNoCriticalErrors>[0])
  expect(
    diagnostics.requestFailures.filter((failure) =>
      failure.url.includes("/api/v1/config/docs-info")
    )
  ).toEqual([])
}

const gotoWizard = async (page: Page) => {
  await setWizardClientConfig(page)
  await page.goto("/settings/family-guardrails")
}

const advanceToDependentsStep = async (page: Page, dependentCount = 1) => {
  await page.getByRole("spinbutton", { name: /Dependents to set up/i }).fill(String(dependentCount))
  await page.getByRole("button", { name: /Save & Continue/i }).click()
  await expect(page.getByRole("heading", { name: "Add Guardians" })).toBeVisible()
  await page.getByLabel("Guardian 1 display name").fill("Primary Guardian")
  await page.getByLabel("Guardian 1 user ID").fill("guardian-primary")
  await page.getByRole("button", { name: /Save & Continue/i }).click()
  await expect(page.getByRole("heading", { name: /Add Dependents \(Accounts\)/ })).toBeVisible()
}

const resumeLatestDraft = async (page: Page) => {
  await expect(page.getByText("Resume saved household")).toBeVisible()
  await page.getByRole("button", { name: "Resume latest draft" }).click()
}

const mockFamilyWizardApi = async (
  page: Page,
  options: {
    drafts?: DraftRecord[]
    snapshot?: SnapshotRecord | null
    tracker?: TrackerRecord
  } = {}
): Promise<RequestLog> => {
  const requestLog: RequestLog = {
    createDraftBodies: [],
    updateDraftBodies: [],
    memberBodies: [],
    relationshipBodies: [],
    planBodies: [],
    bulkResendBodies: [],
    provisionMemberIds: [],
    resendInviteIds: [],
    reissueInviteIds: []
  }

  const drafts = options.drafts ?? []
  const snapshot = options.snapshot ?? null
  const tracker =
    options.tracker ??
    {
      household_draft_id: "draft-e2e-1",
      active_count: 0,
      pending_count: 1,
      failed_count: 0,
      items: [buildTrackerItem()]
    }

  let activeDraftId = drafts[0]?.id ?? snapshot?.household.id ?? "draft-e2e-1"
  let relationshipIndex = 0
  let planIndex = 0
  let inviteIndex = 1
  const memberIndexByRole: Record<string, number> = {
    guardian: 0,
    caregiver: 0,
    dependent: 0
  }
  const memberAccountModeById = new Map<string, AccountMode>()

  await page.route("**/api/v1/config/docs-info", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        openapi_url: "/openapi.json",
        docs_url: "/docs",
        redoc_url: "/redoc",
        capabilities: {}
      })
    })
  })

  await page.route("**/openapi.json", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        openapi: "3.1.0",
        info: { title: "tldw test API", version: "1.0.0" },
        paths: {
          "/api/v1/guardian/wizard/drafts": {}
        }
      })
    })
  })

  await page.route("**/api/v1/guardian/wizard/**", async (route, request) => {
    const method = request.method()
    const url = new URL(request.url())
    const path = url.pathname

    if (method === "GET" && path === "/api/v1/guardian/wizard/drafts") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(drafts)
      })
      return
    }

    if (method === "GET" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/snapshot$/.test(path)) {
      if (!snapshot) {
        await route.fulfill({
          status: 404,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Household snapshot not found" })
        })
        return
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(snapshot)
      })
      return
    }

    if (method === "POST" && path === "/api/v1/guardian/wizard/drafts") {
      const body = request.postDataJSON() as Record<string, unknown>
      requestLog.createDraftBodies.push(body)
      activeDraftId = "draft-e2e-1"
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          buildDraft({
            id: activeDraftId,
            name: String(body.name ?? "My Household"),
            mode: (body.mode as "family" | "institutional") ?? "family"
          })
        )
      })
      return
    }

    if (method === "PATCH" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+$/.test(path)) {
      const body = request.postDataJSON() as Record<string, unknown>
      requestLog.updateDraftBodies.push(body)
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(
          buildDraft({
            id: activeDraftId,
            name: String(body.name ?? "My Household"),
            mode: (body.mode as "family" | "institutional") ?? "family",
            status: (body.status as HouseholdStatus) ?? "draft",
            updated_at: LATER
          })
        )
      })
      return
    }

    if (method === "POST" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/members$/.test(path)) {
      const body = request.postDataJSON() as Record<string, unknown>
      requestLog.memberBodies.push(body)
      const role = String(body.role ?? "guardian")
      memberIndexByRole[role] = (memberIndexByRole[role] ?? 0) + 1
      const index = memberIndexByRole[role]
      const memberId = `member-${role}-${index}`
      const accountMode = (body.account_mode as AccountMode | undefined) ?? "existing_account"
      const userId =
        body.user_id == null && role === "dependent" && accountMode === "invite_new"
          ? null
          : String(body.user_id ?? `${role}-${index}`)
      memberAccountModeById.set(memberId, accountMode)
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: memberId,
          household_draft_id: activeDraftId,
          role,
          display_name: String(body.display_name ?? `${role} ${index}`),
          user_id: userId,
          email: body.email ?? null,
          invite_required: Boolean(body.invite_required ?? role === "dependent"),
          account_mode: role === "dependent" ? accountMode : "existing_account",
          provisioning_status:
            role === "dependent"
              ? accountMode === "invite_new"
                ? "not_started"
                : "not_started"
              : "not_started",
          metadata: {},
          created_at: NOW,
          updated_at: NOW
        })
      })
      return
    }

    if (method === "POST" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/relationships$/.test(path)) {
      const body = request.postDataJSON() as Record<string, unknown>
      requestLog.relationshipBodies.push(body)
      relationshipIndex += 1
      const dependentId = String(body.dependent_member_draft_id ?? "member-dependent-1")
      const relationshipStatus: RelationshipStatus =
        memberAccountModeById.get(dependentId) === "invite_new" ? "pending_provisioning" : "pending"
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: `relationship-draft-${relationshipIndex}`,
          household_draft_id: activeDraftId,
          guardian_member_draft_id: String(
            body.guardian_member_draft_id ?? "member-guardian-1"
          ),
          dependent_member_draft_id: dependentId,
          relationship_type: String(body.relationship_type ?? "parent"),
          dependent_visible: Boolean(body.dependent_visible ?? true),
          status: relationshipStatus,
          relationship_id: null,
          created_at: NOW,
          updated_at: NOW
        })
      })
      return
    }

    if (method === "POST" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/plans$/.test(path)) {
      const body = request.postDataJSON() as Record<string, unknown>
      requestLog.planBodies.push(body)
      planIndex += 1
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: `plan-draft-${planIndex}`,
          household_draft_id: activeDraftId,
          dependent_member_draft_id: String(
            body.dependent_member_draft_id ?? "member-dependent-1"
          ),
          dependent_user_id:
            body.dependent_user_id == null ? null : String(body.dependent_user_id),
          relationship_draft_id: String(
            body.relationship_draft_id ?? "relationship-draft-1"
          ),
          template_id: String(body.template_id ?? "default-child-safe"),
          overrides: body.overrides ?? {},
          status: "queued",
          materialized_policy_id: null,
          failure_reason: null,
          created_at: NOW,
          updated_at: NOW
        })
      })
      return
    }

    if (method === "GET" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/tracker$/.test(path)) {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(tracker)
      })
      return
    }

    if (
      method === "POST" &&
      /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/members\/[^/]+\/invite\/provision$/.test(path)
    ) {
      const memberId = path.split("/").at(-3) ?? "member-dependent-1"
      requestLog.provisionMemberIds.push(memberId)
      inviteIndex += 1
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: `invite-${inviteIndex}`,
          household_draft_id: activeDraftId,
          member_draft_id: memberId,
          status: "ready",
          delivery_channel: "guardian_copy",
          delivery_target: null,
          invite_token: `token-${inviteIndex}`,
          resend_count: 0,
          last_sent_at: null,
          accepted_at: null,
          expires_at: EXPIRY,
          revoked_at: null,
          failure_reason: null,
          created_at: NOW,
          updated_at: NOW
        })
      })
      return
    }

    if (method === "POST" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/invites\/resend$/.test(path)) {
      const body = request.postDataJSON() as Record<string, unknown>
      requestLog.bulkResendBodies.push(body)
      const memberDraftIds = Array.isArray(body.member_draft_ids)
        ? body.member_draft_ids.map(String)
        : []
      const dependentUserIds = Array.isArray(body.dependent_user_ids)
        ? body.dependent_user_ids.map(String)
        : []
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          household_draft_id: activeDraftId,
          resent_count: memberDraftIds.length || dependentUserIds.length,
          skipped_count: 0,
          resent_user_ids: dependentUserIds,
          skipped_user_ids: [],
          resent_member_draft_ids: memberDraftIds,
          skipped_member_draft_ids: []
        })
      })
      return
    }

    if (method === "POST" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/invites\/[^/]+\/resend$/.test(path)) {
      const inviteId = path.split("/").at(-2) ?? "invite-1"
      requestLog.resendInviteIds.push(inviteId)
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: inviteId,
          household_draft_id: activeDraftId,
          member_draft_id: "member-dependent-1",
          status: "sent",
          delivery_channel: "email",
          delivery_target: "alex@example.com",
          invite_token: "token-1",
          resend_count: 1,
          last_sent_at: LATER,
          accepted_at: null,
          expires_at: EXPIRY,
          revoked_at: null,
          failure_reason: null,
          created_at: NOW,
          updated_at: LATER
        })
      })
      return
    }

    if (method === "POST" && /\/api\/v1\/guardian\/wizard\/drafts\/[^/]+\/invites\/[^/]+\/reissue$/.test(path)) {
      const inviteId = path.split("/").at(-2) ?? "invite-1"
      requestLog.reissueInviteIds.push(inviteId)
      inviteIndex += 1
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: `invite-${inviteIndex}`,
          household_draft_id: activeDraftId,
          member_draft_id: "member-dependent-1",
          status: "ready",
          delivery_channel: "guardian_copy",
          delivery_target: null,
          invite_token: `token-${inviteIndex}`,
          resend_count: 0,
          last_sent_at: null,
          accepted_at: null,
          expires_at: EXPIRY,
          revoked_at: null,
          failure_reason: null,
          created_at: NOW,
          updated_at: LATER
        })
      })
      return
    }

    await route.continue()
  })

  return requestLog
}

test.describe("Family Guardrails Wizard Workflow", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
  })

  test("renders the explicit household entry state and route shell", async ({
    authedPage,
    diagnostics
  }) => {
    await mockFamilyWizardApi(authedPage, { drafts: [] })
    await gotoWizard(authedPage)

    await expect(authedPage).toHaveURL(/\/settings\/family-guardrails/)
    // The settings route shell now renders its own governed h1 with the same
    // text, so scope to the wizard shell to avoid a strict-mode violation.
    await expect(
      authedPage
        .getByTestId("wizard-shell")
        .getByRole("heading", { name: /Family Guardrails Wizard/i })
    ).toBeVisible()
    await expect(authedPage.getByRole("heading", { name: "Household Basics" })).toBeVisible()
    await expect(authedPage.getByText("Step 1 of 8")).toBeVisible()
    await expect(authedPage.getByText("Next: Guardians")).toBeVisible()
    await expect(
      authedPage.getByText(
        "Start by choosing your household model. Family mode supports one or two guardians with children."
      )
    ).toBeVisible()
    await expect(authedPage.getByText("Resume saved household")).toHaveCount(0)

    await assertWizardDiagnostics(diagnostics)
  })

  test("loads a saved household through the explicit resume flow", async ({
    authedPage,
    diagnostics
  }) => {
    const savedDraft = buildDraft({
      id: "draft-resume-1",
      name: "Resume Home"
    })
    await mockFamilyWizardApi(authedPage, {
      drafts: [savedDraft],
      snapshot: buildSnapshot({
        draft: savedDraft,
        planStatus: null
      })
    })

    await gotoWizard(authedPage)
    await resumeLatestDraft(authedPage)

    await expect(
      authedPage.getByText("Apply a template first, then customize if needed.")
    ).toBeVisible()

    await authedPage.getByRole("button", { name: /^Back$/i }).click()

    await expect(
      authedPage.getByText(
        "Choose whether each dependent already has an account or needs a new invite."
      )
    ).toBeVisible()
    await expect(
      authedPage.getByRole("textbox", { name: "Dependent 1 user ID" })
    ).toHaveValue("alex-kid")

    await assertWizardDiagnostics(diagnostics)
  })

  test("supports invite-new dependents without a user ID and persists member-draft keyed plans", async ({
    authedPage,
    diagnostics
  }) => {
    const requestLog = await mockFamilyWizardApi(authedPage, { drafts: [] })

    await gotoWizard(authedPage)
    await advanceToDependentsStep(authedPage, 1)

    await authedPage.getByLabel("Dependent 1 display name").fill("Alex")
    await authedPage.getByRole("radio", { name: "Invite new" }).check({ force: true })
    await expect(
      authedPage.getByText("A new dependent account will be created when this invite is accepted.")
    ).toBeVisible()
    await authedPage.getByLabel("Dependent 1 email").fill("alex@example.com")
    await authedPage.getByRole("button", { name: /Save & Continue/i }).click()

    await expect(
      authedPage.getByText("Apply a template first, then customize if needed.")
    ).toBeVisible()

    await authedPage.getByRole("button", { name: /Save & Continue/i }).click()
    await expect(
      authedPage.getByText("Choose how guardians receive moderation context when alerts trigger.")
    ).toBeVisible()

    expect(requestLog.memberBodies).toContainEqual(
      expect.objectContaining({
        role: "dependent",
        display_name: "Alex",
        account_mode: "invite_new",
        email: "alex@example.com"
      })
    )
    expect(requestLog.memberBodies[1]).not.toHaveProperty("user_id")
    expect(requestLog.relationshipBodies).toContainEqual(
      expect.objectContaining({
        guardian_member_draft_id: "member-guardian-1",
        dependent_member_draft_id: "member-dependent-1"
      })
    )
    expect(requestLog.planBodies).toContainEqual(
      expect.objectContaining({
        dependent_member_draft_id: "member-dependent-1",
        relationship_draft_id: "relationship-draft-1",
        template_id: "default-child-safe"
      })
    )
    expect(requestLog.planBodies[0]).not.toHaveProperty("dependent_user_id")

    await assertWizardDiagnostics(diagnostics)
  })

  test("uses real bulk and per-row resend actions from the tracker", async ({
    authedPage,
    diagnostics
  }) => {
    const savedDraft = buildDraft({
      id: "draft-tracker-1",
      name: "Tracker Home",
      status: "invites_pending"
    })
    const requestLog = await mockFamilyWizardApi(authedPage, {
      drafts: [savedDraft],
      snapshot: buildSnapshot({
        draft: savedDraft,
        relationshipStatus: "pending_provisioning"
      }),
      tracker: {
        household_draft_id: savedDraft.id,
        active_count: 1,
        pending_count: 1,
        failed_count: 0,
        items: [
          buildTrackerItem({
            member_draft_id: "member-dependent-1",
            display_name: "Alex",
            relationship_status: "pending_provisioning"
          }),
          buildTrackerItem({
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
      }
    })

    await gotoWizard(authedPage)
    await resumeLatestDraft(authedPage)

    await expect(
      authedPage.getByText("1 dependent is waiting on invite acceptance.")
    ).toBeVisible()
    await expect(
      authedPage.getByRole("button", { name: "Resend Pending Invites" })
    ).toBeEnabled()
    await expect(
      authedPage.getByRole("button", { name: "Resend Invite for Alex" })
    ).toBeVisible()
    await expect(
      authedPage.getByRole("button", { name: "Copy pending invite reminder" })
    ).toHaveCount(0)

    await authedPage.getByRole("button", { name: "Resend Invite for Alex" }).click()
    await authedPage.getByRole("button", { name: "Resend Pending Invites" }).click()

    expect(requestLog.resendInviteIds).toEqual(["invite-1"])
    expect(requestLog.bulkResendBodies).toContainEqual({
      dependent_user_ids: [],
      member_draft_ids: ["member-dependent-1"]
    })

    await assertWizardDiagnostics(diagnostics)
  })

  test("shows first-class tracker blockers and create or reissue invite actions", async ({
    authedPage,
    diagnostics
  }) => {
    const savedDraft = buildDraft({
      id: "draft-track-actions-1",
      name: "Action Home",
      status: "needs_attention"
    })
    const requestLog = await mockFamilyWizardApi(authedPage, {
      drafts: [savedDraft],
      snapshot: buildSnapshot({
        draft: savedDraft,
        relationshipStatus: "pending_provisioning",
        dependentUserId: "alex-kid"
      }),
      tracker: {
        household_draft_id: savedDraft.id,
        active_count: 0,
        pending_count: 2,
        failed_count: 1,
        items: [
          buildTrackerItem({
            member_draft_id: "member-dependent-1",
            display_name: "Alex",
            invite_id: "invite-expired-1",
            invite_status: "expired",
            invite_delivery_channel: "guardian_copy",
            invite_delivery_target: null,
            invite_last_sent_at: null,
            blocker_codes: ["invite_expired", "plan_waiting_for_acceptance"],
            available_actions: ["reissue_invite"]
          }),
          buildTrackerItem({
            member_draft_id: "member-dependent-2",
            display_name: "Sam",
            account_mode: "invite_new",
            dependent_user_id: null,
            relationship_draft_id: "relationship-draft-2",
            relationship_status: "pending_provisioning",
            plan_draft_id: "plan-draft-2",
            plan_status: "queued",
            invite_id: null,
            invite_status: "not_started",
            invite_delivery_channel: "guardian_copy",
            invite_delivery_target: null,
            invite_last_sent_at: null,
            invite_accepted_at: null,
            invite_expires_at: null,
            blocker_codes: [
              "invite_not_provisioned",
              "account_not_accepted",
              "plan_waiting_for_acceptance"
            ],
            available_actions: ["provision_invite"]
          })
        ]
      }
    })

    await gotoWizard(authedPage)
    await resumeLatestDraft(authedPage)

    await expect(authedPage.getByText("Invite expired")).toBeVisible()
    await expect(authedPage.getByText("Invite not provisioned")).toBeVisible()
    await expect(authedPage.getByText("Account not accepted")).toBeVisible()
    await expect(authedPage.getByText("Guardian copy").first()).toBeVisible()
    await expect(
      authedPage.getByRole("button", { name: "Reissue Invite for Alex" })
    ).toBeVisible()
    await expect(
      authedPage.getByRole("button", { name: "Create Invite for Sam" })
    ).toBeVisible()

    await authedPage.getByRole("button", { name: "Reissue Invite for Alex" }).click()
    await authedPage.getByRole("button", { name: "Create Invite for Sam" }).click()

    expect(requestLog.reissueInviteIds).toEqual(["invite-expired-1"])
    expect(requestLog.provisionMemberIds).toEqual(["member-dependent-2"])

    await assertWizardDiagnostics(diagnostics)
  })

  test("routes tracker review-template actions back into template customization", async ({
    authedPage,
    diagnostics
  }) => {
    const savedDraft = buildDraft({
      id: "draft-track-fixes-1",
      name: "Needs Attention",
      status: "needs_attention"
    })
    await mockFamilyWizardApi(authedPage, {
      drafts: [savedDraft],
      snapshot: buildSnapshot({
        draft: savedDraft,
        relationshipStatus: "declined",
        dependentUserId: "alex-kid"
      }),
      tracker: {
        household_draft_id: savedDraft.id,
        active_count: 0,
        pending_count: 0,
        failed_count: 2,
        items: [
          buildTrackerItem({
            relationship_status: "declined",
            plan_status: "queued",
            blocker_codes: ["plan_waiting_for_acceptance"],
            available_actions: []
          }),
          buildTrackerItem({
            member_draft_id: "member-dependent-2",
            display_name: "Sam",
            dependent_user_id: "sam-kid",
            relationship_draft_id: "relationship-draft-2",
            relationship_status: "active",
            plan_draft_id: "plan-draft-2",
            plan_status: "failed",
            invite_id: "invite-2",
            invite_status: "accepted",
            blocker_codes: [],
            available_actions: []
          })
        ]
      }
    })

    await gotoWizard(authedPage)
    await resumeLatestDraft(authedPage)

    await expect(
      authedPage.getByRole("button", { name: "Review Template for Sam" })
    ).toBeVisible()

    await authedPage.getByRole("button", { name: "Review Template for Sam" }).click()
    await expect(
      authedPage.getByText("Apply a template first, then customize if needed.")
    ).toBeVisible()
    await expect(authedPage.getByText("Reviewing template for Sam.")).toBeVisible()

    await assertWizardDiagnostics(diagnostics)
  })

  test("routes tracker fix-mapping actions back into dependent editing", async ({
    authedPage,
    diagnostics
  }) => {
    const savedDraft = buildDraft({
      id: "draft-track-fix-only-1",
      name: "Needs Mapping Fix",
      status: "needs_attention"
    })
    await mockFamilyWizardApi(authedPage, {
      drafts: [savedDraft],
      snapshot: buildSnapshot({
        draft: savedDraft,
        relationshipStatus: "declined",
        dependentUserId: "alex-kid"
      }),
      tracker: {
        household_draft_id: savedDraft.id,
        active_count: 0,
        pending_count: 0,
        failed_count: 1,
        items: [
          buildTrackerItem({
            relationship_status: "declined",
            plan_status: "queued",
            blocker_codes: ["plan_waiting_for_acceptance"],
            available_actions: []
          })
        ]
      }
    })

    await gotoWizard(authedPage)
    await resumeLatestDraft(authedPage)

    await expect(
      authedPage.getByRole("button", { name: "Fix Mapping for Alex" })
    ).toBeVisible()
    await authedPage.getByRole("button", { name: "Fix Mapping for Alex" }).click()

    await expect(
      authedPage.getByText(
        "Choose whether each dependent already has an account or needs a new invite."
      )
    ).toBeVisible()
    await expect(authedPage.getByText("Fixing mapping for Alex.")).toBeVisible()
    await expect(
      authedPage.getByRole("textbox", { name: "Dependent 1 user ID" })
    ).toHaveValue("alex-kid")

    await assertWizardDiagnostics(diagnostics)
  })
})
