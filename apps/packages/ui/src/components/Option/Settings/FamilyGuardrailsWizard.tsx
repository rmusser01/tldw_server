import React, { useMemo, useState } from "react"
import {
  Alert,
  Button,
  Card,
  Divider,
  Form,
  Input,
  InputNumber,
  Radio,
  Select,
  Space,
  Steps,
  Table,
  Tag,
  Typography,
  message
} from "antd"
import type { InputRef } from "antd"
import { DeleteOutlined, PlusOutlined, ReloadOutlined } from "@ant-design/icons"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"

import {
  addHouseholdMemberDraft,
  createHouseholdDraft,
  getHouseholdDraftSnapshot,
  getHouseholdInviteTracker,
  listHouseholdDrafts,
  provisionHouseholdMemberInvite,
  reissueHouseholdMemberInvite,
  resendPendingInvites,
  resendHouseholdMemberInvite,
  saveGuardrailPlanDraft,
  saveRelationshipDraft,
  type AccountMode,
  type GuardrailPlanDraft,
  type HouseholdDraft,
  type HouseholdDraftSnapshot,
  type HouseholdInviteTracker,
  type HouseholdInviteTrackerItem,
  type HouseholdMemberDraft,
  type MemberRole,
  type RelationshipDraft,
  type SaveGuardrailPlanDraftBody,
  updateHouseholdDraft
} from "@/services/family-wizard"
import {
  trackFamilyGuardrailsWizardTelemetry,
  type FamilyGuardrailsWizardTelemetryCohort,
  type FamilyGuardrailsWizardTelemetryStep
} from "@/utils/family-guardrails-wizard-telemetry"

const { Title, Text, Paragraph } = Typography

type Mode = "family" | "institutional"
type TemplateId = "default-child-safe" | "teen-balanced" | "school-research"
type HouseholdPreset = "single_parent" | "two_guardians" | "caregiver"

type MemberInput = {
  key: string
  displayName: string
  userId: string
  email: string
  accountMode: AccountMode
}

const normalizeAccountMode = (value: unknown): AccountMode =>
  value === "invite_new" ? "invite_new" : "existing_account"

type EntryMode = "card" | "bulk"
type DependentEntryMode = EntryMode | "table"

type OverrideInput = {
  action: "block" | "redact" | "warn" | "notify"
  notify_context: "topic_only" | "snippet" | "full_message"
}

type MemberFieldKey = "displayName" | "userId"
type TrackerAction =
  | "provision_invite"
  | "resend_invite"
  | "reissue_invite"
  | "review_template"
  | "fix_mapping"
type StepDefinition = {
  title: string
  shortTitle: string
  cue: string
}

export interface FamilyGuardrailsWizardProps {
  initialStep?: number
  initialDraft?: HouseholdDraft | null
}

const STEP_DEFINITIONS: StepDefinition[] = [
  {
    title: "Household Basics",
    shortTitle: "Basics",
    cue: "Choose a household preset, set the family name, and pick how many dependents to set up."
  },
  {
    title: "Add Guardians",
    shortTitle: "Guardians",
    cue: "Add every adult account that can manage moderation alerts and safety settings."
  },
  {
    title: "Add Dependents (Accounts)",
    shortTitle: "Dependents",
    cue: "Create or link each dependent account that will receive guardrails."
  },
  {
    title: "Relationship Mapping",
    shortTitle: "Mapping",
    cue: "Confirm which guardian manages each dependent before templates are activated."
  },
  {
    title: "Templates + Customization",
    shortTitle: "Templates",
    cue: "Apply a baseline template per dependent and adjust advanced overrides if needed."
  },
  {
    title: "Alert Preferences",
    shortTitle: "Alerts",
    cue: "Choose the default moderation context guardians should receive when alerts trigger."
  },
  {
    title: "Invite + Acceptance Tracker",
    shortTitle: "Tracker",
    cue: "Track invite acceptance and guardrail activation progress, then resend pending invites."
  },
  {
    title: "Review + Activate",
    shortTitle: "Review",
    cue: "Confirm the setup summary and finish activation for your household."
  }
]

const TELEMETRY_STEP_KEYS: FamilyGuardrailsWizardTelemetryStep[] = [
  "basics",
  "guardians",
  "dependents",
  "mapping",
  "templates",
  "alerts",
  "tracker",
  "review"
]

const TEMPLATE_OPTIONS: { label: string; value: TemplateId; description: string }[] = [
  {
    value: "default-child-safe",
    label: "Default Child Safe",
    description: "Strict baseline for younger dependents."
  },
  {
    value: "teen-balanced",
    label: "Teen Balanced",
    description: "Balanced guidance with fewer hard blocks."
  },
  {
    value: "school-research",
    label: "School Research",
    description: "Education-focused with expanded research access."
  }
]

const STATUS_COLOR: Record<string, string> = {
  queued: "gold",
  active: "green",
  failed: "red",
  pending: "orange",
  pending_provisioning: "orange",
  declined: "red",
  revoked: "default",
  ready: "blue",
  sent: "cyan",
  accepted: "green",
  expired: "red",
  not_started: "default"
}

let memberSuffixCounter = 0

const secureMemberSuffix = (): string => {
  try {
    if (typeof globalThis !== "undefined" && typeof globalThis.crypto?.randomUUID === "function") {
      return globalThis.crypto.randomUUID().replace(/-/g, "").slice(0, 12)
    }
    if (typeof globalThis !== "undefined" && typeof globalThis.crypto?.getRandomValues === "function") {
      const bytes = new Uint8Array(6)
      globalThis.crypto.getRandomValues(bytes)
      return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")
    }
  } catch {
    // Fall through to deterministic fallback.
  }
  memberSuffixCounter += 1
  return `${Date.now().toString(36)}-${memberSuffixCounter.toString(36)}`
}

const newMember = (prefix: string): MemberInput => ({
  key: `${prefix}-${secureMemberSuffix()}`,
  displayName: "",
  userId: "",
  email: "",
  accountMode: "existing_account"
})

const DEFAULT_DEPENDENT_COUNT = 2
const MIN_DEPENDENTS = 1
const MAX_DEPENDENTS = 12
const LARGE_HOUSEHOLD_TABLE_THRESHOLD = 4
const BULK_ENTRY_PLACEHOLDER = "One per line: Display Name | user_id | email(optional)"
const FAMILY_WIZARD_DRAFTS_PATH = "/api/v1/guardian/wizard/drafts"
const UNSUPPORTED_FAMILY_WIZARD_STATUS_CODES = new Set([404, 405, 410, 501])

const getFamilyWizardErrorStatus = (error: unknown): number | null => {
  const status = (error as { status?: unknown } | null)?.status
  if (typeof status === "number" && Number.isFinite(status)) return status
  const message = error instanceof Error ? error.message : String(error || "")
  const matched = message.match(/\b([45]\d{2})\b/)
  if (!matched) return null
  const parsed = Number.parseInt(matched[1], 10)
  return Number.isNaN(parsed) ? null : parsed
}

const isFamilyWizardDraftsUnsupported = (error: unknown): boolean => {
  const status = getFamilyWizardErrorStatus(error)
  if (status == null) return false
  return UNSUPPORTED_FAMILY_WIZARD_STATUS_CODES.has(status)
}

const isEditableTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tagName = target.tagName
  return tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT"
}

const normalizeMemberUserId = (userId: string): string => userId.trim().toLowerCase()

const isGuardianComplete = (member: MemberInput): boolean =>
  Boolean(member.displayName.trim() && member.userId.trim())

const isDependentComplete = (member: MemberInput): boolean =>
  Boolean(member.displayName.trim() && (member.accountMode === "invite_new" || member.userId.trim()))

const findFirstIncompleteGuardianField = (
  members: MemberInput[]
): { memberKey: string; field: MemberFieldKey } | null => {
  for (const member of members) {
    if (!member.displayName.trim()) return { memberKey: member.key, field: "displayName" }
    if (!member.userId.trim()) return { memberKey: member.key, field: "userId" }
  }
  return null
}

const findFirstIncompleteDependentField = (
  members: MemberInput[]
): { memberKey: string; field: MemberFieldKey } | null => {
  for (const member of members) {
    if (!member.displayName.trim()) return { memberKey: member.key, field: "displayName" }
    if (member.accountMode === "existing_account" && !member.userId.trim()) {
      return { memberKey: member.key, field: "userId" }
    }
  }
  return null
}

const collectDuplicateUserIds = (members: MemberInput[]): Set<string> => {
  const counts = new Map<string, number>()
  members.forEach((member) => {
    const normalizedUserId = normalizeMemberUserId(member.userId)
    if (!normalizedUserId) return
    counts.set(normalizedUserId, (counts.get(normalizedUserId) ?? 0) + 1)
  })

  return new Set(
    Array.from(counts.entries())
      .filter(([, count]) => count > 1)
      .map(([normalizedUserId]) => normalizedUserId)
  )
}

const toSortedUserIdList = (userIds: Set<string>): string[] =>
  Array.from(userIds).sort((left, right) => left.localeCompare(right))

const findFirstDuplicateUserId = (
  members: MemberInput[],
  existingUserIds: Set<string> = new Set<string>()
): { memberKey: string; normalizedUserId: string } | null => {
  const seen = new Set<string>()
  for (const member of members) {
    const normalizedUserId = normalizeMemberUserId(member.userId)
    if (!normalizedUserId) continue
    if (existingUserIds.has(normalizedUserId) || seen.has(normalizedUserId)) {
      return {
        memberKey: member.key,
        normalizedUserId
      }
    }
    seen.add(normalizedUserId)
  }
  return null
}

const createGuardianMembersForPreset = (preset: HouseholdPreset): MemberInput[] => {
  if (preset === "two_guardians") {
    return [
      {
        key: "guardian-primary",
        displayName: "Primary Guardian",
        userId: "guardian-primary",
        email: "",
        accountMode: "existing_account"
      },
      {
        key: "guardian-secondary",
        displayName: "Second Guardian",
        userId: "guardian-secondary",
        email: "",
        accountMode: "existing_account"
      }
    ]
  }
  if (preset === "caregiver") {
    return [
      {
        key: "caregiver-primary",
        displayName: "Lead Caregiver",
        userId: "caregiver-primary",
        email: "",
        accountMode: "existing_account"
      }
    ]
  }
  return [
    {
      key: "guardian-primary",
      displayName: "Primary Guardian",
      userId: "guardian-primary",
      email: "",
      accountMode: "existing_account"
    }
  ]
}

const createDependents = (count: number): MemberInput[] => {
  const target = Math.max(MIN_DEPENDENTS, Math.min(MAX_DEPENDENTS, Math.floor(count)))
  return Array.from({ length: target }, () => newMember("dependent"))
}

const resizeMemberList = (members: MemberInput[], count: number, prefix: string): MemberInput[] => {
  const target = Math.max(MIN_DEPENDENTS, Math.min(MAX_DEPENDENTS, Math.floor(count)))
  if (target === members.length) return members
  if (target < members.length) return members.slice(0, target)
  return [...members, ...Array.from({ length: target - members.length }, () => newMember(prefix))]
}

const toRoleLabel = (role: MemberRole): string => {
  if (role === "guardian") return "Guardian"
  if (role === "caregiver") return "Caregiver"
  return "Dependent"
}

const toDefaultUserId = (displayName: string, fallback: string): string => {
  const candidate = displayName
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  return candidate || fallback
}

const toUniqueUserId = (base: string, usedIds: Set<string>): string => {
  let candidate = base
  let suffix = 2
  while (usedIds.has(candidate)) {
    candidate = `${base}-${suffix}`
    suffix += 1
  }
  usedIds.add(candidate)
  return candidate
}

const autofillMissingUserIds = (
  members: MemberInput[],
  prefix: "guardian" | "dependent"
): MemberInput[] => {
  const usedIds = new Set(
    members
      .filter((member) => member.accountMode !== "invite_new")
      .map((member) => member.userId.trim().toLowerCase())
      .filter(Boolean)
  )

  return members.map((member, index) => {
    if (member.accountMode === "invite_new") return member
    if (member.userId.trim()) return member
    const base = toDefaultUserId(member.displayName, `${prefix}-${index + 1}`)
    const generated = toUniqueUserId(base, usedIds)
    return {
      ...member,
      userId: generated
    }
  })
}

const parseBulkMembers = (
  input: string,
  prefix: "guardian" | "dependent"
): MemberInput[] => {
  const lines = input
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)

  return lines.map((line, index) => {
    const [displayNameRaw = "", userIdRaw = "", emailRaw = ""] = line
      .split("|")
      .map((segment) => segment.trim())
    const fallbackUserId = `${prefix}-${index + 1}`
    const displayName = displayNameRaw || userIdRaw
    const userId = userIdRaw || toDefaultUserId(displayNameRaw, fallbackUserId)

    return {
      key: newMember(prefix).key,
      displayName,
      userId,
      email: emailRaw,
      accountMode: "existing_account"
    }
  })
}

const getTrackerBlockerLabel = (code: string): string => {
  if (code === "invite_not_provisioned") return "Invite not provisioned"
  if (code === "invite_pending_acceptance") return "Waiting on acceptance"
  if (code === "invite_expired") return "Invite expired"
  if (code === "invite_failed") return "Invite failed"
  if (code === "account_not_accepted") return "Account not accepted"
  if (code === "plan_waiting_for_acceptance") return "Plan queued for acceptance"
  return code
}

const getTrackerActionLabel = (action: TrackerAction): string => {
  if (action === "provision_invite") return "Create Invite"
  if (action === "resend_invite") return "Resend Invite"
  if (action === "reissue_invite") return "Reissue Invite"
  if (action === "review_template") return "Review Template"
  return "Fix Mapping"
}

const TEMPLATE_ID_SET = new Set<TemplateId>(TEMPLATE_OPTIONS.map((option) => option.value))

const toTemplateId = (templateId: string): TemplateId => {
  if (TEMPLATE_ID_SET.has(templateId as TemplateId)) return templateId as TemplateId
  return "default-child-safe"
}

const inferHouseholdPreset = (householdMode: Mode, guardianCount: number): HouseholdPreset => {
  if (householdMode === "institutional") return "caregiver"
  if (guardianCount > 1) return "two_guardians"
  return "single_parent"
}

const resolveResumeStep = ({
  status,
  guardianCount,
  dependentCount,
  mappedDependentCount,
  plannedDependentCount
}: {
  status: string
  guardianCount: number
  dependentCount: number
  mappedDependentCount: number
  plannedDependentCount: number
}): number => {
  if (status !== "draft") return 6
  if (guardianCount === 0) return 1
  if (dependentCount === 0) return 2
  if (guardianCount > 1 && mappedDependentCount < dependentCount) return 3
  if (plannedDependentCount < dependentCount) return 4
  return 6
}

export function FamilyGuardrailsWizard({
  initialStep = 0,
  initialDraft = null
}: FamilyGuardrailsWizardProps = {}) {
  const { config: connectionConfig, loading: connectionConfigLoading } =
    useCanonicalConnectionConfig()
  const [currentStep, setCurrentStep] = useState(initialStep)
  const [submitting, setSubmitting] = useState(false)
  const [resendingInvites, setResendingInvites] = useState(false)
  const [loadingDraftList, setLoadingDraftList] = useState(!initialDraft)
  const [savedDraftsSupported, setSavedDraftsSupported] = useState<boolean | null>(
    initialDraft ? true : null
  )
  const [loadingSavedDraft, setLoadingSavedDraft] = useState(false)
  const [draft, setDraft] = useState<HouseholdDraft | null>(initialDraft)
  const [telemetryCohort, setTelemetryCohort] = useState<FamilyGuardrailsWizardTelemetryCohort>(
    initialDraft ? "existing_household" : "new_household"
  )
  const [savedDrafts, setSavedDrafts] = useState<HouseholdDraft[]>(initialDraft ? [initialDraft] : [])
  const [mode, setMode] = useState<Mode>("family")
  const [householdName, setHouseholdName] = useState("My Household")
  const [alertNotifyContext, setAlertNotifyContext] = useState<"topic_only" | "snippet" | "full_message">(
    "snippet"
  )
  const [showAdvancedOverrides, setShowAdvancedOverrides] = useState(false)
  const [inviteTracker, setInviteTracker] = useState<HouseholdInviteTracker | null>(null)
  const [householdPreset, setHouseholdPreset] = useState<HouseholdPreset>("single_parent")
  const [guardianEntryMode, setGuardianEntryMode] = useState<EntryMode>("card")
  const [dependentEntryMode, setDependentEntryMode] = useState<DependentEntryMode>("card")
  const [guardianBulkInput, setGuardianBulkInput] = useState("")
  const [dependentBulkInput, setDependentBulkInput] = useState("")
  const [guardianBulkCount, setGuardianBulkCount] = useState<number | null>(null)
  const [dependentBulkCount, setDependentBulkCount] = useState<number | null>(null)
  const [selectedDependentKeys, setSelectedDependentKeys] = useState<string[]>([])
  const [dependentTableMessage, setDependentTableMessage] = useState<string | null>(null)
  const [stepValidationAttempted, setStepValidationAttempted] = useState<Record<number, boolean>>({})

  const [guardians, setGuardians] = useState<MemberInput[]>(() =>
    createGuardianMembersForPreset("single_parent")
  )
  const [dependents, setDependents] = useState<MemberInput[]>(() =>
    createDependents(DEFAULT_DEPENDENT_COUNT)
  )

  const [guardianDraftByKey, setGuardianDraftByKey] = useState<Record<string, HouseholdMemberDraft>>({})
  const [dependentDraftByKey, setDependentDraftByKey] = useState<Record<string, HouseholdMemberDraft>>({})
  const [relationshipByDependentKey, setRelationshipByDependentKey] = useState<Record<string, RelationshipDraft>>({})
  const [planByDependentKey, setPlanByDependentKey] = useState<Record<string, GuardrailPlanDraft>>({})
  const [dependentGuardianKey, setDependentGuardianKey] = useState<Record<string, string>>({})
  const [templateByDependentKey, setTemplateByDependentKey] = useState<Record<string, TemplateId>>({})
  const [overridesByDependentKey, setOverridesByDependentKey] = useState<Record<string, OverrideInput>>({})
  const [templateReviewTargetLabel, setTemplateReviewTargetLabel] = useState<string | null>(null)
  const [mappingFixTargetLabel, setMappingFixTargetLabel] = useState<string | null>(null)
  const memberFieldInputRefs = React.useRef<Record<string, HTMLInputElement | null>>({})
  const initialDraftListLoadRef = React.useRef(false)
  const telemetryStartedRef = React.useRef(false)
  const telemetryCompletedRef = React.useRef(false)
  const lastViewedStepRef = React.useRef<FamilyGuardrailsWizardTelemetryStep | null>(null)

  const setMemberFieldInputRef = React.useCallback(
    (
      role: "guardian" | "dependent",
      memberKey: string,
      field: MemberFieldKey,
      instance: InputRef | null
    ) => {
      memberFieldInputRefs.current[`${role}:${memberKey}:${field}`] = instance?.input ?? null
    },
    []
  )

  const focusRequiredMemberField = React.useCallback(
    (role: "guardian" | "dependent", memberKey: string, field: MemberFieldKey) => {
      const target = memberFieldInputRefs.current[`${role}:${memberKey}:${field}`]
      if (!target) return
      if (typeof target.scrollIntoView === "function") {
        target.scrollIntoView({ block: "center", behavior: "smooth" })
      }
      target.focus()
    },
    []
  )

  const guardianOptions = useMemo(
    () =>
      guardians.map((guardian) => ({
        label: guardian.displayName || guardian.userId || toRoleLabel("guardian"),
        value: guardian.key
      })),
    [guardians]
  )

  const incompleteGuardianCount = useMemo(
    () => guardians.filter((guardian) => !isGuardianComplete(guardian)).length,
    [guardians]
  )

  const guardianUserIdSet = useMemo(
    () =>
      new Set(
        guardians
          .map((guardian) => normalizeMemberUserId(guardian.userId))
          .filter(Boolean)
      ),
    [guardians]
  )

  const guardianDuplicateUserIds = useMemo(
    () => collectDuplicateUserIds(guardians),
    [guardians]
  )
  const guardianDuplicateUserIdList = useMemo(
    () => toSortedUserIdList(guardianDuplicateUserIds),
    [guardianDuplicateUserIds]
  )

  const incompleteDependentCount = useMemo(
    () => dependents.filter((dependent) => !isDependentComplete(dependent)).length,
    [dependents]
  )

  const dependentDuplicateUserIds = useMemo(
    () => collectDuplicateUserIds(dependents.filter((dependent) => dependent.accountMode !== "invite_new")),
    [dependents]
  )
  const dependentDuplicateUserIdList = useMemo(
    () => toSortedUserIdList(dependentDuplicateUserIds),
    [dependentDuplicateUserIds]
  )

  const dependentGuardianCollisionUserIds = useMemo(() => {
    const collisions = new Set<string>()
    dependents.forEach((dependent) => {
      if (dependent.accountMode === "invite_new") return
      const normalizedUserId = normalizeMemberUserId(dependent.userId)
      if (!normalizedUserId) return
      if (guardianUserIdSet.has(normalizedUserId)) {
        collisions.add(normalizedUserId)
      }
    })
    return collisions
  }, [dependents, guardianUserIdSet])
  const dependentGuardianCollisionUserIdList = useMemo(
    () => toSortedUserIdList(dependentGuardianCollisionUserIds),
    [dependentGuardianCollisionUserIds]
  )

  const showGuardianInlineErrors = currentStep === 1 && stepValidationAttempted[1] === true
  const showDependentInlineErrors = currentStep === 2 && stepValidationAttempted[2] === true

  const trackerRows = useMemo<HouseholdInviteTrackerItem[]>(() => {
    if (inviteTracker?.items?.length) return inviteTracker.items
    return dependents.map((dependent) => {
      const dependentDraft = dependentDraftByKey[dependent.key]
      const relationship = relationshipByDependentKey[dependent.key]
      const plan = planByDependentKey[dependent.key]
      const accountMode = normalizeAccountMode(
        dependentDraft?.account_mode ?? dependent.accountMode
      )
      const inviteAction =
        accountMode === "invite_new" && !dependentDraft?.user_id
          ? ["provision_invite"]
          : relationship?.status === "pending_provisioning" || plan?.status === "queued"
            ? ["resend_invite"]
            : []
      const blockerCodes: string[] = []
      if (accountMode === "invite_new" && !dependentDraft?.user_id) {
        blockerCodes.push("invite_not_provisioned")
      }
      if (relationship?.status === "pending_provisioning") {
        blockerCodes.push("account_not_accepted")
      }
      if (plan?.status === "queued") {
        blockerCodes.push("plan_waiting_for_acceptance")
      }

      return {
        member_draft_id: dependentDraft?.id ?? dependent.key,
        display_name: dependent.displayName || dependent.userId || dependent.key,
        account_mode: accountMode,
        dependent_user_id:
          dependentDraft?.user_id ??
          (accountMode === "existing_account" ? dependent.userId.trim() || null : null),
        relationship_draft_id: relationship?.id ?? null,
        relationship_status: relationship?.status ?? null,
        plan_draft_id: plan?.id ?? null,
        plan_status: plan?.status ?? null,
        invite_id: null,
        invite_status: "not_started",
        invite_delivery_channel:
          accountMode === "invite_new" ? (dependent.email.trim() ? "email" : "guardian_copy") : null,
        invite_delivery_target: dependent.email.trim() || null,
        invite_last_sent_at: null,
        invite_accepted_at: null,
        invite_expires_at: null,
        blocker_codes: blockerCodes,
        available_actions: inviteAction
      }
    })
  }, [dependents, dependentDraftByKey, inviteTracker?.items, planByDependentKey, relationshipByDependentKey])

  const trackerCounts = useMemo(() => {
    if (inviteTracker) {
      return {
        active: inviteTracker.active_count,
        pending: inviteTracker.pending_count,
        failed: inviteTracker.failed_count
      }
    }
    const active = trackerRows.filter((row) => row.plan_status === "active").length
    const failed = trackerRows.filter(
      (row) => row.plan_status === "failed" || row.invite_status === "failed"
    ).length
    const pending = trackerRows.filter(
      (row) => row.blocker_codes.length > 0 || row.plan_status === "queued"
    ).length
    return { active, pending, failed }
  }, [inviteTracker, trackerRows])

  const trackerGuidance = useMemo(() => {
    if (trackerCounts.failed > 0) {
      return {
        type: "error" as const,
        message: `${trackerCounts.failed} ${trackerCounts.failed === 1 ? "dependent has" : "dependents have"} activation failures.`,
        description: "Review rows marked Failed and refresh statuses after correcting relationships or templates."
      }
    }
    if (trackerCounts.pending > 0) {
      return {
        type: "warning" as const,
        message: `${trackerCounts.pending} ${trackerCounts.pending === 1 ? "dependent is" : "dependents are"} waiting on invite acceptance.`,
        description: "Guardrails for pending dependents stay queued until acceptance."
      }
    }
    if (trackerCounts.active > 0) {
      return {
        type: "success" as const,
        message: "All dependent guardrails are active.",
        description: "No pending invite acceptances remain for this household."
      }
    }
    return {
      type: "info" as const,
      message: "Refresh statuses to load invite and activation results.",
      description: "This tracker updates when invitations are accepted and queued plans activate."
    }
  }, [trackerCounts])

  const pendingInviteMemberDraftIds = useMemo(
    () =>
      trackerRows
        .filter((row) =>
          row.available_actions.some((action) => action === "provision_invite" || action === "resend_invite")
        )
        .map((row) => row.member_draft_id),
    [trackerRows]
  )

  const trackerRowActions = useMemo<Record<string, TrackerAction[]>>(
    () =>
      Object.fromEntries(
        trackerRows.map((row) => {
          const actions = [...row.available_actions] as TrackerAction[]
          if (row.relationship_status === "declined" || row.relationship_status === "revoked") {
            actions.push("fix_mapping")
          }
          if (row.plan_status === "failed") {
            actions.push("review_template")
          }
          return [row.member_draft_id, Array.from(new Set(actions))]
        })
      ),
    [trackerRows]
  )

  const reviewGuidance = useMemo(() => {
    if (trackerCounts.failed > 0) {
      return {
        type: "error" as const,
        message: `${trackerCounts.failed} ${trackerCounts.failed === 1 ? "dependent still needs" : "dependents still need"} guardrail attention.`,
        description: "Resolve failed activation rows before considering setup fully complete."
      }
    }
    if (trackerCounts.pending > 0) {
      return {
        type: "warning" as const,
        message: `Setup is saved, and ${trackerCounts.pending} ${trackerCounts.pending === 1 ? "dependent is" : "dependents are"} still waiting on acceptance.`,
        description: "Pending dependents activate guardrails automatically after invite acceptance."
      }
    }
    if (trackerCounts.active > 0) {
      return {
        type: "success" as const,
        message: "All dependent guardrails are active.",
        description: "You can finish setup now and revisit templates or mappings anytime."
      }
    }
    return {
      type: "info" as const,
      message: "Activation summary is still loading.",
      description: "Refresh tracker statuses if this state does not update."
    }
  }, [trackerCounts])

  const stepDefinitions = useMemo<StepDefinition[]>(() => {
    if (mode !== "institutional") return STEP_DEFINITIONS
    return STEP_DEFINITIONS.map((step, index) =>
      index === 1
        ? {
            ...step,
            title: "Add Caregivers",
            shortTitle: "Caregivers",
            cue: "Add every caregiver account that can manage moderation alerts and safety settings."
          }
        : index === 3
          ? {
              ...step,
              cue: "Confirm which caregiver manages each dependent before templates are activated."
            }
          : index === 5
            ? {
                ...step,
                cue: "Choose the default moderation context caregivers should receive when alerts trigger."
              }
            : step
    )
  }, [mode])

  const currentStepDefinition = stepDefinitions[currentStep] ?? stepDefinitions[0]
  const nextStepDefinition =
    currentStep < stepDefinitions.length - 1 ? stepDefinitions[currentStep + 1] : null
  const currentTelemetryStep = TELEMETRY_STEP_KEYS[currentStep] ?? "basics"

  const trackWizardEvent = React.useCallback(
    (
      type:
        | "setup_started"
        | "draft_resumed"
        | "step_viewed"
        | "step_completed"
        | "step_error"
        | "setup_completed"
        | "drop_off",
      metadata: Record<string, unknown> = {},
      step: FamilyGuardrailsWizardTelemetryStep = currentTelemetryStep
    ) => {
      void trackFamilyGuardrailsWizardTelemetry({
        type,
        cohort: telemetryCohort,
        step,
        mode,
        guardian_count: guardians.length,
        dependent_count: dependents.length,
        ...metadata
      })
    },
    [currentTelemetryStep, dependents.length, guardians.length, mode, telemetryCohort]
  )

  const refreshInviteTracker = React.useCallback(async () => {
    if (!draft?.id) return
    try {
      const tracker = await getHouseholdInviteTracker(draft.id)
      setInviteTracker(tracker)
    } catch (error) {
      message.warning(
        error instanceof Error
          ? error.message
          : "Unable to refresh acceptance tracker"
      )
    }
  }, [draft?.id])

  const applySnapshot = React.useCallback((snapshot: HouseholdDraftSnapshot) => {
    const household = snapshot.household
    const memberDrafts = snapshot.members ?? []
    const relationshipDrafts = snapshot.relationships ?? []
    const planDrafts = snapshot.plans ?? []
    const guardianMembers = memberDrafts.filter((member) => member.role !== "dependent")
    const dependentMembers = memberDrafts.filter((member) => member.role === "dependent")

    const fallbackPreset = inferHouseholdPreset(household.mode, guardianMembers.length)
    const nextGuardians: MemberInput[] =
      guardianMembers.length > 0
        ? guardianMembers.map((member) => ({
            key: member.id,
            displayName: member.display_name,
            userId: member.user_id ?? "",
            email: member.email ?? "",
            accountMode: "existing_account"
          }))
        : createGuardianMembersForPreset(fallbackPreset)
    const nextDependents: MemberInput[] =
      dependentMembers.length > 0
        ? dependentMembers.map((member) => ({
            key: member.id,
            displayName: member.display_name,
            userId: member.user_id ?? "",
            email: member.email ?? "",
            accountMode: normalizeAccountMode(member.account_mode)
          }))
        : createDependents(DEFAULT_DEPENDENT_COUNT)

    const nextGuardianDraftByKey = Object.fromEntries(
      guardianMembers.map((member) => [member.id, member])
    )
    const nextDependentDraftByKey = Object.fromEntries(
      dependentMembers.map((member) => [member.id, member])
    )

    const relationshipById: Record<string, RelationshipDraft> = {}
    const nextRelationshipByDependentKey: Record<string, RelationshipDraft> = {}
    const nextDependentGuardianKey: Record<string, string> = {}
    relationshipDrafts.forEach((relationship) => {
      relationshipById[relationship.id] = relationship
      if (nextRelationshipByDependentKey[relationship.dependent_member_draft_id]) return
      nextRelationshipByDependentKey[relationship.dependent_member_draft_id] = relationship
      nextDependentGuardianKey[relationship.dependent_member_draft_id] =
        relationship.guardian_member_draft_id
    })

    const dependentUserIdToKey: Record<string, string> = {}
    nextDependents.forEach((dependent) => {
      const normalized = dependent.userId.trim().toLowerCase()
      if (!normalized) return
      dependentUserIdToKey[normalized] = dependent.key
    })

    const nextPlanByDependentKey: Record<string, GuardrailPlanDraft> = {}
    const nextTemplateByDependentKey: Record<string, TemplateId> = {}
    const nextOverridesByDependentKey: Record<string, OverrideInput> = {}
    planDrafts.forEach((plan) => {
      const relationship = relationshipById[plan.relationship_draft_id]
      const normalizedDependentId = (plan.dependent_user_id ?? "").trim().toLowerCase()
      const dependentKey =
        relationship?.dependent_member_draft_id ||
        dependentUserIdToKey[normalizedDependentId]
      if (!dependentKey || nextPlanByDependentKey[dependentKey]) return

      nextPlanByDependentKey[dependentKey] = plan
      nextTemplateByDependentKey[dependentKey] = toTemplateId(plan.template_id)

      const action = plan.overrides?.action
      const notifyContext = plan.overrides?.notify_context
      if (
        (action === "block" ||
          action === "redact" ||
          action === "warn" ||
          action === "notify") &&
        (notifyContext === "topic_only" ||
          notifyContext === "snippet" ||
          notifyContext === "full_message")
      ) {
        nextOverridesByDependentKey[dependentKey] = {
          action,
          notify_context: notifyContext
        }
      }
    })

    const resumeStep = resolveResumeStep({
      status: household.status,
      guardianCount: nextGuardians.length,
      dependentCount: nextDependents.length,
      mappedDependentCount: Object.keys(nextRelationshipByDependentKey).length,
      plannedDependentCount: Object.keys(nextPlanByDependentKey).length
    })

    setDraft(household)
    setMode(household.mode)
    setHouseholdName(household.name)
    setHouseholdPreset(inferHouseholdPreset(household.mode, nextGuardians.length))
    setGuardians(nextGuardians)
    setDependents(nextDependents)
    setGuardianDraftByKey(nextGuardianDraftByKey)
    setDependentDraftByKey(nextDependentDraftByKey)
    setRelationshipByDependentKey(nextRelationshipByDependentKey)
    setPlanByDependentKey(nextPlanByDependentKey)
    setDependentGuardianKey(nextDependentGuardianKey)
    setTemplateByDependentKey(nextTemplateByDependentKey)
    setOverridesByDependentKey(nextOverridesByDependentKey)
    setTemplateReviewTargetLabel(null)
    setMappingFixTargetLabel(null)
    setGuardianEntryMode("card")
    setDependentEntryMode(
      nextDependents.length >= LARGE_HOUSEHOLD_TABLE_THRESHOLD ? "table" : "card"
    )
    setSelectedDependentKeys([])
    setDependentTableMessage(null)
    setInviteTracker(null)
    setSavedDrafts((prev) => {
      const filtered = prev.filter((item) => item.id !== household.id)
      return [household, ...filtered]
    })
    setCurrentStep(resumeStep)
  }, [])

  React.useEffect(() => {
    if (initialDraft) {
      setSavedDraftsSupported(true)
      setLoadingDraftList(false)
      return
    }
    if (connectionConfigLoading) return

    const serverUrl = connectionConfig?.serverUrl?.trim()
    if (!serverUrl) {
      setSavedDraftsSupported(true)
      return
    }

    let cancelled = false

    const probeDraftEndpointSupport = async () => {
      try {
        const response = await fetch(`${serverUrl}/openapi.json`)
        if (!response.ok) {
          if (!cancelled) {
            setSavedDraftsSupported(true)
          }
          return
        }
        const spec = await response.json()
        const paths =
          spec && typeof spec === "object" && spec.paths && typeof spec.paths === "object"
            ? (spec.paths as Record<string, unknown>)
            : null
        if (!cancelled) {
          setSavedDraftsSupported(Boolean(paths && FAMILY_WIZARD_DRAFTS_PATH in paths))
        }
      } catch {
        if (!cancelled) {
          setSavedDraftsSupported(true)
        }
      }
    }

    void probeDraftEndpointSupport()

    return () => {
      cancelled = true
    }
  }, [connectionConfig?.serverUrl, connectionConfigLoading, initialDraft])

  React.useEffect(() => {
    if (initialDraft) return
    if (connectionConfigLoading) return
    if (savedDraftsSupported === null) return
    if (savedDraftsSupported === false) {
      setLoadingDraftList(false)
      return
    }
    if (initialDraftListLoadRef.current) return
    initialDraftListLoadRef.current = true

    let cancelled = false

    const loadDraftEntries = async () => {
      try {
        setLoadingDraftList(true)
        const drafts = await listHouseholdDrafts()
        if (cancelled) return
        setSavedDrafts(drafts)
      } catch (error) {
        if (cancelled) return
        if (isFamilyWizardDraftsUnsupported(error)) {
          setSavedDraftsSupported(false)
          setSavedDrafts([])
          return
        }
        console.error("Failed to load family wizard drafts:", error)
      } finally {
        if (!cancelled) {
          setLoadingDraftList(false)
        }
      }
    }

    void loadDraftEntries()
    return () => {
      cancelled = true
    }
  }, [connectionConfigLoading, initialDraft, savedDraftsSupported])

  React.useEffect(() => {
    if (telemetryStartedRef.current) return
    telemetryStartedRef.current = true
    trackWizardEvent("setup_started", {
      initial_draft: Boolean(initialDraft)
    })
  }, [initialDraft, trackWizardEvent])

  React.useEffect(() => {
    if (lastViewedStepRef.current === currentTelemetryStep) return
    lastViewedStepRef.current = currentTelemetryStep
    trackWizardEvent("step_viewed")
  }, [currentTelemetryStep, trackWizardEvent])

  React.useEffect(() => {
    return () => {
      if (telemetryCompletedRef.current) return
      if (!telemetryStartedRef.current) return
      void trackFamilyGuardrailsWizardTelemetry({
        type: "drop_off",
        cohort: telemetryCohort,
        step: lastViewedStepRef.current || currentTelemetryStep,
        mode,
        guardian_count: guardians.length,
        dependent_count: dependents.length
      })
    }
  }, [currentTelemetryStep, dependents.length, guardians.length, mode, telemetryCohort])

  const handleLoadSavedDraft = React.useCallback(
    async (draftId: string) => {
      try {
        setLoadingSavedDraft(true)
        const snapshot = await getHouseholdDraftSnapshot(draftId)
        setTelemetryCohort("existing_household")
        applySnapshot(snapshot)
        void trackFamilyGuardrailsWizardTelemetry({
          type: "draft_resumed",
          cohort: "existing_household",
          step: TELEMETRY_STEP_KEYS[resolveResumeStep({
            status: snapshot.household.status,
            guardianCount: snapshot.members.filter((member) => member.role !== "dependent").length,
            dependentCount: snapshot.members.filter((member) => member.role === "dependent").length,
            mappedDependentCount: snapshot.relationships.length,
            plannedDependentCount: snapshot.plans.length
          })] || "review",
          household_status: snapshot.household.status,
          household_mode: snapshot.household.mode
        })
      } catch (error) {
        message.error(error instanceof Error ? error.message : "Unable to load saved household")
      } finally {
        setLoadingSavedDraft(false)
      }
    },
    [applySnapshot]
  )

  const resendInvitesForTargets = async (memberDraftIds: string[]) => {
    if (!draft?.id) {
      message.info("Save household setup before resending invites.")
      return
    }
    const targets = Array.from(
      new Set(
        memberDraftIds
          .map((id) => id.trim())
          .filter(Boolean)
      )
    )
    if (!targets.length) {
      message.info("No pending invites to resend.")
      return
    }

    try {
      setResendingInvites(true)
      const result = await resendPendingInvites(draft.id, {
        dependent_user_ids: [],
        member_draft_ids: targets
      })
      if (result.resent_count > 0) {
        message.success(
          `Resent ${result.resent_count} pending invite${result.resent_count === 1 ? "" : "s"}.`
        )
      } else {
        message.info("No pending invites were eligible for resend.")
      }
      await refreshInviteTracker()
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Unable to resend pending invites")
    } finally {
      setResendingInvites(false)
    }
  }

  const handleResendPendingInvites = async () => {
    await resendInvitesForTargets(pendingInviteMemberDraftIds)
  }

  const handleTrackerRowAction = async (
    row: HouseholdInviteTrackerItem,
    action: TrackerAction
  ) => {
    if (!draft?.id && (action === "provision_invite" || action === "resend_invite" || action === "reissue_invite")) {
      message.info("Save household setup before managing invites.")
      return
    }

    try {
      if (action === "provision_invite") {
        setResendingInvites(true)
        await provisionHouseholdMemberInvite(draft!.id, row.member_draft_id)
        message.success(`Invite created for ${row.display_name}.`)
        await refreshInviteTracker()
        return
      }
      if (action === "resend_invite") {
        if (row.invite_id) {
          setResendingInvites(true)
          await resendHouseholdMemberInvite(draft!.id, row.invite_id)
          message.success(`Invite resent for ${row.display_name}.`)
          await refreshInviteTracker()
          return
        }
        await resendInvitesForTargets([row.member_draft_id])
        return
      }
      if (action === "reissue_invite") {
        if (!row.invite_id) {
          message.info("No expired invite is available to reissue.")
          return
        }
        setResendingInvites(true)
        await reissueHouseholdMemberInvite(draft!.id, row.invite_id)
        message.success(`Invite reissued for ${row.display_name}.`)
        await refreshInviteTracker()
        return
      }
      if (action === "review_template") {
        setMappingFixTargetLabel(null)
        setTemplateReviewTargetLabel(row.display_name)
        setCurrentStep(4)
        return
      }
      if (action === "fix_mapping") {
        setMappingFixTargetLabel(row.display_name)
        setTemplateReviewTargetLabel(null)
        setCurrentStep(guardians.length <= 1 ? 2 : 3)
      }
    } catch (error) {
      message.error(error instanceof Error ? error.message : "Unable to update invite state")
    } finally {
      setResendingInvites(false)
    }
  }

  React.useEffect(() => {
    if (currentStep >= 6 && draft?.id) {
      void refreshInviteTracker()
    }
  }, [currentStep, draft?.id, refreshInviteTracker])

  React.useEffect(() => {
    if (
      dependents.length >= LARGE_HOUSEHOLD_TABLE_THRESHOLD &&
      dependentEntryMode === "card"
    ) {
      setDependentEntryMode("table")
    }
  }, [dependentEntryMode, dependents.length])

  const ensureDraft = async (): Promise<HouseholdDraft> => {
    if (!householdName.trim()) {
      throw new Error("Household name is required")
    }
    if (draft) {
      const updated = await updateHouseholdDraft(draft.id, {
        name: householdName.trim(),
        mode
      })
      setDraft(updated)
      return updated
    }
    const created = await createHouseholdDraft({
      name: householdName.trim(),
      mode
    })
    setDraft(created)
    return created
  }

  const persistMembers = async (
    role: MemberRole,
    members: MemberInput[],
    draftId: string,
    existing: Record<string, HouseholdMemberDraft>,
    setExisting: React.Dispatch<React.SetStateAction<Record<string, HouseholdMemberDraft>>>
  ): Promise<Record<string, HouseholdMemberDraft>> => {
    const next = { ...existing }
    for (const member of members) {
      if (next[member.key]) continue
      const requiresUserId = role !== "dependent" || member.accountMode === "existing_account"
      if (!member.displayName.trim() || (requiresUserId && !member.userId.trim())) {
        throw new Error(
          requiresUserId
            ? `${toRoleLabel(role)} display name and user ID are required`
            : `${toRoleLabel(role)} display name is required`
        )
      }
      const created = await addHouseholdMemberDraft(draftId, {
        role,
        display_name: member.displayName.trim(),
        user_id: requiresUserId ? member.userId.trim() : undefined,
        email: member.email.trim() || undefined,
        invite_required: role === "dependent",
        account_mode: role === "dependent" ? member.accountMode : "existing_account",
        provisioning_status: role === "dependent" ? "not_started" : undefined
      })
      next[member.key] = created
    }
    setExisting(next)
    return next
  }

  const persistRelationships = async (
    draftId: string,
    guardianDrafts: Record<string, HouseholdMemberDraft> = guardianDraftByKey,
    dependentDrafts: Record<string, HouseholdMemberDraft> = dependentDraftByKey,
    dependentMembers: MemberInput[] = dependents
  ) => {
    const next = { ...relationshipByDependentKey }
    for (const dependent of dependentMembers) {
      if (next[dependent.key]) continue
      const dependentDraft = dependentDrafts[dependent.key]
      const preferredGuardianKey = dependentGuardianKey[dependent.key] || guardians[0]?.key
      const guardianDraft = preferredGuardianKey ? guardianDrafts[preferredGuardianKey] : null
      if (!dependentDraft || !guardianDraft) {
        throw new Error("Complete guardian and dependent account setup before mapping relationships")
      }
      const created = await saveRelationshipDraft(draftId, {
        guardian_member_draft_id: guardianDraft.id,
        dependent_member_draft_id: dependentDraft.id,
        relationship_type: mode === "institutional" ? "institutional" : "parent",
        dependent_visible: true
      })
      next[dependent.key] = created
    }
    setRelationshipByDependentKey(next)
  }

  const persistPlans = async (draftId: string) => {
    const next = { ...planByDependentKey }
    for (const dependent of dependents) {
      if (next[dependent.key]) continue
      const dependentDraft = dependentDraftByKey[dependent.key]
      const relationshipDraft = relationshipByDependentKey[dependent.key]
      if (!dependentDraft || !relationshipDraft) {
        throw new Error("Relationship mapping is required before plan setup")
      }
      const template = templateByDependentKey[dependent.key] ?? "default-child-safe"
      const overrides = overridesByDependentKey[dependent.key] ?? {
        action: "block",
        notify_context: alertNotifyContext
      }
      const payload: SaveGuardrailPlanDraftBody = {
        dependent_member_draft_id: dependentDraft.id,
        dependent_user_id: dependentDraft.user_id || dependent.userId.trim() || undefined,
        relationship_draft_id: relationshipDraft.id,
        template_id: template,
        overrides
      }
      const created = await saveGuardrailPlanDraft(draftId, payload)
      next[dependent.key] = created
    }
    setPlanByDependentKey(next)
  }

  const handleNext = async () => {
    try {
      setSubmitting(true)

      if (currentStep === 0) {
        if (!householdName.trim()) {
          message.error("Household name is required")
          return
        }
        trackWizardEvent("step_completed", { household_name: householdName.trim() })
        setCurrentStep(1)
        return
      }

      let nextGuardians = guardians
      let nextDependents = dependents

      if (currentStep === 1) {
        if (guardianEntryMode === "bulk" && guardianBulkInput.trim()) {
          const parsedGuardians = parseBulkMembers(guardianBulkInput, "guardian")
          if (!parsedGuardians.length) {
            message.error("Enter at least one guardian bulk entry before continuing.")
            return
          }
          nextGuardians = parsedGuardians
          setGuardians(parsedGuardians)
          setGuardianBulkCount(parsedGuardians.length)
        }
        nextGuardians = autofillMissingUserIds(nextGuardians, "guardian")
        if (nextGuardians !== guardians) {
          setGuardians(nextGuardians)
        }
        setStepValidationAttempted((prev) => ({ ...prev, 1: true }))
        const firstIncomplete = findFirstIncompleteGuardianField(nextGuardians)
        if (firstIncomplete) {
          focusRequiredMemberField("guardian", firstIncomplete.memberKey, firstIncomplete.field)
          return
        }
        const duplicateGuardian = findFirstDuplicateUserId(nextGuardians)
        if (duplicateGuardian) {
          focusRequiredMemberField("guardian", duplicateGuardian.memberKey, "userId")
          return
        }
      }

      if (currentStep === 2) {
        if (dependentEntryMode === "bulk" && dependentBulkInput.trim()) {
          const parsedDependents = parseBulkMembers(dependentBulkInput, "dependent")
          if (!parsedDependents.length) {
            message.error("Enter at least one dependent bulk entry before continuing.")
            return
          }
          nextDependents = parsedDependents
          setDependents(parsedDependents)
          setDependentBulkCount(parsedDependents.length)
          setSelectedDependentKeys([])
          setDependentTableMessage(null)
        }
        nextDependents = autofillMissingUserIds(nextDependents, "dependent")
        if (nextDependents !== dependents) {
          setDependents(nextDependents)
        }
        setStepValidationAttempted((prev) => ({ ...prev, 2: true }))
        const firstIncomplete = findFirstIncompleteDependentField(nextDependents)
        if (firstIncomplete) {
          focusRequiredMemberField("dependent", firstIncomplete.memberKey, firstIncomplete.field)
          return
        }
        const guardianUserIds = new Set(
          nextGuardians
            .map((guardian) => normalizeMemberUserId(guardian.userId))
            .filter(Boolean)
        )
        const duplicateDependent = findFirstDuplicateUserId(
          nextDependents.filter((dependent) => dependent.accountMode !== "invite_new"),
          guardianUserIds
        )
        if (duplicateDependent) {
          focusRequiredMemberField("dependent", duplicateDependent.memberKey, "userId")
          return
        }
      }

      const ensuredDraft = await ensureDraft()
      let nextGuardianDraftByKey = guardianDraftByKey
      let nextDependentDraftByKey = dependentDraftByKey

      if (currentStep === 1) {
        nextGuardianDraftByKey = await persistMembers(
          "guardian",
          nextGuardians,
          ensuredDraft.id,
          guardianDraftByKey,
          setGuardianDraftByKey
        )
      }
      if (currentStep === 2) {
        nextDependentDraftByKey = await persistMembers(
          "dependent",
          nextDependents,
          ensuredDraft.id,
          dependentDraftByKey,
          setDependentDraftByKey
        )
        if (nextGuardians.length <= 1) {
          await persistRelationships(
            ensuredDraft.id,
            nextGuardianDraftByKey,
            nextDependentDraftByKey,
            nextDependents
          )
          trackWizardEvent("step_completed", {
            existing_account_count: nextDependents.filter(
              (dependent) => dependent.accountMode === "existing_account"
            ).length,
            invite_new_count: nextDependents.filter(
              (dependent) => dependent.accountMode === "invite_new"
            ).length
          })
          setCurrentStep(4)
          return
        }
      }
      if (currentStep === 3) {
        await persistRelationships(ensuredDraft.id)
      }
      if (currentStep === 4) {
        setTemplateReviewTargetLabel(null)
        setMappingFixTargetLabel(null)
        await persistPlans(ensuredDraft.id)
      }
      if (currentStep === 6) {
        await refreshInviteTracker()
      }
      if (currentStep < stepDefinitions.length - 1) {
        trackWizardEvent("step_completed")
        setCurrentStep((step) => step + 1)
      } else {
        telemetryCompletedRef.current = true
        trackWizardEvent("setup_completed", {
          active_count: trackerCounts.active,
          pending_count: trackerCounts.pending,
          failed_count: trackerCounts.failed,
          existing_account_count: dependents.filter(
            (dependent) => dependent.accountMode === "existing_account"
          ).length,
          invite_new_count: dependents.filter((dependent) => dependent.accountMode === "invite_new").length
        })
        message.success("Family guardrails wizard setup saved.")
      }
    } catch (error) {
      trackWizardEvent("step_error", {
        error_message: error instanceof Error ? error.message : "unknown"
      })
      message.error(error instanceof Error ? error.message : "Unable to continue wizard")
    } finally {
      setSubmitting(false)
    }
  }

  const handleBack = () =>
    setCurrentStep((step) => {
      if (step === 4 && guardians.length <= 1) {
        return 2
      }
      return Math.max(0, step - 1)
    })

  const applyTemplateToAll = (template: TemplateId) => {
    const next: Record<string, TemplateId> = {}
    dependents.forEach((dependent) => {
      next[dependent.key] = template
    })
    setTemplateByDependentKey(next)
  }

  const applyBulkMembers = (role: "guardian" | "dependent") => {
    const source = role === "guardian" ? guardianBulkInput : dependentBulkInput
    const parsed = parseBulkMembers(source, role)
    if (!parsed.length) {
      message.error("Enter at least one line before applying bulk entries.")
      return
    }
    if (role === "guardian") {
      setGuardians(parsed)
      setGuardianBulkCount(parsed.length)
      return
    }
    setDependents(parsed)
    setDependentBulkCount(parsed.length)
    setDependentTableMessage(null)
    setSelectedDependentKeys([])
  }

  const applyHouseholdPreset = (preset: HouseholdPreset) => {
    setHouseholdPreset(preset)
    setMode(preset === "caregiver" ? "institutional" : "family")
    setGuardians(createGuardianMembersForPreset(preset))
    setDependents(createDependents(DEFAULT_DEPENDENT_COUNT))
    setGuardianDraftByKey({})
    setDependentDraftByKey({})
    setRelationshipByDependentKey({})
    setPlanByDependentKey({})
    setDependentGuardianKey({})
    setTemplateByDependentKey({})
    setOverridesByDependentKey({})
    setMappingFixTargetLabel(null)
    setTemplateReviewTargetLabel(null)
    setGuardianEntryMode("card")
    setDependentEntryMode("card")
    setGuardianBulkInput("")
    setDependentBulkInput("")
    setGuardianBulkCount(null)
    setDependentBulkCount(null)
    setSelectedDependentKeys([])
    setDependentTableMessage(null)
    setInviteTracker(null)
  }

  const guardianEntityLabel = mode === "institutional" ? "Caregiver" : "Guardian"
  const guardianEntityLabelLower = guardianEntityLabel.toLowerCase()
  const guardianEntityLabelPluralLower =
    guardianEntityLabelLower === "caregiver" ? "caregivers" : "guardians"
  const primaryGuardianName = useMemo(() => {
    const firstGuardian = guardians[0]
    if (!firstGuardian) return "Primary Guardian"
    const displayName = firstGuardian.displayName.trim()
    const userId = firstGuardian.userId.trim()
    return displayName || userId || "Primary Guardian"
  }, [guardians])

  const updateDependentMember = (
    dependentKey: string,
    patch: Partial<MemberInput>
  ) => {
    setDependents((prev) =>
      prev.map((item) => {
        if (item.key !== dependentKey) return item
        const next = { ...item, ...patch }
        if (patch.accountMode === "invite_new") {
          next.userId = ""
        }
        return next
      })
    )
  }

  const pruneRecordByKeys = <T,>(
    record: Record<string, T>,
    keysToRemove: Set<string>
  ): Record<string, T> => {
    const next: Record<string, T> = {}
    Object.entries(record).forEach(([key, value]) => {
      if (keysToRemove.has(key)) return
      next[key] = value
    })
    return next
  }

  const removeDependentsByKeys = (keys: string[]) => {
    if (!keys.length) return
    const removeSet = new Set(keys)
    setDependents((prev) => {
      const remaining = prev.filter((dependent) => !removeSet.has(dependent.key))
      if (remaining.length >= MIN_DEPENDENTS) return remaining
      return [
        ...remaining,
        ...Array.from(
          { length: MIN_DEPENDENTS - remaining.length },
          () => newMember("dependent")
        )
      ]
    })
    setDependentDraftByKey((prev) => pruneRecordByKeys(prev, removeSet))
    setRelationshipByDependentKey((prev) => pruneRecordByKeys(prev, removeSet))
    setPlanByDependentKey((prev) => pruneRecordByKeys(prev, removeSet))
    setDependentGuardianKey((prev) => pruneRecordByKeys(prev, removeSet))
    setTemplateByDependentKey((prev) => pruneRecordByKeys(prev, removeSet))
    setOverridesByDependentKey((prev) => pruneRecordByKeys(prev, removeSet))
    setSelectedDependentKeys((prev) => prev.filter((key) => !removeSet.has(key)))
  }

  const selectAllDependents = () => {
    setSelectedDependentKeys(dependents.map((dependent) => dependent.key))
    setDependentTableMessage(null)
  }

  const clearDependentSelection = () => {
    setSelectedDependentKeys([])
    setDependentTableMessage(null)
  }

  const removeSelectedDependents = () => {
    const count = selectedDependentKeys.length
    removeDependentsByKeys(selectedDependentKeys)
    setSelectedDependentKeys([])
    if (count > 0) {
      setDependentTableMessage(`Removed ${count} selected dependents.`)
      return
    }
    setDependentTableMessage("Select at least one dependent before removing.")
  }

  const autofillMissingGuardianUserIds = () => {
    setGuardians((prev) => autofillMissingUserIds(prev, "guardian"))
  }

  const autofillMissingDependentUserIds = () => {
    setDependents((prev) => autofillMissingUserIds(prev, "dependent"))
  }

  const applyTemplateToSelectedDependents = (template: TemplateId) => {
    if (!selectedDependentKeys.length) {
      setDependentTableMessage("Select at least one dependent before applying templates.")
      return
    }
    setTemplateByDependentKey((prev) => {
      const next = { ...prev }
      selectedDependentKeys.forEach((key) => {
        next[key] = template
      })
      return next
    })
    const label =
      TEMPLATE_OPTIONS.find((option) => option.value === template)?.label || template
    setDependentTableMessage(
      `Applied ${label} template to ${selectedDependentKeys.length} selected dependents.`
    )
  }

  React.useEffect(() => {
    if (currentStep !== 2 || dependentEntryMode !== "table") return

    const handleTableKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return

      const key = event.key.toLowerCase()
      if ((event.ctrlKey || event.metaKey) && key === "a") {
        event.preventDefault()
        selectAllDependents()
        return
      }

      if (key === "delete") {
        event.preventDefault()
        removeSelectedDependents()
      }
    }

    window.addEventListener("keydown", handleTableKeyDown)
    return () => window.removeEventListener("keydown", handleTableKeyDown)
  }, [currentStep, dependentEntryMode, removeSelectedDependents, selectAllDependents])

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        const latestSavedDraft = savedDrafts[0] ?? null
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            {savedDraftsSupported === false ? (
              <Alert
                showIcon
                type="info"
                title="Saved household drafts unavailable"
                description="This server does not expose family wizard draft endpoints yet. Start a new household to continue."
              />
            ) : loadingDraftList ? (
              <Alert
                showIcon
                type="info"
                title="Checking for saved household drafts..."
              />
            ) : savedDrafts.length ? (
              <Card size="small" title="Resume saved household">
                <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                  <Text type="secondary">
                    Resume the latest draft or reopen any saved household before starting from scratch.
                  </Text>
                  <Space wrap>
                    <Button
                      loading={loadingSavedDraft}
                      onClick={() =>
                        latestSavedDraft && void handleLoadSavedDraft(latestSavedDraft.id)
                      }
                    >
                      Resume latest draft
                    </Button>
                  </Space>
                  <Table
                    rowKey="id"
                    pagination={false}
                    dataSource={savedDrafts}
                    columns={[
                      {
                        title: "Household",
                        dataIndex: "name"
                      },
                      {
                        title: "Status",
                        dataIndex: "status",
                        render: (value: string) => <Tag color={STATUS_COLOR[value] || "default"}>{value}</Tag>
                      },
                      {
                        title: "Updated",
                        dataIndex: "updated_at"
                      },
                      {
                        title: "Action",
                        render: (_value: unknown, row: HouseholdDraft) => (
                          <Button
                            size="small"
                            loading={loadingSavedDraft}
                            onClick={() => void handleLoadSavedDraft(row.id)}
                          >
                            Edit household
                          </Button>
                        )
                      }
                    ]}
                  />
                </Space>
              </Card>
            ) : null}
            <Paragraph type="secondary">
              Start by choosing your household model. Family mode supports one or two guardians with children.
              Institutional mode supports caregivers and classroom-style setups.
            </Paragraph>
            <Form layout="vertical">
              <Form.Item label="Household Preset">
                <Radio.Group
                  value={householdPreset}
                  onChange={(event) => applyHouseholdPreset(event.target.value as HouseholdPreset)}
                >
                  <Space orientation="vertical">
                    <Radio value="single_parent">Single Parent (recommended)</Radio>
                    <Radio value="two_guardians">Two Guardians (shared household)</Radio>
                    <Radio value="caregiver">Caregiver/Institutional</Radio>
                  </Space>
                </Radio.Group>
              </Form.Item>
              <Form.Item label="Household Name" required>
                <Input
                  value={householdName}
                  onChange={(event) => setHouseholdName(event.target.value)}
                  placeholder="e.g. Rivera Family"
                />
              </Form.Item>
              <Form.Item label="Dependents to set up">
                <InputNumber
                  min={MIN_DEPENDENTS}
                  max={MAX_DEPENDENTS}
                  value={dependents.length}
                  onChange={(value) =>
                    setDependents((prev) =>
                      resizeMemberList(prev, value == null ? MIN_DEPENDENTS : value, "dependent")
                    )
                  }
                  aria-label="Dependents to set up"
                />
              </Form.Item>
            </Form>
          </Space>
        )
      case 1:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type="info"
              title={
                mode === "institutional"
                  ? "Add every caregiver who can manage alerts and safety settings."
                  : "Add every guardian who can manage alerts and safety settings."
              }
            />
            <Text type="secondary">
              {mode === "institutional"
                ? "Use each caregiver's existing account user ID (the one used to sign in)."
                : "Use each guardian's existing account user ID (the one used to sign in)."}
            </Text>
            {showGuardianInlineErrors && incompleteGuardianCount > 0 ? (
              <Text type="secondary">
                {`Complete display name and user ID for ${incompleteGuardianCount} ${
                  incompleteGuardianCount === 1
                    ? guardianEntityLabelLower
                    : guardianEntityLabelPluralLower
                } to continue.`}
              </Text>
            ) : null}
            {showGuardianInlineErrors && guardianDuplicateUserIds.size > 0 ? (
              <Text type="danger">{`${guardianEntityLabel} user IDs must be unique before continuing.`}</Text>
            ) : null}
            {showGuardianInlineErrors && guardianDuplicateUserIdList.length > 0 ? (
              <Text type="secondary">
                {`Duplicate ${guardianEntityLabelLower} user IDs: ${guardianDuplicateUserIdList.join(", ")}`}
              </Text>
            ) : null}
            <Space>
              <Button
                type={guardianEntryMode === "card" ? "primary" : "default"}
                onClick={() => setGuardianEntryMode("card")}
              >
                Card entry
              </Button>
              <Button
                type={guardianEntryMode === "bulk" ? "primary" : "default"}
                onClick={() => setGuardianEntryMode("bulk")}
              >
                Bulk entry
              </Button>
              <Button
                onClick={autofillMissingGuardianUserIds}
                disabled={incompleteGuardianCount === 0}
              >
                Auto-fill missing user IDs
              </Button>
            </Space>
            {guardianEntryMode === "bulk" ? (
              <Space orientation="vertical" style={{ width: "100%" }}>
                <Input.TextArea
                  value={guardianBulkInput}
                  onChange={(event) => setGuardianBulkInput(event.target.value)}
                  placeholder={BULK_ENTRY_PLACEHOLDER}
                  rows={6}
                />
                <Button onClick={() => applyBulkMembers("guardian")}>Apply bulk entries</Button>
                {guardianBulkCount != null ? (
                  <Text type="secondary">{`${guardianBulkCount} entries ready`}</Text>
                ) : null}
              </Space>
            ) : (
              <>
                {guardians.map((guardian, index) => {
                  const normalizedGuardianUserId = normalizeMemberUserId(guardian.userId)
                  const guardianHasDuplicateUserId =
                    normalizedGuardianUserId.length > 0 &&
                    guardianDuplicateUserIds.has(normalizedGuardianUserId)

                  return (
                    <Card key={guardian.key} size="small">
                      <Space orientation="vertical" style={{ width: "100%" }}>
                        <Text type="secondary">{`${guardianEntityLabel} ${index + 1} display name`}</Text>
                        <Input
                          ref={(instance: InputRef | null) =>
                            setMemberFieldInputRef("guardian", guardian.key, "displayName", instance)
                          }
                          value={guardian.displayName}
                          onChange={(event) =>
                            setGuardians((prev) =>
                              prev.map((item) =>
                                item.key === guardian.key ? { ...item, displayName: event.target.value } : item
                              )
                            )
                          }
                          placeholder={`${guardianEntityLabel} ${index + 1} display name`}
                          aria-label={`${guardianEntityLabel} ${index + 1} display name`}
                          status={showGuardianInlineErrors && !guardian.displayName.trim() ? "error" : undefined}
                          data-guardrails-role="guardian"
                          data-member-key={guardian.key}
                          data-member-field="displayName"
                        />
                        {showGuardianInlineErrors && !guardian.displayName.trim() ? (
                          <Text type="danger">Display name is required.</Text>
                        ) : null}
                        <Text type="secondary">{`${guardianEntityLabel} ${index + 1} user ID`}</Text>
                        <Input
                          ref={(instance: InputRef | null) =>
                            setMemberFieldInputRef("guardian", guardian.key, "userId", instance)
                          }
                          value={guardian.userId}
                          onChange={(event) =>
                            setGuardians((prev) =>
                              prev.map((item) =>
                                item.key === guardian.key ? { ...item, userId: event.target.value } : item
                              )
                            )
                          }
                          placeholder={`${guardianEntityLabel} account user ID`}
                          aria-label={`${guardianEntityLabel} ${index + 1} user ID`}
                          status={
                            showGuardianInlineErrors &&
                            (!guardian.userId.trim() || guardianHasDuplicateUserId)
                              ? "error"
                              : undefined
                          }
                          data-guardrails-role="guardian"
                          data-member-key={guardian.key}
                          data-member-field="userId"
                        />
                        {showGuardianInlineErrors && !guardian.userId.trim() ? (
                          <Text type="danger">User ID is required.</Text>
                        ) : showGuardianInlineErrors && guardianHasDuplicateUserId ? (
                          <Text type="danger">User ID must be unique.</Text>
                        ) : null}
                        <Text type="secondary">{`${guardianEntityLabel} ${index + 1} email`}</Text>
                        <Input
                          value={guardian.email}
                          onChange={(event) =>
                            setGuardians((prev) =>
                              prev.map((item) =>
                                item.key === guardian.key ? { ...item, email: event.target.value } : item
                              )
                            )
                          }
                          placeholder={`${guardianEntityLabel} email (optional)`}
                          aria-label={`${guardianEntityLabel} ${index + 1} email`}
                        />
                        {guardians.length > 1 ? (
                          <Button
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() =>
                              setGuardians((prev) => prev.filter((item) => item.key !== guardian.key))
                            }
                          >
                            {`Remove ${guardianEntityLabel}`}
                          </Button>
                        ) : null}
                      </Space>
                    </Card>
                  )
                })}
                <Button
                  icon={<PlusOutlined />}
                  onClick={() => setGuardians((prev) => [...prev, newMember("guardian")])}
                >
                  {`Add ${guardianEntityLabel}`}
                </Button>
              </>
            )}
          </Space>
        )
      case 2:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type="info"
              title="Choose whether each dependent already has an account or needs a new invite."
            />
            <span style={{ display: "none" }}>
              Create or link dependent accounts here. User IDs are required for invitation and acceptance.
            </span>
            <Text type="secondary">
              Existing accounts need a sign-in user ID. Invite-new dependents can be provisioned by email or via a guardian copy link in the tracker.
            </Text>
            <span style={{ display: "none" }}>
              Use each dependent account user ID exactly as it appears at sign-in so invites can be accepted.
            </span>
            {mappingFixTargetLabel ? (
              <Alert
                showIcon
                type="warning"
                title={`Fixing mapping for ${mappingFixTargetLabel}.`}
                description="Update dependent account details as needed, then continue to regenerate relationship mapping."
              />
            ) : null}
            {showDependentInlineErrors && incompleteDependentCount > 0 ? (
              <Text type="secondary">
                {`Complete display name and any required user ID for ${incompleteDependentCount} ${
                  incompleteDependentCount === 1 ? "dependent" : "dependents"
                } to continue.`}
              </Text>
            ) : null}
            {showDependentInlineErrors &&
            (dependentDuplicateUserIds.size > 0 || dependentGuardianCollisionUserIds.size > 0) ? (
              <Text type="danger">
                {`Dependent user IDs must be unique and cannot match ${guardianEntityLabelLower} user IDs.`}
              </Text>
            ) : null}
            {showDependentInlineErrors && dependentDuplicateUserIdList.length > 0 ? (
              <Text type="secondary">
                {`Duplicate dependent user IDs: ${dependentDuplicateUserIdList.join(", ")}`}
              </Text>
            ) : null}
            {showDependentInlineErrors && dependentGuardianCollisionUserIdList.length > 0 ? (
              <Text type="secondary">
                {`Dependent user IDs already used by ${guardianEntityLabelPluralLower}: ${dependentGuardianCollisionUserIdList.join(", ")}`}
              </Text>
            ) : null}
            <Space>
              <Button
                type={dependentEntryMode === "card" ? "primary" : "default"}
                onClick={() => setDependentEntryMode("card")}
              >
                Card entry
              </Button>
              {dependents.length >= LARGE_HOUSEHOLD_TABLE_THRESHOLD || dependentEntryMode === "table" ? (
                <Button
                  type={dependentEntryMode === "table" ? "primary" : "default"}
                  onClick={() => setDependentEntryMode("table")}
                >
                  Table entry
                </Button>
              ) : null}
              <Button
                type={dependentEntryMode === "bulk" ? "primary" : "default"}
                onClick={() => setDependentEntryMode("bulk")}
              >
                Bulk entry
              </Button>
              <Button
                onClick={autofillMissingDependentUserIds}
                disabled={
                  dependents.every(
                    (dependent) =>
                      dependent.accountMode === "invite_new" || dependent.userId.trim().length > 0
                  )
                }
              >
                Auto-fill missing user IDs
              </Button>
            </Space>
            {dependentEntryMode === "bulk" ? (
              <Space orientation="vertical" style={{ width: "100%" }}>
                <Text type="secondary">
                  Bulk entry defaults every dependent to an existing account. Switch rows to invite-new after applying if needed.
                </Text>
                <Input.TextArea
                  value={dependentBulkInput}
                  onChange={(event) => setDependentBulkInput(event.target.value)}
                  placeholder={BULK_ENTRY_PLACEHOLDER}
                  rows={6}
                />
                <Button onClick={() => applyBulkMembers("dependent")}>Apply bulk entries</Button>
                {dependentBulkCount != null ? (
                  <Text type="secondary">{`${dependentBulkCount} entries ready`}</Text>
                ) : null}
              </Space>
            ) : dependentEntryMode === "table" ? (
              <Space orientation="vertical" style={{ width: "100%" }}>
                <Space wrap>
                  <Button onClick={selectAllDependents}>
                    Select all
                  </Button>
                  <Button onClick={clearDependentSelection}>
                    Clear selection
                  </Button>
                  <Button danger onClick={removeSelectedDependents}>
                    Remove selected
                  </Button>
                  {TEMPLATE_OPTIONS.map((option) => (
                    <Button
                      key={`table-template-${option.value}`}
                      onClick={() => applyTemplateToSelectedDependents(option.value)}
                    >
                      {`Apply "${option.label}" to selected`}
                    </Button>
                  ))}
                </Space>
                <Text type="secondary">{`Selected: ${selectedDependentKeys.length}`}</Text>
                <Text type="secondary">Shortcuts: Ctrl/Cmd+A select all, Delete removes selected.</Text>
                {dependentTableMessage ? (
                  <Text type="secondary">{dependentTableMessage}</Text>
                ) : null}
                <Table
                  rowKey="key"
                  size="small"
                  pagination={false}
                  dataSource={dependents}
                  rowSelection={{
                    selectedRowKeys: selectedDependentKeys,
                    onChange: (selectedRowKeys) => {
                      setSelectedDependentKeys(selectedRowKeys.map((key) => String(key)))
                      setDependentTableMessage(null)
                    }
                  }}
                  columns={[
                    {
                      title: "Account Setup",
                      dataIndex: "accountMode",
                      render: (_value: string, dependent: MemberInput) => (
                        <Radio.Group
                          value={dependent.accountMode}
                          onChange={(event) =>
                            updateDependentMember(dependent.key, {
                              accountMode: normalizeAccountMode(event.target.value)
                            })
                          }
                        >
                          <Radio.Button value="existing_account">Existing</Radio.Button>
                          <Radio.Button value="invite_new">Invite new</Radio.Button>
                        </Radio.Group>
                      )
                    },
                    {
                      title: "Display Name",
                      dataIndex: "displayName",
                      render: (_value: string, dependent: MemberInput, index: number) => (
                        <Input
                          ref={(instance: InputRef | null) =>
                            setMemberFieldInputRef("dependent", dependent.key, "displayName", instance)
                          }
                          value={dependent.displayName}
                          onChange={(event) =>
                            updateDependentMember(dependent.key, { displayName: event.target.value })
                          }
                          placeholder={`Child ${index + 1} display name`}
                          aria-label={`Dependent ${index + 1} display name`}
                          status={showDependentInlineErrors && !dependent.displayName.trim() ? "error" : undefined}
                          data-guardrails-role="dependent"
                          data-member-key={dependent.key}
                          data-member-field="displayName"
                        />
                      )
                    },
                    {
                      title: "User ID",
                      dataIndex: "userId",
                      render: (_value: string, dependent: MemberInput, index: number) => {
                        if (dependent.accountMode === "invite_new") {
                          return <Text type="secondary">Created after invite acceptance</Text>
                        }
                        const normalizedDependentUserId = normalizeMemberUserId(dependent.userId)
                        const dependentHasDuplicateUserId =
                          normalizedDependentUserId.length > 0 &&
                          dependentDuplicateUserIds.has(normalizedDependentUserId)
                        const dependentMatchesGuardianUserId =
                          normalizedDependentUserId.length > 0 &&
                          dependentGuardianCollisionUserIds.has(normalizedDependentUserId)
                        return (
                          <Input
                            ref={(instance: InputRef | null) =>
                              setMemberFieldInputRef("dependent", dependent.key, "userId", instance)
                            }
                            value={dependent.userId}
                            onChange={(event) =>
                              updateDependentMember(dependent.key, { userId: event.target.value })
                            }
                            placeholder="Child account user ID"
                            aria-label={`Dependent ${index + 1} user ID`}
                            status={
                              showDependentInlineErrors &&
                              (!dependent.userId.trim() ||
                                dependentHasDuplicateUserId ||
                                dependentMatchesGuardianUserId)
                                ? "error"
                                : undefined
                            }
                            data-guardrails-role="dependent"
                            data-member-key={dependent.key}
                            data-member-field="userId"
                          />
                        )
                      }
                    },
                    {
                      title: "Invite Email (optional)",
                      dataIndex: "email",
                      render: (_value: string, dependent: MemberInput, index: number) => (
                        <Input
                          value={dependent.email}
                          onChange={(event) =>
                            updateDependentMember(dependent.key, { email: event.target.value })
                          }
                          placeholder={
                            dependent.accountMode === "invite_new"
                              ? "Child email or leave blank for guardian copy"
                              : "Child email (optional)"
                          }
                          aria-label={`Dependent ${index + 1} email`}
                        />
                      )
                    },
                    {
                      title: "Actions",
                      key: "actions",
                      render: (_value: unknown, dependent: MemberInput) =>
                        dependents.length > 1 ? (
                          <Button
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() => removeDependentsByKeys([dependent.key])}
                          >
                            Remove
                          </Button>
                        ) : (
                          <Text type="secondary">Required</Text>
                        )
                    }
                  ]}
                />
                <Button
                  icon={<PlusOutlined />}
                  onClick={() => setDependents((prev) => [...prev, newMember("dependent")])}
                >
                  Add Dependent
                </Button>
              </Space>
            ) : (
              <>
                {dependents.map((dependent, index) => {
                  const normalizedDependentUserId = normalizeMemberUserId(dependent.userId)
                  const dependentHasDuplicateUserId =
                    normalizedDependentUserId.length > 0 &&
                    dependentDuplicateUserIds.has(normalizedDependentUserId)
                  const dependentMatchesGuardianUserId =
                    normalizedDependentUserId.length > 0 &&
                    dependentGuardianCollisionUserIds.has(normalizedDependentUserId)

                  return (
                    <Card key={dependent.key} size="small">
                      <Space orientation="vertical" style={{ width: "100%" }}>
                        <Text type="secondary">{`Dependent ${index + 1} account setup`}</Text>
                        <Radio.Group
                          value={dependent.accountMode}
                          onChange={(event) =>
                            updateDependentMember(dependent.key, {
                              accountMode: normalizeAccountMode(event.target.value)
                            })
                          }
                        >
                          <Radio.Button value="existing_account">Existing account</Radio.Button>
                          <Radio.Button value="invite_new">Invite new</Radio.Button>
                        </Radio.Group>
                        <Text type="secondary">{`Dependent ${index + 1} display name`}</Text>
                        <Input
                          ref={(instance: InputRef | null) =>
                            setMemberFieldInputRef("dependent", dependent.key, "displayName", instance)
                          }
                          value={dependent.displayName}
                          onChange={(event) =>
                            updateDependentMember(dependent.key, { displayName: event.target.value })
                          }
                          placeholder={`Child ${index + 1} display name`}
                          aria-label={`Dependent ${index + 1} display name`}
                          status={showDependentInlineErrors && !dependent.displayName.trim() ? "error" : undefined}
                          data-guardrails-role="dependent"
                          data-member-key={dependent.key}
                          data-member-field="displayName"
                        />
                        {showDependentInlineErrors && !dependent.displayName.trim() ? (
                          <Text type="danger">Display name is required.</Text>
                        ) : null}
                        {dependent.accountMode === "existing_account" ? (
                          <>
                            <Text type="secondary">{`Dependent ${index + 1} user ID`}</Text>
                            <Input
                              ref={(instance: InputRef | null) =>
                                setMemberFieldInputRef("dependent", dependent.key, "userId", instance)
                              }
                              value={dependent.userId}
                              onChange={(event) =>
                                updateDependentMember(dependent.key, { userId: event.target.value })
                              }
                              placeholder="Child account user ID"
                              aria-label={`Dependent ${index + 1} user ID`}
                              status={
                                showDependentInlineErrors &&
                                (!dependent.userId.trim() ||
                                  dependentHasDuplicateUserId ||
                                  dependentMatchesGuardianUserId)
                                  ? "error"
                                  : undefined
                              }
                              data-guardrails-role="dependent"
                              data-member-key={dependent.key}
                              data-member-field="userId"
                            />
                            {showDependentInlineErrors && !dependent.userId.trim() ? (
                              <Text type="danger">User ID is required for existing accounts.</Text>
                            ) : showDependentInlineErrors && dependentHasDuplicateUserId ? (
                              <Text type="danger">User ID must be unique.</Text>
                            ) : showDependentInlineErrors && dependentMatchesGuardianUserId ? (
                              <Text type="danger">{`User ID cannot match a ${guardianEntityLabelLower}.`}</Text>
                            ) : null}
                          </>
                        ) : (
                          <Text type="secondary">
                            A new dependent account will be created when this invite is accepted.
                          </Text>
                        )}
                        <Text type="secondary">
                          {dependent.accountMode === "invite_new"
                            ? `Dependent ${index + 1} invite email`
                            : `Dependent ${index + 1} email`}
                        </Text>
                        <Input
                          value={dependent.email}
                          onChange={(event) =>
                            updateDependentMember(dependent.key, { email: event.target.value })
                          }
                          placeholder={
                            dependent.accountMode === "invite_new"
                              ? "Child email or leave blank for guardian copy"
                              : "Child email (optional)"
                          }
                          aria-label={`Dependent ${index + 1} email`}
                        />
                        {dependents.length > 1 ? (
                          <Button
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() =>
                              setDependents((prev) => prev.filter((item) => item.key !== dependent.key))
                            }
                          >
                            Remove Dependent
                          </Button>
                        ) : null}
                      </Space>
                    </Card>
                  )
                })}
                <Button
                  icon={<PlusOutlined />}
                  onClick={() => setDependents((prev) => [...prev, newMember("dependent")])}
                >
                  Add Dependent
                </Button>
              </>
            )}
          </Space>
        )
      case 3:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Paragraph type="secondary">
              {`Map each dependent to a ${guardianEntityLabelLower}. For shared households, dependents can be mapped to different ${guardianEntityLabelPluralLower}.`}
            </Paragraph>
            {mappingFixTargetLabel ? (
              <Alert
                showIcon
                type="warning"
                title={`Fixing mapping for ${mappingFixTargetLabel}.`}
                description="Remap the dependent guardian assignment below, then continue to refresh activation readiness."
              />
            ) : null}
            {dependents.map((dependent) => (
              <Card key={dependent.key} size="small">
                <Space orientation="vertical" style={{ width: "100%" }}>
                  <Text strong>{dependent.displayName || dependent.userId || dependent.key}</Text>
                  <Select
                    value={dependentGuardianKey[dependent.key] || guardians[0]?.key}
                    options={guardianOptions}
                    onChange={(value) =>
                      setDependentGuardianKey((prev) => ({
                        ...prev,
                        [dependent.key]: value
                      }))
                    }
                  />
                </Space>
              </Card>
            ))}
          </Space>
        )
      case 4:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type="info"
              title="Apply a template first, then customize if needed."
            />
            {templateReviewTargetLabel ? (
              <Alert
                showIcon
                type="warning"
                title={`Reviewing template for ${templateReviewTargetLabel}.`}
                description="Adjust template or advanced overrides, then continue to re-check activation status."
              />
            ) : null}
            {guardians.length <= 1 ? (
              <Alert
                showIcon
                type="info"
                title={`Relationship mapping was auto-applied to ${primaryGuardianName} for all dependents.`}
              />
            ) : null}
            <Space wrap>
              {TEMPLATE_OPTIONS.map((option) => (
                <Button key={option.value} onClick={() => applyTemplateToAll(option.value)}>
                  Apply "{option.label}" to all
                </Button>
              ))}
              <Button onClick={() => setShowAdvancedOverrides((value) => !value)}>
                {showAdvancedOverrides ? "Hide Advanced Overrides" : "Show Advanced Overrides"}
              </Button>
            </Space>
            {dependents.map((dependent) => {
              const template = templateByDependentKey[dependent.key] || "default-child-safe"
              const override = overridesByDependentKey[dependent.key] || {
                action: "block",
                notify_context: alertNotifyContext
              }
              return (
                <Card key={dependent.key} size="small">
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    <Text strong>{dependent.displayName || dependent.userId || dependent.key}</Text>
                    <Select
                      value={template}
                      options={TEMPLATE_OPTIONS.map((option) => ({
                        label: `${option.label} - ${option.description}`,
                        value: option.value
                      }))}
                      onChange={(value) =>
                        setTemplateByDependentKey((prev) => ({
                          ...prev,
                          [dependent.key]: value as TemplateId
                        }))
                      }
                    />
                    {showAdvancedOverrides ? (
                      <Space>
                        <Select
                          value={override.action}
                          options={[
                            { label: "Block", value: "block" },
                            { label: "Redact", value: "redact" },
                            { label: "Warn", value: "warn" },
                            { label: "Notify", value: "notify" }
                          ]}
                          onChange={(value) =>
                            setOverridesByDependentKey((prev) => ({
                              ...prev,
                              [dependent.key]: {
                                ...override,
                                action: value as OverrideInput["action"]
                              }
                            }))
                          }
                        />
                        <Select
                          value={override.notify_context}
                          options={[
                            { label: "Topic only", value: "topic_only" },
                            { label: "Snippet", value: "snippet" },
                            { label: "Full message", value: "full_message" }
                          ]}
                          onChange={(value) =>
                            setOverridesByDependentKey((prev) => ({
                              ...prev,
                              [dependent.key]: {
                                ...override,
                                notify_context: value as OverrideInput["notify_context"]
                              }
                            }))
                          }
                        />
                      </Space>
                    ) : null}
                  </Space>
                </Card>
              )
            })}
          </Space>
        )
      case 5:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Paragraph type="secondary">
              {`Choose how ${guardianEntityLabelPluralLower} receive moderation context when alerts trigger.`}
            </Paragraph>
            <Select
              value={alertNotifyContext}
              options={[
                { label: "Topic only", value: "topic_only" },
                { label: "Snippet", value: "snippet" },
                { label: "Full message", value: "full_message" }
              ]}
              onChange={(value) =>
                setAlertNotifyContext(value as "topic_only" | "snippet" | "full_message")
              }
              style={{ maxWidth: 320 }}
            />
            <Text type="secondary">
              You can fine-tune these settings per dependent using advanced overrides in the templates step.
            </Text>
          </Space>
        )
      case 6:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type={trackerGuidance.type}
              title={trackerGuidance.message}
              description={trackerGuidance.description}
            />
            <Space>
              <Button icon={<ReloadOutlined />} onClick={() => void refreshInviteTracker()}>
                Refresh statuses
              </Button>
              <Button
                loading={resendingInvites}
                disabled={!draft?.id || !pendingInviteMemberDraftIds.length}
                onClick={() => void handleResendPendingInvites()}
              >
                Resend Pending Invites
              </Button>
            </Space>
            <Table
              rowKey={(row) => row.member_draft_id}
              dataSource={trackerRows}
              pagination={false}
              columns={[
                {
                  title: "Dependent",
                  render: (_value: unknown, row: HouseholdInviteTrackerItem) => (
                    <Space orientation="vertical" size={0}>
                      <Text strong>{row.display_name}</Text>
                      <Text type="secondary">
                        {row.account_mode === "invite_new"
                          ? "Invite new dependent"
                          : row.dependent_user_id || "Existing account"}
                      </Text>
                    </Space>
                  )
                },
                {
                  title: "Invite",
                  render: (_value: unknown, row: HouseholdInviteTrackerItem) => (
                    <Space orientation="vertical" size={0}>
                      <Tag color={STATUS_COLOR[row.invite_status] || "default"}>{row.invite_status}</Tag>
                      <Text type="secondary">
                        {row.invite_delivery_target ||
                          (row.invite_delivery_channel === "guardian_copy"
                            ? "Guardian copy"
                            : row.invite_delivery_channel || "Not provisioned")}
                      </Text>
                      {row.invite_last_sent_at ? (
                        <Text type="secondary">{`Last sent: ${row.invite_last_sent_at}`}</Text>
                      ) : null}
                      {row.invite_accepted_at ? (
                        <Text type="secondary">{`Accepted: ${row.invite_accepted_at}`}</Text>
                      ) : null}
                      {row.invite_expires_at ? (
                        <Text type="secondary">{`Expires: ${row.invite_expires_at}`}</Text>
                      ) : null}
                    </Space>
                  )
                },
                {
                  title: "Relationship",
                  render: (_value: unknown, row: HouseholdInviteTrackerItem) =>
                    row.relationship_status ? (
                      <Tag color={STATUS_COLOR[row.relationship_status] || "default"}>
                        {row.relationship_status}
                      </Tag>
                    ) : (
                      <Text type="secondary">Not mapped</Text>
                    )
                },
                {
                  title: "Guardrail Activation",
                  render: (_value: unknown, row: HouseholdInviteTrackerItem) =>
                    row.plan_status ? (
                      <Tag color={STATUS_COLOR[row.plan_status] || "default"}>
                        {row.plan_status === "queued"
                          ? "Queued until acceptance"
                          : row.plan_status === "active"
                            ? "Active"
                            : "Failed"}
                      </Tag>
                    ) : (
                      <Text type="secondary">Not queued</Text>
                    )
                },
                {
                  title: "Blockers",
                  render: (_value: unknown, row: HouseholdInviteTrackerItem) =>
                    row.blocker_codes.length ? (
                      <Space wrap>
                        {row.blocker_codes.map((code) => (
                          <Tag key={code} color="orange">
                            {getTrackerBlockerLabel(code)}
                          </Tag>
                        ))}
                      </Space>
                    ) : (
                      <Text type="secondary">None</Text>
                    )
                },
                {
                  title: "Actions",
                  key: "actions",
                  render: (_value: unknown, row: HouseholdInviteTrackerItem) => {
                    const actions = trackerRowActions[row.member_draft_id] ?? []
                    if (!actions.length) return <Text type="secondary">None</Text>
                    return (
                      <Space wrap>
                        {actions.map((action) => (
                          <Button
                            key={`${row.member_draft_id}-${action}`}
                            size="small"
                            loading={
                              resendingInvites &&
                              (action === "provision_invite" ||
                                action === "resend_invite" ||
                                action === "reissue_invite")
                            }
                            aria-label={`${getTrackerActionLabel(action)} for ${row.display_name}`}
                            onClick={() => void handleTrackerRowAction(row, action)}
                          >
                            {getTrackerActionLabel(action)}
                          </Button>
                        ))}
                      </Space>
                    )
                  }
                }
              ]}
            />
          </Space>
        )
      case 7:
      default:
        return (
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            <Alert
              showIcon
              type={reviewGuidance.type}
              title={reviewGuidance.message}
              description={reviewGuidance.description}
            />
            <Card size="small">
              <Space orientation="vertical" style={{ width: "100%" }}>
                <Text>
                  <Text strong>Household:</Text> {householdName}
                </Text>
                <Text>
                  <Text strong>Mode:</Text> {mode}
                </Text>
                <Text>
                  <Text strong>{`${guardianEntityLabelLower === "caregiver" ? "Caregivers" : "Guardians"}:`}</Text>{" "}
                  {guardians.length}
                </Text>
                <Text>
                  <Text strong>Dependents:</Text> {dependents.length}
                </Text>
                <Text>
                  <Text strong>Activation:</Text>{" "}
                  {`${trackerCounts.active} active, ${trackerCounts.pending} pending, ${trackerCounts.failed} failed`}
                </Text>
              </Space>
            </Card>
          </Space>
        )
    }
  }

  const finalStepCtaLabel =
    trackerCounts.failed > 0
      ? "Finish Setup (Needs Attention)"
      : trackerCounts.pending > 0
        ? "Finish Setup (Invites Pending)"
        : "Finish Setup"

  return (
    <Space
      orientation="vertical"
      size="large"
      style={{ width: "100%", minHeight: "100%" }}
      data-testid="wizard-shell"
    >
      <div>
        <Title level={4}>Family Guardrails Wizard</Title>
        <Paragraph type="secondary">
          {`Template-first setup for ${guardianEntityLabelPluralLower}, dependents, moderation templates, and acceptance tracking.`}
        </Paragraph>
      </div>

      <Card size="small">
        <Space orientation="vertical" size={2} style={{ width: "100%" }}>
          <Text type="secondary">{`Step ${currentStep + 1} of ${stepDefinitions.length}`}</Text>
          <Title level={5} style={{ margin: 0 }}>
            {currentStepDefinition.title}
          </Title>
          <Text type="secondary">{currentStepDefinition.cue}</Text>
          <Text type="secondary">
            {nextStepDefinition ? `Next: ${nextStepDefinition.shortTitle}` : "Next: Finish Setup"}
          </Text>
        </Space>
      </Card>

      <div style={{ overflowX: "auto", paddingBottom: 4 }}>
        <Steps
          current={currentStep}
          size="small"
          items={stepDefinitions.map((step) => ({ title: step.shortTitle }))}
        />
      </div>

      <Card>{renderStepContent()}</Card>

      <Divider style={{ margin: "0" }} />

      <div
        data-testid="wizard-action-footer"
        style={{
          marginTop: "auto",
          position: "sticky",
          bottom: 0,
          zIndex: 20,
          paddingTop: 8,
          background:
            "linear-gradient(180deg, rgba(0,0,0,0) 0%, var(--ant-color-bg-layout, #ffffff) 42%)"
        }}
      >
        <Card size="small">
          <div
            data-testid="wizard-action-controls"
            style={{
              width: "100%",
              display: "flex",
              flexWrap: "wrap",
              alignItems: "center",
              gap: 8
            }}
          >
            <Button
              disabled={currentStep === 0 || submitting}
              onClick={handleBack}
              style={{ minWidth: 96 }}
            >
              Back
            </Button>
            <Button
              type="primary"
              disabled={submitting}
              loading={submitting}
              onClick={() => void handleNext()}
              style={{ marginInlineStart: "auto", minWidth: 180, flex: "1 1 220px" }}
            >
              {currentStep === stepDefinitions.length - 1 ? finalStepCtaLabel : "Save & Continue"}
            </Button>
          </div>
        </Card>
      </div>
    </Space>
  )
}

export default FamilyGuardrailsWizard
