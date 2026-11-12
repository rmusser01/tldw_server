import React from "react"
import { smoothScrollBehavior } from "@/utils/reduced-motion"

import { tldwClient } from "@/services/tldw/TldwApiClient"

/**
 * Types re-exported for consumer convenience.
 */

export type PersonaGovernanceScopeContext = {
  server_id?: string | null
  server_name?: string | null
  requested_slots?: string[]
  bound_slots?: string[]
  missing_bound_slots?: string[]
  missing_secret_slots?: string[]
  workspace_id?: string | null
  workspace_bundle_ids?: string[]
  workspace_bundle_roots?: string[]
  normalized_paths?: string[]
  selected_workspace_trust_source?: string | null
  selected_assignment_id?: number | null
  blocked_reason?: string | null
  reason?: string | null
}

export type PersonaRuntimeApprovalDuration = "once" | "session" | "conversation"

export type PersonaRuntimeApprovalPayload = {
  approval_policy_id?: number | null
  mode?: string | null
  tool_name?: string | null
  context_key?: string | null
  conversation_id?: string | null
  scope_key?: string | null
  reason?: string | null
  duration_options?: string[]
  arguments_summary?: Record<string, unknown>
  scope_context?: PersonaGovernanceScopeContext | null
}

export type PersonaRuntimeApprovalRequest = {
  key: string
  approval_policy_id?: number | null
  mode?: string | null
  tool_name: string
  context_key: string
  conversation_id?: string | null
  scope_key: string
  reason?: string | null
  duration_options: PersonaRuntimeApprovalDuration[]
  arguments_summary: Record<string, unknown>
  scope_context?: PersonaGovernanceScopeContext | null
  selected_duration: PersonaRuntimeApprovalDuration
  session_id?: string | null
  plan_id?: string | null
  step_idx?: number
  step_type?: string | null
  tool?: string | null
  args?: Record<string, unknown>
  why?: string | null
  description?: string | null
}

export type ApprovalHighlightPhase =
  | "none"
  | "landing_primary"
  | "landing_secondary"
  | "steady"

// ── Internal helpers ──

const RESOLVED_RUNTIME_APPROVAL_FADE_MS = 1500
const APPROVAL_HIGHLIGHT_PRIMARY_MS = 900
const APPROVAL_HIGHLIGHT_SECONDARY_MS = 650

const _normalizeStringList = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => String(entry || "").trim())
    .filter((entry) => entry.length > 0)
}

export const coerceGovernanceContext = (
  value: unknown
): PersonaGovernanceScopeContext | null => {
  if (!value || typeof value !== "object") return null
  const raw = value as Record<string, unknown>
  const context: PersonaGovernanceScopeContext = {
    server_id: raw.server_id ? String(raw.server_id) : null,
    server_name: raw.server_name ? String(raw.server_name) : null,
    requested_slots: _normalizeStringList(raw.requested_slots),
    bound_slots: _normalizeStringList(raw.bound_slots),
    missing_bound_slots: _normalizeStringList(raw.missing_bound_slots),
    missing_secret_slots: _normalizeStringList(raw.missing_secret_slots),
    workspace_id: raw.workspace_id ? String(raw.workspace_id) : null,
    workspace_bundle_ids: _normalizeStringList(raw.workspace_bundle_ids),
    workspace_bundle_roots: _normalizeStringList(raw.workspace_bundle_roots),
    normalized_paths: _normalizeStringList(raw.normalized_paths),
    selected_workspace_trust_source: raw.selected_workspace_trust_source
      ? String(raw.selected_workspace_trust_source)
      : null,
    selected_assignment_id:
      typeof raw.selected_assignment_id === "number"
        ? raw.selected_assignment_id
        : Number.isFinite(Number(raw.selected_assignment_id))
          ? Number(raw.selected_assignment_id)
          : null,
    blocked_reason: raw.blocked_reason ? String(raw.blocked_reason) : null,
    reason: raw.reason ? String(raw.reason) : null,
  }
  const hasContent =
    Boolean(
      context.server_id ||
        context.server_name ||
        context.workspace_id ||
        context.selected_workspace_trust_source ||
        context.blocked_reason ||
        context.reason
    ) ||
    Boolean(context.workspace_bundle_ids?.length) ||
    Boolean(context.workspace_bundle_roots?.length) ||
    Boolean(context.normalized_paths?.length) ||
    Boolean(context.requested_slots?.length) ||
    Boolean(context.bound_slots?.length) ||
    Boolean(context.missing_bound_slots?.length) ||
    Boolean(context.missing_secret_slots?.length)
  return hasContent ? context : null
}

export const formatGovernanceDenyMessage = (
  context: PersonaGovernanceScopeContext | null,
  reasonCode: string | null
): string | null => {
  const normalizedReason = String(
    reasonCode || context?.blocked_reason || ""
  )
    .trim()
    .toLowerCase()
  if (normalizedReason === "required_slot_not_granted") {
    const slots = context?.missing_bound_slots?.length
      ? context.missing_bound_slots
      : context?.requested_slots || []
    return slots.length
      ? `Credential slots not granted: ${slots.join(", ")}`
      : null
  }
  if (normalizedReason === "required_slot_secret_missing") {
    const slots = context?.missing_secret_slots?.length
      ? context.missing_secret_slots
      : context?.requested_slots || []
    return slots.length
      ? `Credential secrets missing: ${slots.join(", ")}`
      : null
  }
  if (normalizedReason === "workspace_unresolvable_for_trust_source") {
    return "Blocked: workspace is not resolvable through the required trust source."
  }
  if (normalizedReason === "path_matches_multiple_workspace_roots") {
    return "Blocked: path matched multiple trusted workspace roots."
  }
  if (normalizedReason === "path_outside_workspace_bundle") {
    return "Blocked: path falls outside the allowed workspace bundle."
  }
  return null
}

export const approvalRequestKey = (
  approval: PersonaRuntimeApprovalPayload,
  payload: Record<string, unknown>
): string =>
  [
    String(approval.conversation_id || payload.session_id || "").trim(),
    String(approval.scope_key || "").trim(),
    String(approval.tool_name || payload.tool || "").trim(),
    String(payload.plan_id || "").trim(),
    String(payload.step_idx ?? "").trim(),
  ].join("|")

const _approvalDecisionPayload = (
  decision: "approved" | "denied",
  duration: string
): { duration: PersonaRuntimeApprovalDuration } => {
  const normalized = String(duration || "").trim().toLowerCase()
  if (decision !== "approved") {
    return { duration: "once" }
  }
  if (normalized === "conversation") {
    return { duration: "conversation" }
  }
  if (normalized === "session") {
    return { duration: "session" }
  }
  return { duration: "once" }
}

// ── Deps interface ──

export interface UsePersonaGovernanceContextDeps {
  connected: boolean
  sessionId: string | null
  wsRef: React.MutableRefObject<WebSocket | null>
  appendLog: (kind: "user" | "assistant" | "tool" | "notice", text: string) => void
  setError: React.Dispatch<React.SetStateAction<string | null>>
}

// ── Hook ──

export function usePersonaGovernanceContext(
  deps: UsePersonaGovernanceContextDeps
) {
  const { connected, sessionId, wsRef, appendLog, setError } = deps

  // ── Refs ──
  const runtimeApprovalCardRef = React.useRef<HTMLDivElement | null>(null)
  const runtimeApprovalRowRefs = React.useRef<
    Map<string, HTMLDivElement | null>
  >(new Map())
  const resolvedApprovalFadeTimerRef = React.useRef<number | null>(null)
  const approvalHighlightPhaseTimerRef = React.useRef<number | null>(null)

  // ── State ──
  const [pendingApprovals, setPendingApprovals] = React.useState<
    PersonaRuntimeApprovalRequest[]
  >([])
  const [activeApprovalKey, setActiveApprovalKey] = React.useState<
    string | null
  >(null)
  const [approvalHighlightPhase, setApprovalHighlightPhase] =
    React.useState<ApprovalHighlightPhase>("none")
  const [approvalHighlightSequence, setApprovalHighlightSequence] =
    React.useState(0)
  const [resolvedApprovalSnapshot, setResolvedApprovalSnapshot] =
    React.useState<{ key: string; toolName: string } | null>(null)
  const [approvedStepMap, setApprovedStepMap] = React.useState<
    Record<number, boolean>
  >({})
  const [submittingApprovalKey, setSubmittingApprovalKey] = React.useState<
    string | null
  >(null)

  // ── Timer helpers ──
  const clearResolvedApprovalFadeTimer = React.useCallback(() => {
    if (resolvedApprovalFadeTimerRef.current == null) return
    window.clearTimeout(resolvedApprovalFadeTimerRef.current)
    resolvedApprovalFadeTimerRef.current = null
  }, [])

  const clearApprovalHighlightPhaseTimer = React.useCallback(() => {
    if (approvalHighlightPhaseTimerRef.current == null) return
    window.clearTimeout(approvalHighlightPhaseTimerRef.current)
    approvalHighlightPhaseTimerRef.current = null
  }, [])

  const resetApprovalHighlightMotion = React.useCallback(() => {
    clearApprovalHighlightPhaseTimer()
    setApprovalHighlightPhase("none")
  }, [clearApprovalHighlightPhaseTimer])

  const triggerApprovalHighlightPhase = React.useCallback(
    (
      phase: Extract<
        ApprovalHighlightPhase,
        "landing_primary" | "landing_secondary"
      >
    ) => {
      const durationMs =
        phase === "landing_primary"
          ? APPROVAL_HIGHLIGHT_PRIMARY_MS
          : APPROVAL_HIGHLIGHT_SECONDARY_MS
      clearApprovalHighlightPhaseTimer()
      setApprovalHighlightPhase(phase)
      setApprovalHighlightSequence((prev) => prev + 1)
      approvalHighlightPhaseTimerRef.current = window.setTimeout(() => {
        approvalHighlightPhaseTimerRef.current = null
        setApprovalHighlightPhase("steady")
      }, durationMs)
    },
    [clearApprovalHighlightPhaseTimer]
  )

  // ── Effects: auto-select next approval & fade resolved snapshot ──
  React.useEffect(() => {
    if (!activeApprovalKey) return
    if (
      pendingApprovals.some(
        (approval) => approval.key === activeApprovalKey
      )
    )
      return
    const nextApprovalKey = pendingApprovals.length
      ? pendingApprovals[0]?.key || null
      : null
    setActiveApprovalKey(nextApprovalKey)
    if (nextApprovalKey) {
      triggerApprovalHighlightPhase("landing_secondary")
      return
    }
    resetApprovalHighlightMotion()
  }, [
    activeApprovalKey,
    pendingApprovals,
    resetApprovalHighlightMotion,
    triggerApprovalHighlightPhase,
  ])

  React.useEffect(() => {
    if (!resolvedApprovalSnapshot) {
      clearResolvedApprovalFadeTimer()
      return
    }
    if (
      pendingApprovals.length > 0 ||
      pendingApprovals.some(
        (approval) => approval.key === resolvedApprovalSnapshot.key
      )
    ) {
      clearResolvedApprovalFadeTimer()
      setResolvedApprovalSnapshot(null)
      return
    }
    clearResolvedApprovalFadeTimer()
    resolvedApprovalFadeTimerRef.current = window.setTimeout(() => {
      resolvedApprovalFadeTimerRef.current = null
      setResolvedApprovalSnapshot(null)
    }, RESOLVED_RUNTIME_APPROVAL_FADE_MS)
    return () => {
      clearResolvedApprovalFadeTimer()
    }
  }, [
    clearResolvedApprovalFadeTimer,
    pendingApprovals,
    resolvedApprovalSnapshot,
  ])

  React.useEffect(() => {
    return () => {
      clearResolvedApprovalFadeTimer()
    }
  }, [clearResolvedApprovalFadeTimer])

  React.useEffect(() => {
    return () => {
      clearApprovalHighlightPhaseTimer()
    }
  }, [clearApprovalHighlightPhaseTimer])

  // ── Callbacks ──

  const updateApprovalDuration = React.useCallback(
    (approvalKey: string, duration: PersonaRuntimeApprovalDuration) => {
      setPendingApprovals((prev) =>
        prev.map((approval) =>
          approval.key === approvalKey
            ? { ...approval, selected_duration: duration }
            : approval
        )
      )
    },
    []
  )

  const submitApprovalDecision = React.useCallback(
    async (
      approval: PersonaRuntimeApprovalRequest,
      decision: "approved" | "denied"
    ) => {
      const approvalDecision = _approvalDecisionPayload(
        decision,
        approval.selected_duration
      )
      setSubmittingApprovalKey(approval.key)
      setError(null)
      try {
        const response = await tldwClient.fetchWithAuth(
          "/api/v1/mcp/hub/approval-decisions" as any,
          {
            method: "POST",
            body: {
              approval_policy_id: approval.approval_policy_id,
              context_key: approval.context_key,
              conversation_id: approval.conversation_id,
              tool_name: approval.tool_name,
              scope_key: approval.scope_key,
              decision,
              duration: approvalDecision.duration,
            },
          }
        )
        if (!response.ok) {
          throw new Error(
            response.error || "Failed to submit approval decision"
          )
        }
        await response.json()
        if (approval.key === activeApprovalKey) {
          clearResolvedApprovalFadeTimer()
          setResolvedApprovalSnapshot({
            key: approval.key,
            toolName: approval.tool_name,
          })
        }
        setPendingApprovals((prev) =>
          prev.filter((entry) => entry.key !== approval.key)
        )
        appendLog(
          "notice",
          decision === "approved"
            ? `Approved ${approval.tool_name} and retrying`
            : `Denied ${approval.tool_name}`
        )
        if (
          decision === "approved" &&
          connected &&
          wsRef.current &&
          approval.step_type &&
          approval.tool
        ) {
          wsRef.current.send(
            JSON.stringify({
              type: "retry_tool_call",
              session_id: approval.session_id || sessionId,
              plan_id: approval.plan_id,
              step_idx: approval.step_idx,
              step_type: approval.step_type,
              tool: approval.tool,
              args: approval.args || {},
              why: approval.why,
              description: approval.description,
            })
          )
        }
      } catch (err: any) {
        setError(
          String(err?.message || "Failed to submit approval decision")
        )
      } finally {
        setSubmittingApprovalKey(null)
      }
    },
    [
      activeApprovalKey,
      appendLog,
      clearResolvedApprovalFadeTimer,
      connected,
      sessionId,
      setError,
      wsRef,
    ]
  )

  const activePendingApproval = React.useMemo(() => {
    if (!activeApprovalKey) return null
    return (
      pendingApprovals.find(
        (approval) => approval.key === activeApprovalKey
      ) || null
    )
  }, [activeApprovalKey, pendingApprovals])

  const pendingApprovalSummary = React.useMemo(() => {
    if (!pendingApprovals.length) return null
    const primaryApproval =
      activePendingApproval || pendingApprovals[0] || null
    if (!primaryApproval) return null
    const primaryToolName =
      String(primaryApproval.tool_name || "tool").trim() || "tool"
    const additionalCount = pendingApprovals.filter(
      (approval) => approval.key !== primaryApproval.key
    ).length
    if (additionalCount <= 0) {
      return `Waiting for approval: ${primaryToolName}`
    }
    return `Waiting for approval: ${primaryToolName} (+${additionalCount} more)`
  }, [activePendingApproval, pendingApprovals])

  const registerRuntimeApprovalRow = React.useCallback(
    (approvalKey: string, node: HTMLDivElement | null) => {
      if (node) {
        runtimeApprovalRowRefs.current.set(approvalKey, node)
        return
      }
      runtimeApprovalRowRefs.current.delete(approvalKey)
    },
    []
  )

  const handleJumpToRuntimeApproval = React.useCallback(() => {
    if (!pendingApprovals.length) return
    const targetApprovalKey =
      activeApprovalKey || pendingApprovals[0]?.key || null
    if (!targetApprovalKey) return
    setActiveApprovalKey(targetApprovalKey)
    triggerApprovalHighlightPhase("landing_primary")
    const card = runtimeApprovalCardRef.current
    const targetRow =
      runtimeApprovalRowRefs.current.get(targetApprovalKey) || null
    const scrollTarget = targetRow || card
    if (!scrollTarget) return
    try {
      scrollTarget.scrollIntoView?.({
        block: "start",
        behavior: smoothScrollBehavior(),
      })
    } catch {
      // Ignore environments without scrollIntoView support.
    }
    const focusRoot = targetRow || card
    const buttons = Array.from(
      focusRoot.querySelectorAll("button")
    ) as HTMLButtonElement[]
    const preferredButton =
      buttons.find((button) =>
        String(button.textContent || "")
          .toLowerCase()
          .includes("approve")
      ) || buttons.find((button) => !button.disabled)
    preferredButton?.focus()
  }, [activeApprovalKey, pendingApprovals, triggerApprovalHighlightPhase])

  return {
    // state
    pendingApprovals,
    setPendingApprovals,
    activeApprovalKey,
    setActiveApprovalKey,
    approvalHighlightPhase,
    approvalHighlightSequence,
    resolvedApprovalSnapshot,
    setResolvedApprovalSnapshot,
    approvedStepMap,
    setApprovedStepMap,
    submittingApprovalKey,
    // refs
    runtimeApprovalCardRef,
    runtimeApprovalRowRefs,
    // computed
    activePendingApproval,
    pendingApprovalSummary,
    // callbacks
    clearResolvedApprovalFadeTimer,
    resetApprovalHighlightMotion,
    triggerApprovalHighlightPhase,
    updateApprovalDuration,
    submitApprovalDecision,
    registerRuntimeApprovalRow,
    handleJumpToRuntimeApproval,
  }
}
