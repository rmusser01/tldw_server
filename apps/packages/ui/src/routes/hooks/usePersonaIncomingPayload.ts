import React from "react"
import type { PersonaSetupStep } from "@/hooks/usePersonaSetupWizard"
import type { SetupTestOutcome } from "@/components/PersonaGarden/SetupTestAndFinishStep"
import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import {
  asPersonaVisualStateId,
  isPersonaVisualStateIdText
} from "@/types/persona-visuals"
import {
  coerceGovernanceContext as _coerceGovernanceContext,
  formatGovernanceDenyMessage as _formatGovernanceDenyMessage,
  approvalRequestKey as _approvalRequestKey,
  type PersonaRuntimeApprovalRequest,
  type PersonaRuntimeApprovalPayload,
  type PersonaRuntimeApprovalDuration,
} from "./usePersonaGovernanceContext"
import type {
  PersonaPlanStep,
  PersonaMemoryUsage,
  PersonaCompanionUsage,
  PersonaToolPolicy,
  PersonaLogEntry,
  SetupHandoffConsumedAction,
  SetupLiveDetourState,
  SetupHandoffState,
  PendingPlan
} from "../personaTypes"

export type UsePersonaIncomingPayloadArgs = {
  appendLog: (kind: PersonaLogEntry["kind"], text: string) => void
  clearResolvedApprovalFadeTimer: () => void
  consumeSetupHandoffAction: (action: SetupHandoffConsumedAction) => void
  emitSetupAnalyticsEvent: (event: Record<string, unknown>) => void
  liveVoiceController: { handlePayload: (payload: Record<string, unknown> | null) => void }
  personaId?: string | null
  personaSetupWizardCurrentStep: PersonaSetupStep
  personaSetupWizardIsSetupRequired: boolean
  resolvedApprovalSnapshot: { key: string; toolName: string } | null
  sessionId: string | null
  setApprovedStepMap: React.Dispatch<React.SetStateAction<Record<number, boolean>>>
  setPendingApprovals: React.Dispatch<React.SetStateAction<PersonaRuntimeApprovalRequest[]>>
  setPendingPlan: React.Dispatch<React.SetStateAction<PendingPlan | null>>
  setResolvedApprovalSnapshot: React.Dispatch<React.SetStateAction<{ key: string; toolName: string } | null>>
  setSetupTestOutcome: React.Dispatch<React.SetStateAction<SetupTestOutcome | null>>
  setSetupLiveDetour: React.Dispatch<React.SetStateAction<SetupLiveDetourState | null>>
  setSetupTestResumeNote: React.Dispatch<React.SetStateAction<string | null>>
  setupLiveDetourRef: React.MutableRefObject<SetupLiveDetourState | null>
  setupHandoffRef: React.MutableRefObject<SetupHandoffState | null>
  activeTabRef: React.MutableRefObject<string>
  setupWizardAwaitingLiveResponseRef: React.MutableRefObject<boolean>
  setupWizardLastLiveTextRef: React.MutableRefObject<string>
}

export const usePersonaIncomingPayload = ({
  appendLog,
  clearResolvedApprovalFadeTimer,
  consumeSetupHandoffAction,
  emitSetupAnalyticsEvent,
  liveVoiceController,
  personaId,
  personaSetupWizardCurrentStep,
  personaSetupWizardIsSetupRequired,
  resolvedApprovalSnapshot,
  sessionId,
  setApprovedStepMap,
  setPendingApprovals,
  setPendingPlan,
  setResolvedApprovalSnapshot,
  setSetupTestOutcome,
  setSetupLiveDetour,
  setSetupTestResumeNote,
  setupLiveDetourRef,
  setupHandoffRef,
  activeTabRef,
  setupWizardAwaitingLiveResponseRef,
  setupWizardLastLiveTextRef,
}: UsePersonaIncomingPayloadArgs) => {
  const setVisualRuntimeOverride = usePersonaVisualRuntimeStore(
    (state) => state.setOverride
  )
  const handleIncomingPayload = React.useCallback(
    (payload: any) => {
      const eventType = String(payload?.event || payload?.type || "").toLowerCase()
      if (!eventType) return
      liveVoiceController.handlePayload(
        payload && typeof payload === "object"
          ? (payload as Record<string, unknown>)
          : null
      )

      if (eventType === "visual_state_override") {
        const state = String(payload?.state || payload?.visual_state || "").trim()
        if (!isPersonaVisualStateIdText(state)) return
        const durationMsRaw =
          typeof payload?.duration_ms === "number"
            ? payload.duration_ms
            : Number.parseInt(String(payload?.duration_ms ?? "1500"), 10)
        const durationMs = Number.isFinite(durationMsRaw)
          ? Math.max(250, Math.min(durationMsRaw, 30_000))
          : 1500
        const reason = String(payload?.reason || "")
          .split(/\s+/)
          .filter(Boolean)
          .join(" ")
          .slice(0, 200)
        setVisualRuntimeOverride({
          personaId: String(payload?.persona_id || personaId || ""),
          sessionId: payload?.session_id
            ? String(payload.session_id)
            : sessionId
              ? String(sessionId)
              : null,
          state: asPersonaVisualStateId(state),
          reason: reason || null,
          expiresAt: Date.now() + durationMs
        })
        return
      }

      if (eventType === "tool_plan") {
        const planId = String(payload?.plan_id || "")
        const stepsRaw = Array.isArray(payload?.steps) ? payload.steps : []
        const steps: PersonaPlanStep[] = stepsRaw
          .map((step: any, idx: number) => ({
            idx:
              typeof step?.idx === "number"
                ? step.idx
                : Number.parseInt(String(step?.idx ?? idx), 10),
            tool: String(step?.tool || "unknown_tool"),
            args:
              step?.args && typeof step.args === "object"
                ? (step.args as Record<string, unknown>)
                : {},
            description: step?.description ? String(step.description) : undefined,
            why: step?.why ? String(step.why) : undefined,
            policy:
              step?.policy && typeof step.policy === "object"
                ? (step.policy as PersonaToolPolicy)
                : undefined
          }))
          .filter((step) => Number.isFinite(step.idx))

        const nextMap: Record<number, boolean> = {}
        for (const step of steps) {
          nextMap[step.idx] = step.policy?.allow !== false
        }
        setApprovedStepMap(nextMap)
        const memoryPayload =
          payload?.memory && typeof payload.memory === "object"
            ? (payload.memory as PersonaMemoryUsage)
            : undefined
        const companionPayload =
          payload?.companion && typeof payload.companion === "object"
            ? (payload.companion as PersonaCompanionUsage)
            : undefined
        setPendingPlan({
          planId,
          steps,
          memory: memoryPayload,
          companion: companionPayload
        })
        appendLog("tool", `Plan proposed (${steps.length} step${steps.length === 1 ? "" : "s"})`)
        return
      }

      if (eventType === "assistant_delta") {
        const textDelta = String(payload?.text_delta || "").trim()
        if (
          personaSetupWizardIsSetupRequired &&
          personaSetupWizardCurrentStep === "test" &&
          setupWizardAwaitingLiveResponseRef.current
        ) {
          if (textDelta) {
            setSetupTestOutcome({
              kind: "live_success",
              text: setupWizardLastLiveTextRef.current,
              responseText: textDelta
            })
            setupWizardAwaitingLiveResponseRef.current = false
            if (setupLiveDetourRef.current) {
              void emitSetupAnalyticsEvent({
                eventType: "detour_returned",
                step: "test",
                detourSource: setupLiveDetourRef.current.source
              })
              setSetupLiveDetour(null)
              setSetupTestResumeNote(
                "Live session responded. Finish setup when you're ready."
              )
            }
          }
        }
        if (
          textDelta &&
          activeTabRef.current === "live" &&
          setupHandoffRef.current &&
          !setupHandoffRef.current.compact
        ) {
          consumeSetupHandoffAction("live_response_received")
        }
        appendLog("assistant", String(payload?.text_delta || ""))
        return
      }

      if (eventType === "partial_transcript") {
        appendLog("user", String(payload?.text_delta || ""))
        return
      }

      if (eventType === "tool_call") {
        appendLog(
          "tool",
          `Calling ${String(payload?.tool || "tool")} (step ${String(payload?.step_idx ?? "?")})`
        )
        return
      }

      if (eventType === "tool_result") {
        const approvalPayload =
          payload?.approval && typeof payload.approval === "object"
            ? (payload.approval as PersonaRuntimeApprovalPayload)
            : null
        if (approvalPayload) {
          const scopeContext = _coerceGovernanceContext(approvalPayload.scope_context)
          const durationOptions = Array.isArray(approvalPayload.duration_options)
            ? approvalPayload.duration_options
                .map((entry) => String(entry || "").trim())
                .filter(
                  (entry): entry is PersonaRuntimeApprovalDuration =>
                    entry === "once" || entry === "session" || entry === "conversation"
                )
            : []
          const request: PersonaRuntimeApprovalRequest = {
            key: _approvalRequestKey(
              approvalPayload,
              payload as Record<string, unknown>
            ),
            approval_policy_id:
              typeof approvalPayload.approval_policy_id === "number"
                ? approvalPayload.approval_policy_id
                : null,
            mode: approvalPayload.mode ? String(approvalPayload.mode) : null,
            tool_name: String(
              approvalPayload.tool_name || payload?.tool || "tool"
            ),
            context_key: String(approvalPayload.context_key || ""),
            conversation_id: approvalPayload.conversation_id
              ? String(approvalPayload.conversation_id)
              : null,
            scope_key: String(approvalPayload.scope_key || ""),
            reason: approvalPayload.reason ? String(approvalPayload.reason) : null,
            duration_options: durationOptions.length ? durationOptions : ["once"],
            selected_duration: durationOptions[0] || "once",
            arguments_summary:
              approvalPayload.arguments_summary &&
              typeof approvalPayload.arguments_summary === "object"
                ? (approvalPayload.arguments_summary as Record<string, unknown>)
                : {},
            scope_context: scopeContext,
            session_id: payload?.session_id ? String(payload.session_id) : sessionId,
            plan_id: payload?.plan_id ? String(payload.plan_id) : null,
            step_idx:
              typeof payload?.step_idx === "number"
                ? payload.step_idx
                : Number.parseInt(String(payload?.step_idx ?? ""), 10),
            step_type: payload?.step_type ? String(payload.step_type) : "mcp_tool",
            tool: payload?.tool ? String(payload.tool) : null,
            args:
              payload?.args && typeof payload.args === "object"
                ? (payload.args as Record<string, unknown>)
                : {},
            why: payload?.why ? String(payload.why) : null,
            description: payload?.description ? String(payload.description) : null
          }
          if (resolvedApprovalSnapshot?.key === request.key) {
            clearResolvedApprovalFadeTimer()
            setResolvedApprovalSnapshot(null)
          }
          setPendingApprovals((prev) => {
            const next = prev.filter((entry) => entry.key !== request.key)
            return [...next, request]
          })
          appendLog("notice", `Runtime approval required for ${request.tool_name}`)
          return
        }
        const governanceContext = _coerceGovernanceContext(
          payload?.external_access ?? payload?.path_scope
        )
        const externalDenyMessage = _formatGovernanceDenyMessage(
          governanceContext,
          payload?.reason_code ? String(payload.reason_code) : null
        )
        if (externalDenyMessage) {
          appendLog("notice", externalDenyMessage)
          return
        }
        const output = payload?.output ?? payload?.result
        const message =
          output == null
            ? JSON.stringify(payload)
            : typeof output === "string"
              ? output
              : JSON.stringify(output)
        appendLog("tool", `Result step ${String(payload?.step_idx ?? "?")}: ${message}`)
        return
      }

      if (eventType === "notice") {
        const reasonCode = String(payload?.reason_code || "").trim().toUpperCase()
        if (
          reasonCode === "VOICE_TURN_PROCESSING" ||
          reasonCode === "VOICE_TOOL_EXECUTION_PROCESSING"
        ) {
          return
        }
        appendLog("notice", String(payload?.message || "notice"))
        return
      }

      if (eventType === "tts_audio") {
        appendLog("notice", "Received persona TTS audio chunk")
      }
    },
    [
      appendLog,
      clearResolvedApprovalFadeTimer,
      consumeSetupHandoffAction,
      emitSetupAnalyticsEvent,
      liveVoiceController,
      personaId,
      personaSetupWizardCurrentStep,
      personaSetupWizardIsSetupRequired,
      resolvedApprovalSnapshot,
      sessionId,
      setVisualRuntimeOverride,
    ]
  )

  return handleIncomingPayload
}
