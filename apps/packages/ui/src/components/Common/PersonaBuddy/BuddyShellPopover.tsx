import React from "react"
import { useTranslation } from "react-i18next"
import { Mic, MicOff, Palette, Send, Square, Sparkles } from "lucide-react"
import { Link } from "react-router-dom"

import type {
  PersonaBuddyLiveControlView,
  PersonaBuddySummary
} from "@/types/persona-buddy"
import { buildPersonaGardenRoute } from "@/utils/persona-garden-route"

import {
  getPersonaVisualDiagnosticToneClassName,
  type PersonaVisualDiagnostic
} from "./personaVisualDiagnostics"

type BuddyShellPopoverProps = {
  buddySummary: PersonaBuddySummary
  personaId?: string | null
  visualDiagnostic?: PersonaVisualDiagnostic | null
  liveControl?: PersonaBuddyLiveControlView | null
}

const generateDraftClientMessageId = () => {
  if (globalThis.crypto?.randomUUID) {
    return `persona-buddy-draft:${globalThis.crypto.randomUUID()}`
  }
  return `persona-buddy-draft:${Date.now().toString(36)}:${Math.random()
    .toString(36)
    .slice(2)}`
}

export const BuddyShellPopover: React.FC<BuddyShellPopoverProps> = ({
  buddySummary,
  personaId = null,
  visualDiagnostic = null,
  liveControl = null
}) => {
  const { t } = useTranslation("common")
  const [draft, setDraft] = React.useState("")
  const [draftClientMessageId, setDraftClientMessageId] =
    React.useState<string | null>(null)
  const [sendError, setSendError] = React.useState<string | null>(null)
  const [sending, setSending] = React.useState(false)
  const normalizedPersonaId = String(personaId ?? "").trim()
  const visualsRoute = normalizedPersonaId
    ? buildPersonaGardenRoute({
        personaId: normalizedPersonaId,
        tab: "visuals"
      })
    : null
  const liveRoute = normalizedPersonaId
    ? buildPersonaGardenRoute({
        personaId: normalizedPersonaId,
        tab: "live"
      })
    : buildPersonaGardenRoute({ tab: "live" })
  const focusedSession = liveControl?.focusedSession ?? null
  const needsApproval = (focusedSession?.pendingApprovalCount ?? 0) > 0
  const sessionOptions = liveControl?.sessions ?? []
  const voiceCapable = Boolean(
    focusedSession &&
      (liveControl?.voiceAvailable || focusedSession.capabilities?.voice)
  )
  const voiceState = String(liveControl?.voiceState ?? "").trim().toLowerCase()
  const isListening = voiceState === "listening"

  const handleDraftChange = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setDraft(event.target.value)
    setDraftClientMessageId(null)
    setSendError(null)
  }

  const handleSend = async () => {
    const trimmed = draft.trim()
    if (!trimmed || !liveControl || sending) return
    setSending(true)
    setSendError(null)
    const clientMessageId =
      draftClientMessageId ?? generateDraftClientMessageId()
    setDraftClientMessageId(clientMessageId)
    try {
      if (!liveControl.focusedSession || !liveControl.canSendText) {
        await liveControl.startTextSession(normalizedPersonaId || null)
      }
      const result = await liveControl.sendText(trimmed, { clientMessageId })
      if (result.ok) {
        setDraft("")
        setDraftClientMessageId(null)
      } else {
        setSendError(result.error || "Failed to send message")
      }
    } catch (error) {
      setSendError(
        error instanceof Error ? error.message : "Failed to send message"
      )
    } finally {
      setSending(false)
    }
  }

  return (
    <div
      data-testid="persona-buddy-popover"
      className="min-w-[220px] rounded-2xl border border-border bg-bg/95 p-3 shadow-xl backdrop-blur"
    >
      <div className="text-xs uppercase tracking-[0.18em] text-text-muted">
        {t("personaBuddy.title", "Persona Buddy")}
      </div>
      <div className="mt-2 text-sm font-semibold text-text">
        {buddySummary.persona_name}
      </div>
      {buddySummary.role_summary ? (
        <div className="mt-1 text-xs leading-5 text-text-muted">
          {buddySummary.role_summary}
        </div>
      ) : null}
      {visualDiagnostic ? (
        <div
          data-testid="persona-buddy-visual-diagnostic-detail"
          data-severity={visualDiagnostic.severity}
          className={`mt-3 rounded-lg border px-2.5 py-2 text-xs leading-5 ${getPersonaVisualDiagnosticToneClassName(visualDiagnostic.severity)}`}
        >
          <div className="font-medium text-inherit">{visualDiagnostic.title}</div>
          <div>{visualDiagnostic.message}</div>
        </div>
      ) : null}
      {liveControl ? (
        <div className="mt-3 space-y-2 border-t border-border pt-3">
          {sessionOptions.length > 1 ? (
            <select
              data-testid="persona-buddy-session-select"
              value={liveControl.focusedSessionId ?? ""}
              onChange={(event) => {
                const sessionId = event.target.value
                if (sessionId) {
                  void liveControl.focusSession(sessionId)
                }
              }}
              className="w-full rounded-md border border-border bg-surface px-2 py-1.5 text-xs text-text"
            >
              {sessionOptions.map((session) => (
                <option key={session.sessionId} value={session.sessionId}>
                  {session.personaName}
                </option>
              ))}
            </select>
          ) : null}

          {needsApproval ? (
            <div
              data-testid="persona-buddy-approval-needed"
              className="rounded-md border border-warning/40 bg-warning/10 px-2.5 py-2 text-xs font-medium text-text"
            >
              Needs approval
            </div>
          ) : null}

          <div className="flex gap-2">
            <button
              type="button"
              onClick={() =>
                void liveControl.startTextSession(normalizedPersonaId || null)
              }
              className="inline-flex flex-1 items-center justify-center gap-1 rounded-md border border-border bg-surface px-2 py-1.5 text-xs font-medium text-text hover:bg-surface2"
            >
              <Sparkles aria-hidden="true" className="h-3.5 w-3.5" />
              Start
            </button>
            <button
              type="button"
              onClick={() => void liveControl.stopSession(focusedSession?.sessionId)}
              disabled={!focusedSession}
              className="inline-flex flex-1 items-center justify-center gap-1 rounded-md border border-border bg-surface px-2 py-1.5 text-xs font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Square aria-hidden="true" className="h-3.5 w-3.5" />
              Stop
            </button>
          </div>
          {voiceCapable ? (
            <Link
              data-testid="persona-buddy-voice-link"
              to={liveRoute}
              className="inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2"
            >
              {isListening ? (
                <MicOff aria-hidden="true" className="h-3.5 w-3.5" />
              ) : (
                <Mic aria-hidden="true" className="h-3.5 w-3.5" />
              )}
              {isListening ? "Stop listening" : "Listen"}
            </Link>
          ) : null}

          <textarea
            data-testid="persona-buddy-text-input"
            value={draft}
            onChange={handleDraftChange}
            rows={3}
            className="w-full resize-none rounded-md border border-border bg-surface px-2 py-1.5 text-xs text-text"
            placeholder="Message your buddy"
          />
          {sendError ? (
            <div className="text-xs font-medium text-danger">{sendError}</div>
          ) : null}
          <button
            type="button"
            onClick={handleSend}
            disabled={sending || !draft.trim()}
            className="inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send aria-hidden="true" className="h-3.5 w-3.5" />
            Send
          </button>
        </div>
      ) : null}
      <Link
        data-testid="persona-buddy-open-live-link"
        to={liveRoute}
        className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2"
      >
        {t("personaBuddy.openLive", "Open Full Live View")}
      </Link>
      {visualsRoute ? (
        <Link
          data-testid="persona-buddy-open-visuals-link"
          to={visualsRoute}
          className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-text hover:bg-surface2"
        >
          <Palette aria-hidden="true" className="h-3.5 w-3.5" />
          <span>{t("personaBuddy.openVisuals", "Choose/Change Buddy")}</span>
        </Link>
      ) : null}
    </div>
  )
}

export default BuddyShellPopover
