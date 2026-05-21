import React from "react"
import { useBlocker, UNSAFE_DataRouterContext } from "react-router-dom"
import { useTranslation } from "react-i18next"

import { tldwClient } from "@/services/tldw/TldwApiClient"

import type {
  PersonaStateDocsResponse,
  PersonaStateHistoryEntry,
  PersonaStateHistoryResponse,
  UnsavedStateDiscardReason,
} from "../personaTypes"
import {
  _historyEntrySortEpoch,
  PERSONA_STATE_EDITOR_EXPANDED_PREF_KEY,
  PERSONA_STATE_HISTORY_ORDER_PREF_KEY,
  _readBoolPreference,
  _readHistoryOrderPreference,
  _confirmWithBrowserPrompt,
} from "../personaTypes"

// ── Types ──

export interface UsePersonaStateDocsDeps {
  /** Returns the effective persona id for state operations. */
  getTargetPersonaId: (override?: string) => string | null
  /** Append a notice to the live log. */
  appendLog: (kind: "user" | "assistant" | "tool" | "notice", text: string) => void
  /** Set the route-level error banner. */
  setError: React.Dispatch<React.SetStateAction<string | null>>
}

export interface UsePersonaStateDocsReturn {
  // ── Editor values ──
  soulMd: string
  setSoulMd: React.Dispatch<React.SetStateAction<string>>
  identityMd: string
  setIdentityMd: React.Dispatch<React.SetStateAction<string>>
  heartbeatMd: string
  setHeartbeatMd: React.Dispatch<React.SetStateAction<string>>
  savedSoulMd: string
  savedIdentityMd: string
  savedHeartbeatMd: string
  stateLastModified: string | null

  // ── Loading / saving flags ──
  personaStateLoading: boolean
  personaStateSaving: boolean
  personaStateHistoryLoading: boolean
  personaStateHistoryLoaded: boolean

  // ── History ──
  personaStateHistory: PersonaStateHistoryEntry[]
  setPersonaStateHistory: React.Dispatch<React.SetStateAction<PersonaStateHistoryEntry[]>>
  setPersonaStateHistoryLoaded: React.Dispatch<React.SetStateAction<boolean>>
  orderedPersonaStateHistory: PersonaStateHistoryEntry[]
  personaStateHistoryOrder: "newest" | "oldest"
  setPersonaStateHistoryOrder: React.Dispatch<React.SetStateAction<"newest" | "oldest">>
  restoringStateEntryId: string | null
  archivingStateEntryId: string | null

  // ── Editor expansion ──
  personaStateEditorExpanded: boolean
  setPersonaStateEditorExpanded: React.Dispatch<React.SetStateAction<boolean>>

  // ── Derived ──
  hasUnsavedPersonaStateChanges: boolean
  stateDirtyLabel: string
  stateEditorToggleLabel: string

  // ── Actions ──
  applyPersonaStatePayload: (payload: PersonaStateDocsResponse) => void
  loadPersonaStateDocs: (personaIdOverride?: string, options?: { silent?: boolean }) => Promise<boolean>
  loadPersonaStateHistory: (personaIdOverride?: string) => Promise<boolean>
  savePersonaStateDocs: () => Promise<boolean | undefined>
  restorePersonaStateHistoryEntry: (entryId: string) => Promise<boolean>
  archivePersonaStateHistoryEntry: (entryId: string) => Promise<boolean>
  revertPersonaStateDraft: () => void
  confirmDiscardUnsavedStateDrafts: (reason?: UnsavedStateDiscardReason) => boolean
}

// ── Idle route-blocker sentinel ──

const IDLE_ROUTE_BLOCKER: ReturnType<typeof useBlocker> = {
  state: "unblocked",
  proceed: undefined,
  reset: undefined,
} as ReturnType<typeof useBlocker>

const useCompatibleRouteBlocker = (
  when: boolean
): ReturnType<typeof useBlocker> => {
  const dataRouterContext = React.useContext(UNSAFE_DataRouterContext)
  if (!dataRouterContext) return IDLE_ROUTE_BLOCKER
  return useBlocker(when)
}

// ── Hook ──

export function usePersonaStateDocs(
  deps: UsePersonaStateDocsDeps
): UsePersonaStateDocsReturn {
  const { getTargetPersonaId, appendLog, setError } = deps
  const { t } = useTranslation(["sidepanel", "common"])

  // ── Editor state ──
  const [soulMd, setSoulMd] = React.useState("")
  const [identityMd, setIdentityMd] = React.useState("")
  const [heartbeatMd, setHeartbeatMd] = React.useState("")
  const [savedSoulMd, setSavedSoulMd] = React.useState("")
  const [savedIdentityMd, setSavedIdentityMd] = React.useState("")
  const [savedHeartbeatMd, setSavedHeartbeatMd] = React.useState("")
  const [stateLastModified, setStateLastModified] = React.useState<string | null>(null)

  // ── Loading / saving flags ──
  const [personaStateLoading, setPersonaStateLoading] = React.useState(false)
  const [personaStateSaving, setPersonaStateSaving] = React.useState(false)
  const [personaStateHistoryLoading, setPersonaStateHistoryLoading] = React.useState(false)
  const [personaStateHistoryLoaded, setPersonaStateHistoryLoaded] = React.useState(false)

  // ── History ──
  const [personaStateHistory, setPersonaStateHistory] = React.useState<
    PersonaStateHistoryEntry[]
  >([])
  const [personaStateHistoryOrder, setPersonaStateHistoryOrder] =
    React.useState<"newest" | "oldest">(_readHistoryOrderPreference)
  const [restoringStateEntryId, setRestoringStateEntryId] = React.useState<string | null>(null)
  const [archivingStateEntryId, setArchivingStateEntryId] = React.useState<string | null>(null)

  // ── Editor expansion ──
  const [personaStateEditorExpanded, setPersonaStateEditorExpanded] =
    React.useState(() =>
      _readBoolPreference(PERSONA_STATE_EDITOR_EXPANDED_PREF_KEY, true)
    )

  // ── Derived ──
  const hasUnsavedPersonaStateChanges =
    soulMd !== savedSoulMd ||
    identityMd !== savedIdentityMd ||
    heartbeatMd !== savedHeartbeatMd

  const stateDirtyLabel = hasUnsavedPersonaStateChanges
    ? t("sidepanel:persona.stateDirty", "unsaved")
    : t("sidepanel:persona.stateSaved", "saved")
  const stateEditorToggleLabel = personaStateEditorExpanded
    ? t("sidepanel:persona.stateEditorHide", "Hide editor")
    : t("sidepanel:persona.stateEditorShow", "Show editor")

  const orderedPersonaStateHistory = React.useMemo(() => {
    const sorted = [...personaStateHistory].sort(
      (left, right) => _historyEntrySortEpoch(left) - _historyEntrySortEpoch(right)
    )
    if (personaStateHistoryOrder === "newest") {
      sorted.reverse()
    }
    return sorted
  }, [personaStateHistory, personaStateHistoryOrder])

  // ── Unsaved-changes prompt ──

  const getUnsavedStateDiscardPrompt = React.useCallback(
    (reason: UnsavedStateDiscardReason): string => {
      switch (reason) {
        case "connect":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptConnect",
            "You have unsaved state-doc changes. Connect and discard local drafts?"
          )
        case "disconnect":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptDisconnect",
            "You have unsaved state-doc changes. Disconnect and discard local drafts?"
          )
        case "reload_state":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptReloadState",
            "You have unsaved state-doc changes. Load state and discard local drafts?"
          )
        case "persona_switch":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptPersonaSwitch",
            "You have unsaved state-doc changes. Switch persona and discard local drafts?"
          )
        case "session_switch":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptSessionSwitch",
            "You have unsaved state-doc changes. Switch session and discard local drafts?"
          )
        case "restore_state":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptRestoreState",
            "You have unsaved state-doc changes. Restore this state version and discard local drafts?"
          )
        case "archive_state":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptArchiveState",
            "You have unsaved state-doc changes. Archive this state version and discard local drafts?"
          )
        case "route_transition":
          return t(
            "sidepanel:persona.unsavedStateDiscardPromptRouteTransition",
            "You have unsaved state-doc changes. Leave this page and discard local drafts?"
          )
        case "before_unload":
          return t(
            "sidepanel:persona.unsavedStateBeforeUnloadPrompt",
            "You have unsaved state-doc changes. Leave this page without saving?"
          )
        case "generic":
        default:
          return t(
            "sidepanel:persona.unsavedStateDiscardPrompt",
            "You have unsaved state-doc changes. Discard local drafts?"
          )
      }
    },
    [t]
  )

  const confirmDiscardUnsavedStateDrafts = React.useCallback(
    (reason: UnsavedStateDiscardReason = "generic"): boolean => {
      if (
        soulMd === savedSoulMd &&
        identityMd === savedIdentityMd &&
        heartbeatMd === savedHeartbeatMd
      ) {
        return true
      }
      return _confirmWithBrowserPrompt(getUnsavedStateDiscardPrompt(reason))
    },
    [
      getUnsavedStateDiscardPrompt,
      heartbeatMd,
      identityMd,
      savedHeartbeatMd,
      savedIdentityMd,
      savedSoulMd,
      soulMd,
    ]
  )

  // ── Apply payload ──

  const applyPersonaStatePayload = React.useCallback(
    (payload: PersonaStateDocsResponse) => {
      const nextSoulMd = String(payload?.soul_md ?? "")
      const nextIdentityMd = String(payload?.identity_md ?? "")
      const nextHeartbeatMd = String(payload?.heartbeat_md ?? "")
      setSoulMd(nextSoulMd)
      setIdentityMd(nextIdentityMd)
      setHeartbeatMd(nextHeartbeatMd)
      setSavedSoulMd(nextSoulMd)
      setSavedIdentityMd(nextIdentityMd)
      setSavedHeartbeatMd(nextHeartbeatMd)
      setStateLastModified(
        payload?.last_modified ? String(payload.last_modified) : null
      )
    },
    []
  )

  // ── Load ──

  const loadPersonaStateDocs = React.useCallback(
    async (personaIdOverride?: string, options?: { silent?: boolean }) => {
      const personaId = getTargetPersonaId(personaIdOverride)
      if (!personaId) return false
      const silent = options?.silent === true
      if (!silent && !confirmDiscardUnsavedStateDrafts("reload_state")) {
        return false
      }
      setPersonaStateLoading(true)
      if (!silent) {
        setError(null)
      }
      try {
        const stateResp = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/state` as any,
          { method: "GET" }
        )
        if (!stateResp.ok) {
          if (!silent) {
            throw new Error(stateResp.error || "Failed to load persona state docs")
          }
          return false
        }
        const statePayload = (await stateResp.json()) as PersonaStateDocsResponse
        applyPersonaStatePayload(statePayload)
        return true
      } catch (err: any) {
        if (!silent) {
          setError(String(err?.message || "Failed to load persona state docs"))
        }
        return false
      } finally {
        setPersonaStateLoading(false)
      }
    },
    [applyPersonaStatePayload, confirmDiscardUnsavedStateDrafts, getTargetPersonaId, setError]
  )

  // ── Load history ──

  const loadPersonaStateHistory = React.useCallback(
    async (personaIdOverride?: string) => {
      const personaId = getTargetPersonaId(personaIdOverride)
      if (!personaId) return false
      setPersonaStateHistoryLoading(true)
      setError(null)
      try {
        const historyResp = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/state/history?include_archived=true&limit=30` as any,
          { method: "GET" }
        )
        if (!historyResp.ok) {
          throw new Error(historyResp.error || "Failed to load persona state history")
        }
        const historyPayload = (await historyResp.json()) as PersonaStateHistoryResponse
        const entries = Array.isArray(historyPayload?.entries)
          ? historyPayload.entries
          : []
        setPersonaStateHistory(entries)
        setPersonaStateHistoryLoaded(true)
        return true
      } catch (err: any) {
        setError(String(err?.message || "Failed to load persona state history"))
        return false
      } finally {
        setPersonaStateHistoryLoading(false)
      }
    },
    [getTargetPersonaId, setError]
  )

  // ── Save ──

  const savePersonaStateDocs = React.useCallback(async () => {
    const personaId = getTargetPersonaId()
    if (!personaId || personaStateSaving) return
    if (
      soulMd === savedSoulMd &&
      identityMd === savedIdentityMd &&
      heartbeatMd === savedHeartbeatMd
    ) {
      return true
    }
    setPersonaStateSaving(true)
    setError(null)
    try {
      const toNullable = (value: string): string | null =>
        String(value || "").trim().length > 0 ? value : null
      const saveResp = await tldwClient.fetchWithAuth(
        `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/state` as any,
        {
          method: "PUT",
          body: {
            soul_md: toNullable(soulMd),
            identity_md: toNullable(identityMd),
            heartbeat_md: toNullable(heartbeatMd),
          },
        }
      )
      if (!saveResp.ok) {
        throw new Error(saveResp.error || "Failed to save persona state docs")
      }
      const savePayload = (await saveResp.json()) as PersonaStateDocsResponse
      applyPersonaStatePayload(savePayload)
      if (personaStateHistoryLoaded) {
        void loadPersonaStateHistory(personaId)
      }
      appendLog("notice", "Saved persona state docs")
      return true
    } catch (err: any) {
      setError(String(err?.message || "Failed to save persona state docs"))
      return false
    } finally {
      setPersonaStateSaving(false)
    }
  }, [
    appendLog,
    applyPersonaStatePayload,
    heartbeatMd,
    identityMd,
    getTargetPersonaId,
    loadPersonaStateHistory,
    personaStateHistoryLoaded,
    personaStateSaving,
    savedHeartbeatMd,
    savedIdentityMd,
    savedSoulMd,
    setError,
    soulMd,
  ])

  // ── Restore history entry ──

  const restorePersonaStateHistoryEntry = React.useCallback(
    async (entryId: string) => {
      const personaId = getTargetPersonaId()
      const trimmedEntryId = String(entryId || "").trim()
      if (!personaId || !trimmedEntryId || restoringStateEntryId) return false
      if (!confirmDiscardUnsavedStateDrafts("restore_state")) {
        return false
      }
      setRestoringStateEntryId(trimmedEntryId)
      setError(null)
      try {
        const restoreResp = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/state/restore` as any,
          {
            method: "POST",
            body: { entry_id: trimmedEntryId },
          }
        )
        if (!restoreResp.ok) {
          throw new Error(
            restoreResp.error || "Failed to restore persona state version"
          )
        }
        const restorePayload =
          (await restoreResp.json()) as PersonaStateDocsResponse
        applyPersonaStatePayload(restorePayload)
        await loadPersonaStateHistory(personaId)
        appendLog("notice", "Restored persona state version")
        return true
      } catch (err: any) {
        setError(
          String(err?.message || "Failed to restore persona state version")
        )
        return false
      } finally {
        setRestoringStateEntryId(null)
      }
    },
    [
      appendLog,
      applyPersonaStatePayload,
      confirmDiscardUnsavedStateDrafts,
      getTargetPersonaId,
      loadPersonaStateHistory,
      restoringStateEntryId,
      setError,
    ]
  )

  // ── Archive history entry ──

  const archivePersonaStateHistoryEntry = React.useCallback(
    async (entryId: string) => {
      const personaId = getTargetPersonaId()
      const trimmedEntryId = String(entryId || "").trim()
      if (!personaId || !trimmedEntryId || archivingStateEntryId) return false
      if (!confirmDiscardUnsavedStateDrafts("archive_state")) {
        return false
      }
      const confirmed = _confirmWithBrowserPrompt(
        t(
          "sidepanel:persona.stateArchiveConfirm",
          "Archive this active state version? This removes it from current Persona state without deleting history."
        )
      )
      if (!confirmed) {
        return false
      }
      setArchivingStateEntryId(trimmedEntryId)
      setError(null)
      try {
        const archiveResp = await tldwClient.fetchWithAuth(
          `/api/v1/persona/profiles/${encodeURIComponent(personaId)}/state/archive` as any,
          {
            method: "POST",
            body: { entry_id: trimmedEntryId },
          }
        )
        if (!archiveResp.ok) {
          throw new Error(
            archiveResp.error || "Failed to archive persona state version"
          )
        }
        const archivePayload =
          (await archiveResp.json()) as PersonaStateDocsResponse
        applyPersonaStatePayload(archivePayload)
        await loadPersonaStateHistory(personaId)
        appendLog("notice", "Archived persona state version")
        return true
      } catch (err: any) {
        setError(
          String(err?.message || "Failed to archive persona state version")
        )
        return false
      } finally {
        setArchivingStateEntryId(null)
      }
    },
    [
      appendLog,
      applyPersonaStatePayload,
      archivingStateEntryId,
      confirmDiscardUnsavedStateDrafts,
      getTargetPersonaId,
      loadPersonaStateHistory,
      setError,
      t,
    ]
  )

  // ── Revert ──

  const revertPersonaStateDraft = React.useCallback(() => {
    setSoulMd(savedSoulMd)
    setIdentityMd(savedIdentityMd)
    setHeartbeatMd(savedHeartbeatMd)
  }, [savedHeartbeatMd, savedIdentityMd, savedSoulMd])

  // ── Route blocker ──

  const routeNavigationBlocker = useCompatibleRouteBlocker(
    hasUnsavedPersonaStateChanges
  )

  React.useEffect(() => {
    if (routeNavigationBlocker.state !== "blocked") return
    if (confirmDiscardUnsavedStateDrafts("route_transition")) {
      routeNavigationBlocker.proceed()
    } else {
      routeNavigationBlocker.reset()
    }
  }, [confirmDiscardUnsavedStateDrafts, routeNavigationBlocker])

  // ── beforeunload guard ──

  React.useEffect(() => {
    if (typeof window === "undefined" || !hasUnsavedPersonaStateChanges) return
    const promptMessage = getUnsavedStateDiscardPrompt("before_unload")
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
      event.returnValue = promptMessage
      return promptMessage
    }
    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload)
    }
  }, [getUnsavedStateDiscardPrompt, hasUnsavedPersonaStateChanges])

  // ── Persist preferences ──

  React.useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(
        PERSONA_STATE_EDITOR_EXPANDED_PREF_KEY,
        personaStateEditorExpanded ? "true" : "false"
      )
    } catch {
      // ignore storage access errors
    }
  }, [personaStateEditorExpanded])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    try {
      window.localStorage.setItem(
        PERSONA_STATE_HISTORY_ORDER_PREF_KEY,
        personaStateHistoryOrder
      )
    } catch {
      // ignore storage access errors
    }
  }, [personaStateHistoryOrder])

  return {
    soulMd,
    setSoulMd,
    identityMd,
    setIdentityMd,
    heartbeatMd,
    setHeartbeatMd,
    savedSoulMd,
    savedIdentityMd,
    savedHeartbeatMd,
    stateLastModified,
    personaStateLoading,
    personaStateSaving,
    personaStateHistoryLoading,
    personaStateHistoryLoaded,
    personaStateHistory,
    setPersonaStateHistory,
    setPersonaStateHistoryLoaded,
    orderedPersonaStateHistory,
    personaStateHistoryOrder,
    setPersonaStateHistoryOrder,
    restoringStateEntryId,
    archivingStateEntryId,
    personaStateEditorExpanded,
    setPersonaStateEditorExpanded,
    hasUnsavedPersonaStateChanges,
    stateDirtyLabel,
    stateEditorToggleLabel,
    applyPersonaStatePayload,
    loadPersonaStateDocs,
    loadPersonaStateHistory,
    savePersonaStateDocs,
    restorePersonaStateHistoryEntry,
    archivePersonaStateHistoryEntry,
    revertPersonaStateDraft,
    confirmDiscardUnsavedStateDrafts,
  }
}
