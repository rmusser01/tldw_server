import React from "react"
import { useNavigate } from "react-router-dom"

import { Button } from "@/components/Common/Button"
import { RouteLeavePrompt } from "@/entries/shared/route-leave-prompt"
import { usePresentationPrincipalScope, type PresentationPrincipalBoundaryKind } from "@/hooks/usePresentationPrincipalScope"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useSlidesCapabilities } from "@/hooks/useSlidesCapabilities"
import {
  tldwClient,
  type PresentationDetailResult,
  type StandalonePresentationDetailResult
} from "@/services/tldw/TldwApiClient"
import { StandaloneHtmlDownloadManager } from "./standalone-html-download"
import {
  StandaloneHtmlOutlineController
} from "./standalone-html-outline-client"
import type { StandaloneHtmlOutline } from "./standalone-html-outline.worker"
import {
  acquireStandaloneHtmlRecoveryStorage,
  clearStandaloneHtmlRecovery,
  readStandaloneHtmlRecovery,
  writeStandaloneHtmlRecovery,
  type PresentationPrincipalScope,
  type StandaloneHtmlRecoveryRecord
} from "./standalone-html-recovery"
import {
  preflightStandaloneHtmlSource,
  validateStandaloneHtmlSource,
  type AcceptedStandaloneHtmlSource
} from "./standalone-html-source"
import { StandaloneHtmlSafeOutline } from "./StandaloneHtmlSafeOutline"
import {
  StandaloneHtmlSourceEditor,
  type StandaloneHtmlSourceEditorHandle
} from "./StandaloneHtmlSourceEditor"

type SaveStatus = "Saved" | "Saving" | "Not saved" | "Conflict"
type LoadStatus = "loading" | "ready" | "error"
type RecoveryOperation = "read" | "write" | "cleanup"

type ConflictAction = {
  controller: AbortController
  epoch: number
}

const RECOVERY_OPERATIONS: readonly RecoveryOperation[] = ["read", "write", "cleanup"]

type LoadedStandalone = {
  title: string
  acceptedSource: AcceptedStandaloneHtmlSource
  etag: string
}

type AvailableRecovery = {
  record: StandaloneHtmlRecoveryRecord
  acceptedSource: AcceptedStandaloneHtmlSource
}

type WorkspaceSnapshot = {
  scope: PresentationPrincipalScope
  base: LoadedStandalone
  accepted: AcceptedStandaloneHtmlSource
  latestPreflightCandidate: string
  title: string
  saveStatus: SaveStatus
  message: string | null
  recovery: AvailableRecovery | null
}

const RECOVERY_FAILURE_MESSAGE = "Recovery unavailable. Keep this tab open or download your draft."

const isReadCapabilitySettled = (status: string): boolean =>
  status === "ready" || status === "generation_disabled" || status === "validator_unavailable"

const isStrongEtag = (value: unknown): value is string => {
  if (
    typeof value !== "string" ||
    value.length < 2 ||
    value.charCodeAt(0) !== 0x22 ||
    value.charCodeAt(value.length - 1) !== 0x22
  ) {
    return false
  }
  for (let index = 1; index < value.length - 1; index += 1) {
    const codePoint = value.charCodeAt(index)
    if (
      codePoint !== 0x21 &&
      (codePoint < 0x23 || codePoint > 0x7e) &&
      (codePoint < 0x80 || codePoint > 0xff)
    ) {
      return false
    }
  }
  return true
}

const errorStatus = (error: unknown): number | null => {
  const status = error && typeof error === "object" ? (error as { status?: unknown }).status : null
  return typeof status === "number" && Number.isFinite(status) ? status : null
}

const abortNoThrow = (controller: AbortController | null | undefined) => {
  try {
    controller?.abort()
  } catch {
    // Security cleanup continues even when a platform disposer is unavailable.
  }
}

const disposeNoThrow = (resource: { dispose: () => void } | null | undefined) => {
  try {
    resource?.dispose()
  } catch {
    // Each owned resource is fenced independently before its disposer runs.
  }
}

const validateStandaloneDetail = async (
  result: PresentationDetailResult | StandalonePresentationDetailResult,
  presentationId: string,
  scope: PresentationPrincipalScope
): Promise<LoadedStandalone> => {
  const record = result.record
  const etag = result.etag
  if (
    record.content_kind !== "standalone_html" ||
    record.id !== presentationId ||
    String(record.client_id ?? "") !== scope.principalId ||
    !isStrongEtag(etag)
  ) {
    throw new Error("Presentation detail could not be verified")
  }
  const acceptedSource = await validateStandaloneHtmlSource(record.html_document)
  if (
    !acceptedSource.ok ||
    acceptedSource.digest !== record.html_sha256 ||
    acceptedSource.byteLength !== record.html_bytes
  ) {
    throw new Error("Presentation source could not be verified")
  }
  return { title: record.title, acceptedSource, etag }
}

export const StandaloneHtmlWorkspace: React.FC<{
  presentationId: string
  kindAuthorityPending?: boolean
  kindAuthorityEpoch?: number | null
  kindAuthorityReleaseRequired?: boolean
  onKindAuthoritySettled?: (authorityEpoch: number, releaseSafe: boolean) => void
  isKindAuthorityCurrent?: (
    capturedAuthorityEpoch: number | null,
    presentationId: string
  ) => boolean
}> = ({
  presentationId,
  kindAuthorityPending = false,
  kindAuthorityEpoch = null,
  kindAuthorityReleaseRequired = false,
  onKindAuthoritySettled,
  isKindAuthorityCurrent
}) => {
  const navigate = useNavigate()
  const online = useServerOnline()
  const slides = useSlidesCapabilities()
  const mountedRef = React.useRef(true)
  const operationEpochRef = React.useRef(0)
  const scopeRef = React.useRef<PresentationPrincipalScope | null>(null)
  const lastTrustedScopeRef = React.useRef<PresentationPrincipalScope | null>(null)
  const acceptedRef = React.useRef<AcceptedStandaloneHtmlSource | null>(null)
  const baseRef = React.useRef<LoadedStandalone | null>(null)
  const latestPreflightCandidateRef = React.useRef<string | null>(null)
  const pendingCandidateRef = React.useRef<string | null>(null)
  const candidateEpochRef = React.useRef(0)
  const quarantineRef = React.useRef<WorkspaceSnapshot | null>(null)
  const recoveryFailuresRef = React.useRef(new Set<string>())
  const authorityCleanupScopeRef = React.useRef<PresentationPrincipalScope | null>(null)
  const reportedAuthoritySettlementRef = React.useRef<{
    authorityEpoch: number
    releaseSafe: boolean
  } | null>(null)
  const unresolvedRecoveryCleanupRef = React.useRef(
    new Map<string, PresentationPrincipalScope>()
  )
  const loadControllerRef = React.useRef<AbortController | null>(null)
  const saveControllerRef = React.useRef<AbortController | null>(null)
  const conflictActionControllerRef = React.useRef<AbortController | null>(null)
  const conflictActionEpochRef = React.useRef(0)
  const overwriteEtagRef = React.useRef<string | null>(null)
  const sourceEditorRef = React.useRef<StandaloneHtmlSourceEditorHandle | null>(null)
  const outlineControllerRef = React.useRef<StandaloneHtmlOutlineController | null>(null)
  const outlineRef = React.useRef<StandaloneHtmlOutline | null>(null)
  const downloadManagerRef = React.useRef<StandaloneHtmlDownloadManager | null>(null)
  const slidesRef = React.useRef(slides)
  slidesRef.current = slides

  const [loadStatus, setLoadStatus] = React.useState<LoadStatus>("loading")
  const [title, setTitle] = React.useState("Standalone HTML presentation")
  const [accepted, setAccepted] = React.useState<AcceptedStandaloneHtmlSource | null>(null)
  const [hasPendingCandidate, setHasPendingCandidate] = React.useState(false)
  const [saveStatus, setSaveStatus] = React.useState<SaveStatus>("Saved")
  const [message, setMessage] = React.useState<string | null>(null)
  const [recoveryWarning, setRecoveryWarning] = React.useState<string | null>(null)
  const [recovery, setRecovery] = React.useState<AvailableRecovery | null>(null)
  const [outline, setOutline] = React.useState<StandaloneHtmlOutline | null>(null)
  const [outlineStatus, setOutlineStatus] = React.useState<"current" | "stale" | "failed">("stale")
  const [activeTab, setActiveTab] = React.useState<"code" | "outline">("code")
  const [confirmRecoveryDiscard, setConfirmRecoveryDiscard] = React.useState(false)
  const [confirmLeave, setConfirmLeave] = React.useState(false)
  const [leaveApproved, setLeaveApproved] = React.useState(false)
  const [confirmOverwrite, setConfirmOverwrite] = React.useState(false)
  const [confirmServerDiscard, setConfirmServerDiscard] = React.useState(false)
  const [authoritySettlementReady, setAuthoritySettlementReady] = React.useState(true)
  const [authorityReleaseReady, setAuthorityReleaseReady] = React.useState(true)
  const codeTabRef = React.useRef<HTMLButtonElement | null>(null)
  const outlineTabRef = React.useRef<HTMLButtonElement | null>(null)
  const codeTabId = React.useId()
  const outlineTabId = React.useId()
  const codePanelId = React.useId()
  const outlinePanelId = React.useId()

  const handleTabKeyDown = React.useCallback((
    event: React.KeyboardEvent<HTMLButtonElement>
  ) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return
    event.preventDefault()
    const nextTab = activeTab === "code" ? "outline" : "code"
    const nextTabRef = nextTab === "code" ? codeTabRef : outlineTabRef
    setActiveTab(nextTab)
    nextTabRef.current?.focus()
  }, [activeTab])

  const workspaceStateRef = React.useRef({ title, saveStatus, message, recovery })
  workspaceStateRef.current = { title, saveStatus, message, recovery }

  const readDraftAuthority = React.useCallback(() => {
    const local = acceptedRef.current
    const base = baseRef.current
    const scope = scopeRef.current
    if (local && base && scope) {
      const source = latestPreflightCandidateRef.current ?? local.source
      return {
        scope,
        base,
        source,
        dirty:
          pendingCandidateRef.current !== null ||
          source !== local.source ||
          local.digest !== base.acceptedSource.digest
      }
    }
    const quarantined = quarantineRef.current
    if (!quarantined) return null
    return {
      scope: quarantined.scope,
      base: quarantined.base,
      source: quarantined.latestPreflightCandidate,
      dirty:
        quarantined.latestPreflightCandidate !== quarantined.accepted.source ||
        quarantined.accepted.digest !== quarantined.base.acceptedSource.digest
    }
  }, [])

  const dirty = readDraftAuthority()?.dirty ?? false

  React.useEffect(() => {
    if (!leaveApproved) return
    const timer = window.setTimeout(() => navigate("/presentation-studio"), 0)
    return () => window.clearTimeout(timer)
  }, [leaveApproved, navigate])

  const invalidateConflictAction = React.useCallback(() => {
    const controller = conflictActionControllerRef.current
    conflictActionControllerRef.current = null
    conflictActionEpochRef.current += 1
    abortNoThrow(controller)
  }, [])

  const beginConflictAction = React.useCallback((): ConflictAction => {
    invalidateConflictAction()
    const controller = new AbortController()
    conflictActionControllerRef.current = controller
    return { controller, epoch: conflictActionEpochRef.current }
  }, [invalidateConflictAction])

  const conflictActionIsCurrent = React.useCallback((action: ConflictAction): boolean => (
    conflictActionControllerRef.current === action.controller &&
    conflictActionEpochRef.current === action.epoch &&
    !action.controller.signal.aborted
  ), [])

  const finishConflictAction = React.useCallback((action: ConflictAction) => {
    if (
      conflictActionControllerRef.current === action.controller &&
      conflictActionEpochRef.current === action.epoch
    ) {
      conflictActionControllerRef.current = null
    }
  }, [])

  const adoptConflictAction = React.useCallback((action: ConflictAction): boolean => {
    if (!conflictActionIsCurrent(action)) return false
    conflictActionControllerRef.current = null
    conflictActionEpochRef.current += 1
    return true
  }, [conflictActionIsCurrent])

  const stopOwnedWork = React.useCallback(() => {
    operationEpochRef.current += 1
    invalidateConflictAction()
    const sourceEditor = sourceEditorRef.current
    sourceEditorRef.current = null
    const loadController = loadControllerRef.current
    loadControllerRef.current = null
    const saveController = saveControllerRef.current
    saveControllerRef.current = null
    overwriteEtagRef.current = null
    const outlineController = outlineControllerRef.current
    outlineControllerRef.current = null
    const downloadManager = downloadManagerRef.current
    downloadManagerRef.current = null
    disposeNoThrow(sourceEditor)
    abortNoThrow(loadController)
    abortNoThrow(saveController)
    disposeNoThrow(outlineController)
    disposeNoThrow(downloadManager)
  }, [invalidateConflictAction])

  const clearPendingCandidate = React.useCallback((resetLatest: boolean) => {
    if (pendingCandidateRef.current !== null) candidateEpochRef.current += 1
    pendingCandidateRef.current = null
    setHasPendingCandidate(false)
    if (resetLatest) {
      latestPreflightCandidateRef.current = acceptedRef.current?.source ?? null
    }
  }, [])

  const publishRecoveryAvailability = React.useCallback(() => {
    setRecoveryWarning(
      recoveryFailuresRef.current.size > 0 ? RECOVERY_FAILURE_MESSAGE : null
    )
  }, [])

  const recoveryScopeKey = React.useCallback(
    (scope: PresentationPrincipalScope) =>
      `${scope.principalScope}|${encodeURIComponent(presentationId)}`,
    [presentationId]
  )

  const recoveryFailureKey = React.useCallback(
    (scope: PresentationPrincipalScope, operation: RecoveryOperation) =>
      `${recoveryScopeKey(scope)}|${operation}`,
    [recoveryScopeKey]
  )

  const clearRecoveryFailureMarkers = React.useCallback(
    (scope: PresentationPrincipalScope, operations: readonly RecoveryOperation[]) => {
      for (const operation of operations) {
        recoveryFailuresRef.current.delete(recoveryFailureKey(scope, operation))
      }
      publishRecoveryAvailability()
    },
    [publishRecoveryAvailability, recoveryFailureKey]
  )

  const markRecoveryUnavailable = React.useCallback(
    (scope: PresentationPrincipalScope, operation: RecoveryOperation) => {
      recoveryFailuresRef.current.add(recoveryFailureKey(scope, operation))
      publishRecoveryAvailability()
    },
    [publishRecoveryAvailability, recoveryFailureKey]
  )

  const retryUnresolvedRecoveryCleanup = React.useCallback(
    (storage: Storage) => {
      for (const [scopeKey, unresolvedScope] of unresolvedRecoveryCleanupRef.current) {
        if (clearStandaloneHtmlRecovery(storage, unresolvedScope, presentationId)) {
          unresolvedRecoveryCleanupRef.current.delete(scopeKey)
          for (const operation of RECOVERY_OPERATIONS) {
            recoveryFailuresRef.current.delete(recoveryFailureKey(unresolvedScope, operation))
          }
          if (
            authorityCleanupScopeRef.current?.principalScope ===
            unresolvedScope.principalScope
          ) {
            authorityCleanupScopeRef.current = null
            setAuthoritySettlementReady(true)
            setAuthorityReleaseReady(true)
          }
        }
      }
      publishRecoveryAvailability()
    },
    [presentationId, publishRecoveryAvailability, recoveryFailureKey]
  )

  const acquireRecoveryStorage = React.useCallback(
    (scope: PresentationPrincipalScope, operation: RecoveryOperation): Storage | null => {
      const result = acquireStandaloneHtmlRecoveryStorage()
      if (result.ok === false) {
        markRecoveryUnavailable(scope, operation)
        return null
      }
      retryUnresolvedRecoveryCleanup(result.storage as Storage)
      return result.storage as Storage
    },
    [markRecoveryUnavailable, retryUnresolvedRecoveryCleanup]
  )

  const clearRecoveryForScope = React.useCallback(
    (scope: PresentationPrincipalScope): boolean => {
      const scopeKey = recoveryScopeKey(scope)
      const storage = acquireRecoveryStorage(scope, "cleanup")
      if (!storage) {
        unresolvedRecoveryCleanupRef.current.set(scopeKey, scope)
        return false
      }
      const cleared = clearStandaloneHtmlRecovery(storage, scope, presentationId)
      if (!cleared) {
        unresolvedRecoveryCleanupRef.current.set(scopeKey, scope)
        markRecoveryUnavailable(scope, "cleanup")
        return false
      }
      unresolvedRecoveryCleanupRef.current.delete(scopeKey)
      clearRecoveryFailureMarkers(scope, RECOVERY_OPERATIONS)
      return true
    },
    [
      acquireRecoveryStorage,
      clearRecoveryFailureMarkers,
      markRecoveryUnavailable,
      presentationId,
      recoveryScopeKey
    ]
  )

  const writeRecoveryFor = React.useCallback(
    (
      scope: PresentationPrincipalScope,
      base: LoadedStandalone,
      next: Pick<AcceptedStandaloneHtmlSource, "source" | "digest"> | { source: string }
    ): boolean => {
      const storage = acquireRecoveryStorage(scope, "write")
      if (!storage) return false
      const result = writeStandaloneHtmlRecovery(storage, scope, {
        presentationId,
        baseEtag: base.etag,
        baseDigest: base.acceptedSource.digest,
        acceptedSource: next,
        updatedAt: Date.now()
      })
      if (result.ok === false) {
        markRecoveryUnavailable(scope, "write")
        return false
      }
      unresolvedRecoveryCleanupRef.current.delete(recoveryScopeKey(scope))
      recoveryFailuresRef.current.delete(recoveryFailureKey(scope, "cleanup"))
      recoveryFailuresRef.current.delete(recoveryFailureKey(scope, "write"))
      publishRecoveryAvailability()
      return true
    },
    [
      acquireRecoveryStorage,
      markRecoveryUnavailable,
      presentationId,
      publishRecoveryAvailability,
      recoveryFailureKey,
      recoveryScopeKey
    ]
  )

  const flushDraftAuthority = React.useCallback(() => {
    const authority = readDraftAuthority()
    if (!authority) return
    const preflight = preflightStandaloneHtmlSource(authority.source)
    if (preflight.ok === false) return
    if (authority.source === authority.base.acceptedSource.source) {
      clearRecoveryForScope(authority.scope)
      return
    }
    writeRecoveryFor(authority.scope, authority.base, { source: authority.source })
  }, [clearRecoveryForScope, readDraftAuthority, writeRecoveryFor])

  const quarantineActive = React.useCallback(() => {
    const scope = scopeRef.current
    const base = baseRef.current
    const local = acceptedRef.current
    if (scope && base && local) {
      const currentState = workspaceStateRef.current
      quarantineRef.current = {
        scope,
        base,
        accepted: local,
        latestPreflightCandidate: latestPreflightCandidateRef.current ?? local.source,
        title: currentState.title,
        saveStatus:
          currentState.saveStatus === "Saving"
            ? local.digest === base.acceptedSource.digest
              ? "Saved"
              : "Not saved"
            : currentState.saveStatus,
        message: currentState.message,
        recovery: currentState.recovery
      }
    }
    workspaceStateRef.current = {
      title: "Standalone HTML presentation",
      saveStatus: "Saved",
      message: null,
      recovery: null
    }
    stopOwnedWork()
    acceptedRef.current = null
    baseRef.current = null
    scopeRef.current = null
    latestPreflightCandidateRef.current = null
    clearPendingCandidate(false)
    setAccepted(null)
    outlineRef.current = null
    setOutline(null)
    setOutlineStatus("stale")
    setRecovery(null)
    setMessage(null)
    setConfirmOverwrite(false)
    setConfirmServerDiscard(false)
    setLoadStatus("loading")
  }, [clearPendingCandidate, stopOwnedWork])

  const scrubActive = React.useCallback(
    (clearStoredRecovery: boolean): boolean => {
      workspaceStateRef.current = {
        title: "Standalone HTML presentation",
        saveStatus: "Saved",
        message: null,
        recovery: null
      }
      stopOwnedWork()
      const previousScope = lastTrustedScopeRef.current
      quarantineRef.current = null
      acceptedRef.current = null
      baseRef.current = null
      scopeRef.current = null
      latestPreflightCandidateRef.current = null
      clearPendingCandidate(false)
      if (clearStoredRecovery) lastTrustedScopeRef.current = null
      setAccepted(null)
      setTitle("Standalone HTML presentation")
      setSaveStatus("Saved")
      outlineRef.current = null
      setOutline(null)
      setOutlineStatus("stale")
      setRecovery(null)
      setMessage(null)
      setConfirmOverwrite(false)
      setConfirmServerDiscard(false)
      setLoadStatus("loading")
      return !clearStoredRecovery || !previousScope || clearRecoveryForScope(previousScope)
    },
    [clearPendingCandidate, clearRecoveryForScope, stopOwnedWork]
  )

  const disposeSensitive = React.useCallback(
    (kind: PresentationPrincipalBoundaryKind = "reauthenticate") => {
      if (kind === "reauthenticate") {
        setAuthoritySettlementReady(false)
        setAuthorityReleaseReady(false)
        quarantineActive()
        return
      }
      const cleanupScope = lastTrustedScopeRef.current
      const cleanupComplete = scrubActive(true)
      authorityCleanupScopeRef.current = cleanupComplete ? null : cleanupScope
      setAuthoritySettlementReady(false)
      setAuthorityReleaseReady(true)
    },
    [quarantineActive, scrubActive]
  )

  const principal = usePresentationPrincipalScope({ onBoundary: disposeSensitive })

  const revokeAuthorityRelease = React.useCallback(() => {
    if (
      !kindAuthorityPending ||
      kindAuthorityEpoch === null ||
      !authoritySettlementReady ||
      principal.status !== "ready" ||
      !principal.scope ||
      scopeRef.current?.principalScope !== principal.scope.principalScope
    ) {
      return
    }
    setAuthorityReleaseReady(false)
    const reported = reportedAuthoritySettlementRef.current
    if (reported?.authorityEpoch === kindAuthorityEpoch && !reported.releaseSafe) return
    reportedAuthoritySettlementRef.current = {
      authorityEpoch: kindAuthorityEpoch,
      releaseSafe: false
    }
    onKindAuthoritySettled?.(kindAuthorityEpoch, false)
  }, [
    authoritySettlementReady,
    kindAuthorityEpoch,
    kindAuthorityPending,
    onKindAuthoritySettled,
    principal.scope,
    principal.status
  ])

  React.useEffect(() => {
    if (
      !kindAuthorityPending ||
      kindAuthorityEpoch === null ||
      !authoritySettlementReady ||
      principal.status !== "ready" ||
      !principal.scope
    ) {
      return
    }
    const reported = reportedAuthoritySettlementRef.current
    if (
      reported?.authorityEpoch === kindAuthorityEpoch &&
      reported.releaseSafe === authorityReleaseReady
    ) {
      return
    }
    reportedAuthoritySettlementRef.current = {
      authorityEpoch: kindAuthorityEpoch,
      releaseSafe: authorityReleaseReady
    }
    onKindAuthoritySettled?.(kindAuthorityEpoch, authorityReleaseReady)
  }, [
    authorityReleaseReady,
    authoritySettlementReady,
    kindAuthorityEpoch,
    kindAuthorityPending,
    onKindAuthoritySettled,
    principal.scope,
    principal.status
  ])

  React.useEffect(() => {
    if (!kindAuthorityPending || principal.status !== "ready" || !principal.scope) return
    const scope = principal.scope
    const pendingCleanupScope = authorityCleanupScopeRef.current
    if (pendingCleanupScope) {
      acquireRecoveryStorage(pendingCleanupScope, "cleanup")
      if (authorityCleanupScopeRef.current) {
        setAuthoritySettlementReady(false)
        return
      }
    }
    const quarantined = quarantineRef.current
    if (quarantined && quarantined.scope.principalScope !== scope.principalScope) {
      const cleanupComplete = clearRecoveryForScope(quarantined.scope)
      authorityCleanupScopeRef.current = cleanupComplete ? null : quarantined.scope
      quarantineRef.current = null
      setAuthoritySettlementReady(cleanupComplete)
      setAuthorityReleaseReady(true)
      return
    }
    const draft = readDraftAuthority()
    let releaseSafe = true
    if (draft?.dirty) {
      releaseSafe = draft.source === draft.base.acceptedSource.source
        ? clearRecoveryForScope(draft.scope)
        : writeRecoveryFor(draft.scope, draft.base, { source: draft.source })
    }
    setAuthoritySettlementReady(true)
    setAuthorityReleaseReady(releaseSafe)
  }, [
    acquireRecoveryStorage,
    clearRecoveryForScope,
    kindAuthorityPending,
    kindAuthorityReleaseRequired,
    principal.scope,
    principal.status,
    readDraftAuthority,
    writeRecoveryFor
  ])

  const ensureOutlineController = React.useCallback(() => {
    if (!outlineControllerRef.current) {
      const epoch = operationEpochRef.current
      let controller: StandaloneHtmlOutlineController
      controller = new StandaloneHtmlOutlineController({
        onState: (state) => {
          if (
            mountedRef.current &&
            operationEpochRef.current === epoch &&
            outlineControllerRef.current === controller
          ) {
            setOutlineStatus(state.status)
          }
        },
        onOutline: (nextOutline) => {
          if (
            mountedRef.current &&
            operationEpochRef.current === epoch &&
            outlineControllerRef.current === controller
          ) {
            outlineRef.current = nextOutline
            setOutline(nextOutline)
          }
        }
      })
      outlineControllerRef.current = controller
    }
    return outlineControllerRef.current
  }, [])

  const publishAccepted = React.useCallback(
    (next: AcceptedStandaloneHtmlSource) => {
      acceptedRef.current = next
      latestPreflightCandidateRef.current = next.source
      setAccepted(next)
      ensureOutlineController().request({ source: next.source, digest: next.digest })
    },
    [ensureOutlineController]
  )

  const ensureDownloadManager = React.useCallback(() => {
    if (!downloadManagerRef.current) {
      downloadManagerRef.current = new StandaloneHtmlDownloadManager()
    }
    return downloadManagerRef.current
  }, [])

  const adoptServer = React.useCallback(
    (loaded: LoadedStandalone) => {
      clearPendingCandidate(false)
      baseRef.current = loaded
      publishAccepted(loaded.acceptedSource)
      setTitle(loaded.title)
      setSaveStatus("Saved")
      setMessage(null)
      setConfirmOverwrite(false)
      setConfirmServerDiscard(false)
      overwriteEtagRef.current = null
      const currentScope = scopeRef.current
      if (currentScope && clearRecoveryForScope(currentScope)) setRecovery(null)
    },
    [clearPendingCandidate, clearRecoveryForScope, publishAccepted]
  )

  const persistCurrentDraft = React.useCallback(
    (next: AcceptedStandaloneHtmlSource) => {
      const scope = scopeRef.current
      const base = baseRef.current
      if (!scope || !base) return
      if (next.digest === base.acceptedSource.digest) {
        if (clearRecoveryForScope(scope)) setRecovery(null)
        return
      }
      writeRecoveryFor(scope, base, next)
    },
    [clearRecoveryForScope, writeRecoveryFor]
  )

  const handleAcceptedChange = React.useCallback(
    (next: AcceptedStandaloneHtmlSource) => {
      const capabilities = slidesRef.current
      if (
        !isReadCapabilitySettled(capabilities.status) ||
        !capabilities.canReadStandalone ||
        !capabilities.canEditStandalone
      ) {
        return
      }
      revokeAuthorityRelease()
      clearPendingCandidate(false)
      publishAccepted(next)
      persistCurrentDraft(next)
      const base = baseRef.current
      setSaveStatus((currentStatus) =>
        currentStatus === "Conflict"
          ? "Conflict"
          : base && next.digest === base.acceptedSource.digest
            ? "Saved"
            : "Not saved"
      )
      setMessage(null)
    },
    [clearPendingCandidate, persistCurrentDraft, publishAccepted, revokeAuthorityRelease]
  )

  const handlePendingChange = React.useCallback(
    (candidate: string | null) => {
      if (candidate === null) {
        clearPendingCandidate(true)
        const current = acceptedRef.current
        setOutlineStatus(
          current && outlineRef.current?.digest === current.digest ? "current" : "stale"
        )
        return
      }
      const capabilities = slidesRef.current
      if (
        !isReadCapabilitySettled(capabilities.status) ||
        !capabilities.canReadStandalone ||
        !capabilities.canEditStandalone
      ) {
        clearPendingCandidate(true)
        return
      }
      pendingCandidateRef.current = candidate
      setHasPendingCandidate(true)
      setOutlineStatus("stale")
    },
    [clearPendingCandidate]
  )

  const resumePendingCandidate = React.useCallback(
    (candidate: string, scope: PresentationPrincipalScope) => {
      pendingCandidateRef.current = candidate
      const pendingCandidateEpoch = ++candidateEpochRef.current
      setHasPendingCandidate(true)
      const epoch = operationEpochRef.current
      void validateStandaloneHtmlSource(candidate).then((result) => {
        if (
          operationEpochRef.current !== epoch ||
          scopeRef.current?.principalScope !== scope.principalScope ||
          !slidesRef.current.canEditStandalone ||
          candidateEpochRef.current !== pendingCandidateEpoch ||
          pendingCandidateRef.current !== candidate ||
          latestPreflightCandidateRef.current !== candidate
        ) {
          return
        }
        if (result.ok === false) {
          clearPendingCandidate(true)
          return
        }
        publishAccepted(result)
        persistCurrentDraft(result)
        clearPendingCandidate(false)
        setSaveStatus("Not saved")
      })
    },
    [clearPendingCandidate, persistCurrentDraft, publishAccepted]
  )

  const adoptSavedBase = React.useCallback(
    (loaded: LoadedStandalone, savedCandidate: AcceptedStandaloneHtmlSource) => {
      baseRef.current = loaded
      setTitle(loaded.title)
      setMessage(null)
      setConfirmOverwrite(false)
      setConfirmServerDiscard(false)
      const current = acceptedRef.current
      const pendingCandidate = pendingCandidateRef.current
      if (current && current.digest !== savedCandidate.digest) {
        setSaveStatus("Not saved")
        setRecovery(null)
        if (pendingCandidate !== null && pendingCandidate !== savedCandidate.source) {
          const scope = scopeRef.current
          if (scope) writeRecoveryFor(scope, loaded, { source: pendingCandidate })
        } else {
          persistCurrentDraft(current)
        }
        return
      }
      publishAccepted(loaded.acceptedSource)
      if (pendingCandidate !== null && pendingCandidate !== savedCandidate.source) {
        latestPreflightCandidateRef.current = pendingCandidate
        setSaveStatus("Not saved")
        const scope = scopeRef.current
        if (scope) writeRecoveryFor(scope, loaded, { source: pendingCandidate })
        return
      }
      setSaveStatus("Saved")
      const scope = scopeRef.current
      if (scope && clearRecoveryForScope(scope)) setRecovery(null)
    },
    [clearRecoveryForScope, persistCurrentDraft, publishAccepted, writeRecoveryFor]
  )

  const operationIsCurrent = React.useCallback(
    (scope: PresentationPrincipalScope, epoch: number, controller: AbortController): boolean => {
      const capabilities = slidesRef.current
      return (
        mountedRef.current &&
        !controller.signal.aborted &&
        operationEpochRef.current === epoch &&
        scopeRef.current?.principalScope === scope.principalScope &&
        isReadCapabilitySettled(capabilities.status) &&
        capabilities.canReadStandalone &&
        (!isKindAuthorityCurrent ||
          isKindAuthorityCurrent(kindAuthorityEpoch, presentationId))
      )
    },
    [isKindAuthorityCurrent, kindAuthorityEpoch, presentationId]
  )

  const capabilityReadReady =
    isReadCapabilitySettled(slides.status) && slides.canReadStandalone

  React.useEffect(() => {
    if (!capabilityReadReady) {
      if (isReadCapabilitySettled(slides.status) && !slides.canReadStandalone) {
        if (kindAuthorityPending || quarantineRef.current) {
          quarantineActive()
        } else {
          scrubActive(false)
          setAuthorityReleaseReady(true)
        }
      } else {
        quarantineActive()
      }
      return
    }
    if (!slides.canDraftStandalone) {
      const downloadManager = downloadManagerRef.current
      downloadManagerRef.current = null
      disposeNoThrow(downloadManager)
    }
    if (!slides.canEditStandalone) {
      invalidateConflictAction()
      const saveController = saveControllerRef.current
      saveControllerRef.current = null
      abortNoThrow(saveController)
      clearPendingCandidate(true)
      const local = acceptedRef.current
      const base = baseRef.current
      if (local && base && local.digest !== base.acceptedSource.digest) setSaveStatus("Not saved")
    }
  }, [
    capabilityReadReady,
    clearPendingCandidate,
    invalidateConflictAction,
    kindAuthorityPending,
    quarantineActive,
    scrubActive,
    slides.canDraftStandalone,
    slides.canEditStandalone,
    slides.canReadStandalone,
    slides.status
  ])

  React.useEffect(() => {
    if (!online || !capabilityReadReady || principal.status !== "ready" || !principal.scope) return
    const scope = principal.scope
    const pendingCleanupScope = authorityCleanupScopeRef.current
    if (pendingCleanupScope) {
      acquireRecoveryStorage(pendingCleanupScope, "cleanup")
      if (kindAuthorityPending && authorityCleanupScopeRef.current) return
    }
    if (kindAuthorityReleaseRequired) return
    const quarantined = quarantineRef.current
    if (quarantined) {
      if (quarantined.scope.principalScope !== scope.principalScope) {
        const cleanupComplete = clearRecoveryForScope(quarantined.scope)
        authorityCleanupScopeRef.current = cleanupComplete ? null : quarantined.scope
        setAuthoritySettlementReady(cleanupComplete)
        quarantineRef.current = null
      } else {
        quarantineRef.current = null
        scopeRef.current = scope
        lastTrustedScopeRef.current = scope
        baseRef.current = quarantined.base
        publishAccepted(quarantined.accepted)
        latestPreflightCandidateRef.current = quarantined.latestPreflightCandidate
        setTitle(quarantined.title)
        setSaveStatus(quarantined.saveStatus)
        setMessage(quarantined.message)
        setRecovery(quarantined.recovery)
        setLoadStatus("ready")
        const pendingCandidate = quarantined.latestPreflightCandidate
        if (
          pendingCandidate !== quarantined.accepted.source &&
          slidesRef.current.canEditStandalone
        ) {
          resumePendingCandidate(pendingCandidate, scope)
        } else {
          clearPendingCandidate(false)
        }
        setAuthoritySettlementReady(true)
        return
      }
    }
    if (kindAuthorityPending) return
    if (
      acceptedRef.current &&
      baseRef.current &&
      scopeRef.current?.principalScope === scope.principalScope
    ) {
      const pendingCandidate = pendingCandidateRef.current
      if (
        pendingCandidate !== null &&
        pendingCandidate !== acceptedRef.current.source &&
        slidesRef.current.canEditStandalone
      ) {
        resumePendingCandidate(pendingCandidate, scope)
      } else if (pendingCandidate !== null) {
        clearPendingCandidate(false)
      }
      return
    }
    if (
      isKindAuthorityCurrent &&
      !isKindAuthorityCurrent(kindAuthorityEpoch, presentationId)
    ) {
      return
    }

    scopeRef.current = scope
    lastTrustedScopeRef.current = scope
    const previousLoad = loadControllerRef.current
    loadControllerRef.current = null
    abortNoThrow(previousLoad)
    const controller = new AbortController()
    loadControllerRef.current = controller
    const epoch = operationEpochRef.current
    let cancelled = false
    setLoadStatus("loading")
    setMessage(null)

    void (async () => {
      let detail: PresentationDetailResult | StandalonePresentationDetailResult | null = null
      try {
        if (
          isKindAuthorityCurrent &&
          !isKindAuthorityCurrent(kindAuthorityEpoch, presentationId)
        ) {
          return
        }
        detail = await tldwClient.getPresentation(presentationId, {
          abortSignal: controller.signal
        })
        if (cancelled || !operationIsCurrent(scope, epoch, controller)) return
        const loaded = await validateStandaloneDetail(detail, presentationId, scope)
        detail = null
        if (cancelled || !operationIsCurrent(scope, epoch, controller)) return

        let recovered: Awaited<ReturnType<typeof readStandaloneHtmlRecovery>> = { kind: "none" }
        if (unresolvedRecoveryCleanupRef.current.has(recoveryScopeKey(scope))) {
          clearRecoveryForScope(scope)
        } else {
          const storage = acquireRecoveryStorage(scope, "read")
          if (storage) {
            recovered = await readStandaloneHtmlRecovery(storage, scope, presentationId)
            if (recovered.kind === "unavailable") {
              markRecoveryUnavailable(scope, "read")
            } else {
              recoveryFailuresRef.current.delete(recoveryFailureKey(scope, "read"))
              publishRecoveryAvailability()
            }
          } else {
            recovered = {
              kind: "unavailable",
              code: "recovery_unavailable",
              message: RECOVERY_FAILURE_MESSAGE
            }
          }
        }
        if (cancelled || !operationIsCurrent(scope, epoch, controller)) return

        clearPendingCandidate(false)
        baseRef.current = loaded
        publishAccepted(loaded.acceptedSource)
        setTitle(loaded.title)
        setSaveStatus("Saved")
        if (recovered.kind === "available") {
          if (recovered.acceptedSource.digest === loaded.acceptedSource.digest) {
            if (clearRecoveryForScope(scope)) setRecovery(null)
            else setRecovery(recovered)
          } else {
            setRecovery(recovered)
          }
        } else {
          setRecovery(null)
        }
        setLoadStatus("ready")
      } catch {
        if (cancelled || !operationIsCurrent(scope, epoch, controller)) return
        acceptedRef.current = null
        baseRef.current = null
        latestPreflightCandidateRef.current = null
        clearPendingCandidate(false)
        setAccepted(null)
        setLoadStatus("error")
        setMessage("This standalone HTML presentation could not be loaded safely.")
      } finally {
        detail = null
      }
    })()

    return () => {
      cancelled = true
      abortNoThrow(controller)
      if (loadControllerRef.current === controller) loadControllerRef.current = null
    }
  }, [
    capabilityReadReady,
    acquireRecoveryStorage,
    clearPendingCandidate,
    clearRecoveryForScope,
    markRecoveryUnavailable,
    kindAuthorityPending,
    kindAuthorityReleaseRequired,
    kindAuthorityEpoch,
    isKindAuthorityCurrent,
    online,
    operationIsCurrent,
    persistCurrentDraft,
    presentationId,
    principal.scope,
    principal.status,
    publishAccepted,
    publishRecoveryAvailability,
    recoveryFailureKey,
    recoveryScopeKey,
    resumePendingCandidate
  ])

  const performSave = React.useCallback(
    async (
      ifMatch: string,
      reconcileAmbiguous = true,
      candidate: AcceptedStandaloneHtmlSource | null = acceptedRef.current,
      conflictAction: ConflictAction | null = null
    ) => {
      const capabilities = slidesRef.current
      const local = candidate
      const scope = scopeRef.current
      if (
        !local ||
        !scope ||
        pendingCandidateRef.current !== null ||
        !isStrongEtag(ifMatch) ||
        !isReadCapabilitySettled(capabilities.status) ||
        !capabilities.canReadStandalone ||
        !capabilities.canEditStandalone
      ) {
        if (conflictAction) finishConflictAction(conflictAction)
        return
      }
      let controller: AbortController
      if (conflictAction) {
        if (!conflictActionIsCurrent(conflictAction)) return
        controller = conflictAction.controller
      } else {
        const previousSave = saveControllerRef.current
        saveControllerRef.current = null
        abortNoThrow(previousSave)
        controller = new AbortController()
        saveControllerRef.current = controller
      }
      const epoch = operationEpochRef.current
      const saveIsCurrent = () => (
        operationIsCurrent(scope, epoch, controller) &&
        slidesRef.current.canEditStandalone &&
        (!conflictAction || conflictActionIsCurrent(conflictAction))
      )
      setSaveStatus("Saving")
      setMessage(null)
      try {
        let response: StandalonePresentationDetailResult | null = await tldwClient.saveStandaloneHtmlSource(
          presentationId,
          local.source,
          { ifMatch, abortSignal: controller.signal }
        )
        if (!saveIsCurrent()) {
          response = null
          return
        }
        let loaded: LoadedStandalone
        try {
          loaded = await validateStandaloneDetail(response, presentationId, scope)
        } finally {
          response = null
        }
        if (!saveIsCurrent()) return
        if (loaded.acceptedSource.digest !== local.digest) {
          throw new Error("Saved source could not be confirmed")
        }
        if (conflictAction && !adoptConflictAction(conflictAction)) return
        adoptSavedBase(loaded, local)
      } catch (error) {
        if (!saveIsCurrent()) return
        if (errorStatus(error) === 412) {
          overwriteEtagRef.current = null
          setConfirmOverwrite(false)
          setSaveStatus("Conflict")
          setMessage("The server version changed. Choose how to continue.")
          return
        }
        if (reconcileAmbiguous) {
          try {
            let detail: PresentationDetailResult | null = await tldwClient.getPresentation(presentationId, {
              abortSignal: controller.signal
            })
            if (!saveIsCurrent()) {
              detail = null
              return
            }
            let loaded: LoadedStandalone
            try {
              loaded = await validateStandaloneDetail(detail, presentationId, scope)
            } finally {
              detail = null
            }
            if (!saveIsCurrent()) return
            if (loaded.acceptedSource.digest === local.digest) {
              if (conflictAction && !adoptConflictAction(conflictAction)) return
              adoptSavedBase(loaded, local)
              return
            }
          } catch {
            // The local source remains authoritative until an exact server digest is observed.
          }
        }
        if (!saveIsCurrent()) return
        setSaveStatus("Not saved")
        setMessage("Save could not be confirmed. Your local draft is preserved.")
      } finally {
        if (conflictAction) finishConflictAction(conflictAction)
        else if (saveControllerRef.current === controller) saveControllerRef.current = null
      }
    },
    [
      adoptConflictAction,
      adoptSavedBase,
      conflictActionIsCurrent,
      finishConflictAction,
      operationIsCurrent,
      presentationId
    ]
  )

  const handleSave = React.useCallback(() => {
    const capabilities = slidesRef.current
    const base = baseRef.current
    if (
      base &&
      pendingCandidateRef.current === null &&
      isReadCapabilitySettled(capabilities.status) &&
      capabilities.canReadStandalone &&
      capabilities.canEditStandalone
    ) {
      void performSave(base.etag)
    }
  }, [performSave])

  const handlePrepareOverwrite = React.useCallback(async () => {
    const capabilities = slidesRef.current
    const scope = scopeRef.current
    if (
      !scope ||
      pendingCandidateRef.current !== null ||
      !isReadCapabilitySettled(capabilities.status) ||
      !capabilities.canReadStandalone ||
      !capabilities.canEditStandalone
    ) {
      return
    }
    setConfirmOverwrite(false)
    overwriteEtagRef.current = null
    const action = beginConflictAction()
    const controller = action.controller
    const epoch = operationEpochRef.current
    const prepareIsCurrent = () => (
      operationIsCurrent(scope, epoch, controller) &&
      conflictActionIsCurrent(action) &&
      slidesRef.current.canEditStandalone &&
      pendingCandidateRef.current === null
    )
    try {
      let fresh: PresentationDetailResult | null = await tldwClient.getPresentation(presentationId, {
        abortSignal: controller.signal
      })
      if (!prepareIsCurrent()) {
        fresh = null
        return
      }
      let loaded: LoadedStandalone
      try {
        loaded = await validateStandaloneDetail(fresh, presentationId, scope)
      } finally {
        fresh = null
      }
      if (!prepareIsCurrent()) return
      overwriteEtagRef.current = loaded.etag
      setSaveStatus("Conflict")
      setConfirmOverwrite(true)
      setMessage("The current server version was verified. Confirm before overwriting it.")
    } catch {
      if (!prepareIsCurrent()) return
      setSaveStatus("Conflict")
      setMessage("The current server version could not be verified. Your draft is preserved.")
    } finally {
      finishConflictAction(action)
    }
  }, [beginConflictAction, conflictActionIsCurrent, finishConflictAction, operationIsCurrent, presentationId])

  const handleOverwrite = React.useCallback(() => {
    const capabilities = slidesRef.current
    const candidate = acceptedRef.current
    const freshEtag = overwriteEtagRef.current
    if (
      !candidate ||
      pendingCandidateRef.current !== null ||
      !isStrongEtag(freshEtag) ||
      !isReadCapabilitySettled(capabilities.status) ||
      !capabilities.canReadStandalone ||
      !capabilities.canEditStandalone
    ) {
      return
    }
    overwriteEtagRef.current = null
    setConfirmOverwrite(false)
    const action = beginConflictAction()
    void performSave(freshEtag, true, candidate, action)
  }, [beginConflictAction, performSave])

  const handleDiscardAndLoad = React.useCallback(async () => {
    const capabilities = slidesRef.current
    const scope = scopeRef.current
    const confirmedDigest = acceptedRef.current?.digest
    const confirmedCandidate = latestPreflightCandidateRef.current
    const confirmedCandidateEpoch = candidateEpochRef.current
    if (
      !scope ||
      !confirmedDigest ||
      pendingCandidateRef.current !== null ||
      !isReadCapabilitySettled(capabilities.status) ||
      !capabilities.canReadStandalone ||
      !capabilities.canEditStandalone
    ) {
      return
    }
    setConfirmServerDiscard(false)
    const action = beginConflictAction()
    const controller = action.controller
    const epoch = operationEpochRef.current
    const discardIsCurrent = () => (
      operationIsCurrent(scope, epoch, controller) &&
      conflictActionIsCurrent(action) &&
      slidesRef.current.canEditStandalone
    )
    try {
      let fresh: PresentationDetailResult | null = await tldwClient.getPresentation(presentationId, {
        abortSignal: controller.signal
      })
      if (!discardIsCurrent()) {
        fresh = null
        return
      }
      let loaded: LoadedStandalone
      try {
        loaded = await validateStandaloneDetail(fresh, presentationId, scope)
      } finally {
        fresh = null
      }
      if (!discardIsCurrent()) return
      if (acceptedRef.current?.digest !== confirmedDigest) {
        setSaveStatus("Conflict")
        setMessage("Your draft changed while the server version was loading. Your newer draft is preserved.")
        return
      }
      if (
        candidateEpochRef.current !== confirmedCandidateEpoch ||
        latestPreflightCandidateRef.current !== confirmedCandidate
      ) {
        setSaveStatus("Conflict")
        setMessage("Your draft changed while the server version was loading. Your newer draft is preserved.")
        return
      }
      if (!adoptConflictAction(action)) return
      adoptServer(loaded)
    } catch {
      if (!discardIsCurrent()) return
      setSaveStatus("Conflict")
      setMessage("The server version could not be loaded. Your draft is preserved.")
    } finally {
      finishConflictAction(action)
    }
  }, [
    adoptConflictAction,
    adoptServer,
    beginConflictAction,
    conflictActionIsCurrent,
    finishConflictAction,
    operationIsCurrent,
    presentationId
  ])

  const downloadSource = React.useCallback(
    async (source: string) => {
      const capabilities = slidesRef.current
      const scope = scopeRef.current
      if (
        !scope ||
        pendingCandidateRef.current !== null ||
        !isReadCapabilitySettled(capabilities.status) ||
        !capabilities.canReadStandalone ||
        !capabilities.canDraftStandalone
      ) {
        return
      }
      const epoch = operationEpochRef.current
      try {
        await ensureDownloadManager().download({ presentationId, source })
        if (
          operationEpochRef.current !== epoch ||
          scopeRef.current?.principalScope !== scope.principalScope ||
          !slidesRef.current.canDraftStandalone
        ) {
          return
        }
      } catch {
        if (
          operationEpochRef.current !== epoch ||
          scopeRef.current?.principalScope !== scope.principalScope ||
          !slidesRef.current.canDraftStandalone
        ) {
          return
        }
        setMessage("Download could not be prepared. Your draft is preserved.")
      }
    },
    [ensureDownloadManager, presentationId]
  )

  const discardRecovery = React.useCallback(() => {
    const scope = scopeRef.current
    if (!scope) return
    if (clearRecoveryForScope(scope)) {
      setRecovery(null)
      setConfirmRecoveryDiscard(false)
    }
  }, [clearRecoveryForScope])

  React.useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!readDraftAuthority()?.dirty) return
      event.preventDefault()
      event.returnValue = ""
    }
    const pagehide = () => {
      try {
        flushDraftAuthority()
      } finally {
        scrubActive(false)
      }
    }
    window.addEventListener("beforeunload", beforeUnload)
    window.addEventListener("pagehide", pagehide)
    return () => {
      window.removeEventListener("beforeunload", beforeUnload)
      window.removeEventListener("pagehide", pagehide)
    }
  }, [flushDraftAuthority, readDraftAuthority, scrubActive])

  React.useEffect(() => {
    mountedRef.current = true
    return () => {
      try {
        flushDraftAuthority()
      } finally {
        mountedRef.current = false
        workspaceStateRef.current = {
          title: "Standalone HTML presentation",
          saveStatus: "Saved",
          message: null,
          recovery: null
        }
        stopOwnedWork()
        quarantineRef.current = null
        acceptedRef.current = null
        baseRef.current = null
        scopeRef.current = null
        lastTrustedScopeRef.current = null
        latestPreflightCandidateRef.current = null
        pendingCandidateRef.current = null
      }
    }
  }, [flushDraftAuthority, stopOwnedWork])

  const renderWorkspaceShell = (content: React.ReactNode) => (
    <>
      <RouteLeavePrompt
        when={dirty && !leaveApproved}
        message="Leave without saving? Your local draft is preserved only in this tab."
      />
      {recoveryWarning ? (
        <p role="alert" className="mb-4 text-sm text-warning">
          {recoveryWarning}
        </p>
      ) : null}
      {content}
    </>
  )

  if (!online) {
    return renderWorkspaceShell(
      <section className="rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-text-muted">Server is offline. Your in-memory draft has not been sent.</p>
      </section>
    )
  }

  if (
    principal.status === "ready" &&
    kindAuthorityPending &&
    kindAuthorityReleaseRequired
  ) {
    return renderWorkspaceShell(
      <section className="rounded-xl border border-border bg-surface p-6" aria-live="polite">
        <p className="text-sm text-text-muted">Confirming current server and account…</p>
      </section>
    )
  }

  const capabilityGuardText = !capabilityReadReady
    ? slides.status === "loading"
      ? "Checking standalone HTML access…"
      : slides.status === "auth_required"
        ? "Current standalone HTML access requires authentication."
        : slides.status === "forbidden"
          ? "This account cannot read standalone HTML presentations."
          : isReadCapabilitySettled(slides.status) && !slides.canReadStandalone
            ? "This server does not support reading standalone HTML presentations."
            : "Standalone HTML access could not be confirmed."
    : null

  if (capabilityGuardText) {
    return renderWorkspaceShell(
      <section className="rounded-xl border border-border bg-surface p-6" aria-live="polite">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-danger">{capabilityGuardText}</p>
        <Button size="lg" variant="secondary" onClick={() => void slides.retry()} className="mt-4">
          Retry
        </Button>
      </section>
    )
  }

  if (principal.status === "guarded") {
    return renderWorkspaceShell(
      <section className="rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-danger">Current server and account could not be confirmed.</p>
        <Button size="lg" variant="secondary" onClick={() => void principal.retry()} className="mt-4">
          Retry
        </Button>
      </section>
    )
  }

  if (principal.status === "loading" || loadStatus === "loading") {
    return renderWorkspaceShell(
      <section className="rounded-xl border border-border bg-surface p-6" aria-live="polite">
        <p className="text-sm text-text-muted">Confirming current server and account…</p>
      </section>
    )
  }

  if (loadStatus === "error" || !accepted) {
    return renderWorkspaceShell(
      <section className="rounded-xl border border-border bg-surface p-6">
        <h1 className="text-2xl font-semibold text-text">Standalone HTML presentation</h1>
        <p className="mt-2 text-sm text-danger">{message ?? "Presentation unavailable."}</p>
      </section>
    )
  }

  const canSave =
    capabilityReadReady &&
    slides.canEditStandalone &&
    !hasPendingCandidate &&
    dirty &&
    saveStatus !== "Saving" &&
    saveStatus !== "Conflict" &&
    Boolean(baseRef.current)
  const canDownload =
    capabilityReadReady && slides.canDraftStandalone && !hasPendingCandidate

  return renderWorkspaceShell(
    <section className="space-y-4">
      <header className="rounded-xl border border-border bg-surface p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">Standalone HTML</p>
            <h1 className="mt-1 text-2xl font-semibold text-text">{title}</h1>
            <p className="mt-2 max-w-2xl text-sm text-text-muted">
              Studio never runs this code. Downloading and opening the file leaves tldw&apos;s security boundary.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="lg" variant="ghost" onClick={() => {
              if (dirty) setConfirmLeave(true)
              else navigate("/presentation-studio")
            }}>
              Back to presentations
            </Button>
            <Button
              size="lg"
              variant="secondary"
              disabled={!canDownload}
              onClick={() => void downloadSource(accepted.source)}
            >
              Download current draft
            </Button>
            <Button size="lg" variant="primary" disabled={!canSave} onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
        <div
          data-testid="standalone-html-save-status"
          role="status"
          aria-live="polite"
          className="mt-3 text-sm font-medium text-text"
        >
          {saveStatus}
        </div>
        {!slides.canEditStandalone ? (
          <p className="mt-2 text-sm text-warning">Saving is unavailable</p>
        ) : null}
        {message ? <p role="alert" className="mt-2 text-sm text-danger">{message}</p> : null}
      </header>

      {recovery ? (
        <section aria-label="Recovered draft" role="region" className="rounded-xl border border-warning/40 bg-warning/10 p-4">
          <h2 className="font-semibold text-text">Recovered draft</h2>
          <p className="mt-1 text-sm text-text-muted">A different draft was saved in this tab. It has not been applied.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              size="lg"
              variant="secondary"
              disabled={!slides.canEditStandalone || hasPendingCandidate}
              onClick={() => handleAcceptedChange(recovery.acceptedSource)}
            >
              Restore recovered draft
            </Button>
            <Button
              size="lg"
              variant="secondary"
              disabled={!canDownload}
              onClick={() => void downloadSource(recovery.acceptedSource.source)}
            >
              Download recovered draft
            </Button>
            <Button size="lg" variant="danger" onClick={() => setConfirmRecoveryDiscard(true)}>
              Discard recovered draft
            </Button>
          </div>
          {confirmRecoveryDiscard ? (
            <div className="mt-3 rounded-lg border border-danger/30 bg-surface p-3">
              <p className="text-sm text-text">Confirm discarding the recovered draft. This cannot be undone.</p>
              <Button size="lg" variant="danger" className="mt-2" onClick={discardRecovery}>
                Confirm discard recovered draft
              </Button>
            </div>
          ) : null}
        </section>
      ) : null}

      {saveStatus === "Conflict" ? (
        <section className="rounded-xl border border-warning/40 bg-warning/10 p-4" aria-label="Save conflict">
          <h2 className="font-semibold text-text">Conflict</h2>
          <p className="mt-1 text-sm text-text-muted">Your draft is unchanged. Choose an explicit next step.</p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              size="lg"
              variant="danger"
              disabled={!slides.canEditStandalone || hasPendingCandidate}
              onClick={() => setConfirmServerDiscard(true)}
            >
              Discard my changes and load server version
            </Button>
            <Button
              size="lg"
              variant="primary"
              disabled={!slides.canEditStandalone || hasPendingCandidate}
              onClick={() => void handlePrepareOverwrite()}
            >
              Overwrite server with my draft
            </Button>
            <Button
              size="lg"
              variant="secondary"
              disabled={!canDownload}
              onClick={() => void downloadSource(accepted.source)}
            >
              Download my draft
            </Button>
          </div>
          {confirmOverwrite ? (
            <div className="mt-3 rounded-lg border border-warning/40 bg-surface p-3">
              <p className="text-sm text-text">Confirm replacing the current server version with your local draft.</p>
              <Button
                size="lg"
                variant="danger"
                className="mt-2"
                disabled={!slides.canEditStandalone || hasPendingCandidate}
                onClick={handleOverwrite}
              >
                Confirm overwrite
              </Button>
            </div>
          ) : null}
          {confirmServerDiscard ? (
            <div className="mt-3 rounded-lg border border-danger/30 bg-surface p-3">
              <p className="text-sm text-text">Confirm discarding your local changes and loading the server version.</p>
              <Button
                size="lg"
                variant="danger"
                className="mt-2"
                disabled={!slides.canEditStandalone || hasPendingCandidate}
                onClick={() => void handleDiscardAndLoad()}
              >
                Confirm discard and load server version
              </Button>
            </div>
          ) : null}
        </section>
      ) : null}

      {confirmLeave ? (
        <section className="rounded-xl border border-danger/30 bg-surface p-4">
          <h2 className="font-semibold text-text">Leave without saving?</h2>
          <p className="mt-1 text-sm text-text-muted">Your in-memory draft will close. Scoped recovery may still be available in this tab.</p>
          <div className="mt-3 flex gap-2">
            <Button size="lg" variant="danger" onClick={() => setLeaveApproved(true)}>
              Leave presentation
            </Button>
            <Button size="lg" variant="secondary" onClick={() => setConfirmLeave(false)}>
              Keep editing
            </Button>
          </div>
        </section>
      ) : null}

      <div role="tablist" aria-label="Standalone HTML workspace views" className="flex gap-2 md:hidden">
        <button
          ref={codeTabRef}
          id={codeTabId}
          type="button"
          role="tab"
          aria-controls={codePanelId}
          aria-selected={activeTab === "code"}
          tabIndex={activeTab === "code" ? 0 : -1}
          onClick={() => setActiveTab("code")}
          onKeyDown={handleTabKeyDown}
          className="min-h-[44px] rounded-md px-4 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          Code
        </button>
        <button
          ref={outlineTabRef}
          id={outlineTabId}
          type="button"
          role="tab"
          aria-controls={outlinePanelId}
          aria-selected={activeTab === "outline"}
          tabIndex={activeTab === "outline" ? 0 : -1}
          onClick={() => setActiveTab("outline")}
          onKeyDown={handleTabKeyDown}
          className="min-h-[44px] rounded-md px-4 text-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          Outline
        </button>
      </div>

      <div className="grid min-w-0 gap-4 md:grid-cols-2">
        <section
          id={codePanelId}
          role="tabpanel"
          aria-labelledby={codeTabId}
          className={`${activeTab === "code" ? "block" : "hidden"} min-w-0 rounded-xl border border-border bg-surface p-4 md:block`}
        >
          <StandaloneHtmlSourceEditor
            ref={sourceEditorRef}
            value={accepted.source}
            draftSeed={hasPendingCandidate ? pendingCandidateRef.current : null}
            onAcceptedChange={handleAcceptedChange}
            onPendingChange={handlePendingChange}
            onPreflightCandidate={(candidate) => {
              const capabilities = slidesRef.current
              if (
                isReadCapabilitySettled(capabilities.status) &&
                capabilities.canReadStandalone &&
                capabilities.canEditStandalone
              ) {
                latestPreflightCandidateRef.current = candidate
                candidateEpochRef.current += 1
                revokeAuthorityRelease()
              }
            }}
            readOnly={!slides.canEditStandalone}
          />
        </section>
        <section
          id={outlinePanelId}
          role="tabpanel"
          aria-labelledby={outlineTabId}
          className={`${activeTab === "outline" ? "block" : "hidden"} min-w-0 rounded-xl border border-border bg-surface p-4 md:block`}
        >
          <StandaloneHtmlSafeOutline status={outlineStatus} outline={outline} />
        </section>
      </div>
    </section>
  )
}

export default StandaloneHtmlWorkspace
