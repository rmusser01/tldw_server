import React from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Button, Form, Input, Skeleton } from "antd"
import { useTranslation } from "react-i18next"
import { Link, useSearchParams } from "react-router-dom"

import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { Alert, Badge } from "@/components/ui/primitives"
import { RecoveryCallout } from "@/components/ui/state"
import {
  clearLegacyServicePromptCandidate,
  importLegacyServicePromptCandidate,
  isServicePromptScopeUnresolvedError,
  readLegacyServicePromptCandidates,
  renderServicePromptPart,
  resolveServicePromptScope,
  subscribeToServicePromptConfigChanges,
  validateServicePromptParts,
  type LegacyServicePromptCandidate,
  type ServicePromptScope
} from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  ServicePromptApiError,
  type KnownServicePromptId,
  type ServicePromptCatalogItem,
  type ServicePromptDetail
} from "@/services/tldw/domains/service-prompts"
import {
  resolveSettingsNavigationUrl,
  SETTINGS_NAVIGATION_REQUEST_EVENT,
  type SettingsNavigationRequestDetail
} from "@/utils/settings-return"

type Draft = {
  scopeKey: string
  definitionId: string
  parts: Record<string, string>
  revision: string | null
}

type PendingHistoryRestore = {
  conflict: boolean
  definitionId: string
  draft: Draft
  fieldErrors: Record<string, string>
  focusId: string | null
  operationError: string | null
  preview: Record<string, string> | null
  scopeKey: string
  url: string
}

let pendingHistoryRestore: PendingHistoryRestore | null = null
let pendingHistoryRestoreTimer: ReturnType<typeof setTimeout> | null = null
let historyEntryTokenSequence = 0

const createHistoryEntryToken = (): string => {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID()
  }
  historyEntryTokenSequence += 1
  return `service-prompt-${Date.now().toString(36)}-${historyEntryTokenSequence.toString(36)}`
}

const clearPendingHistoryRestore = () => {
  if (pendingHistoryRestoreTimer !== null) {
    clearTimeout(pendingHistoryRestoreTimer)
    pendingHistoryRestoreTimer = null
  }
  pendingHistoryRestore = null
}

const stagePendingHistoryRestore = (restore: PendingHistoryRestore) => {
  clearPendingHistoryRestore()
  pendingHistoryRestore = restore
  pendingHistoryRestoreTimer = setTimeout(clearPendingHistoryRestore, 2_000)
}

const claimPendingHistoryRestore = (
  url: string,
  definitionId: string
): PendingHistoryRestore | null => {
  const restore = pendingHistoryRestore
  clearPendingHistoryRestore()
  return restore?.url === url && restore.definitionId === definitionId
    ? restore
    : null
}

type MigrationItem = LegacyServicePromptCandidate & {
  error?: string
}

type OperationKind =
  | "save"
  | "reset"
  | "reload"
  | "migration-import"
  | "migration-discard"

type ActiveOperation = {
  controller: AbortController
  definitionId?: string
  identity: number
  kind: OperationKind
  scopeKey: string
}

const KNOWN_DEFINITIONS = {
  "chat.rag.answer": {
    key: "chatRagAnswer",
    label: "RAG answer",
    description:
      "Controls how retrieved context and the current question are presented to the model."
  },
  "chat.rag.question_rewrite": {
    key: "chatRagQuestionRewrite",
    label: "RAG follow-up rewrite",
    description:
      "Controls how a conversational follow-up is rewritten into a standalone retrieval query."
  },
  "chat.web_search.answer": {
    key: "chatWebSearchAnswer",
    label: "Web-search answer",
    description:
      "Controls how normalized web-search results are presented for the final answer."
  },
  "chat.title.generation": {
    key: "chatTitleGeneration",
    label: "Conversation title",
    description:
      "Controls the instruction used to generate automatic conversation titles."
  },
  "image.prompt.refinement": {
    key: "imagePromptRefinement",
    label: "Image prompt refinement",
    description:
      "Controls the semantic instructions used to refine image-generation prompt drafts."
  },
  "media.text.translation": {
    key: "mediaTextTranslation",
    label: "Text translation",
    description:
      "Controls the visible instructions used by synchronous text translation."
  },
  "media.document.summarization": {
    key: "mediaDocumentSummarization",
    label: "Document summarization",
    description:
      "Controls system instructions for synchronous document analysis. Without a saved override, server defaults apply."
  },
  "media.pdf.summarization": {
    key: "mediaPdfSummarization",
    label: "PDF summarization",
    description:
      "Controls system instructions for synchronous PDF analysis. Without a saved override, server defaults apply."
  },
  "media.ebook.summarization": {
    key: "mediaEbookSummarization",
    label: "EPUB summarization",
    description:
      "Controls system instructions for synchronous EPUB analysis. Without a saved override, server defaults apply."
  },
  "media.video.summarization": {
    key: "mediaVideoSummarization",
    label: "Video summarization",
    description:
      "Controls system instructions and recursive final-summary instructions for synchronous video analysis. Without a saved override, server defaults apply."
  },
  "media.audio.analysis": {
    key: "mediaAudioAnalysis",
    label: "Audio summarization",
    description:
      "Controls system and user instructions for synchronous audio analysis. Without a saved override, server defaults apply."
  },
  "media.web.summarization": {
    key: "mediaWebSummarization",
    label: "Web article summarization",
    description:
      "Controls summary instructions for synchronous web scraping and web-content ingestion. Reset restores each scraping engine's existing defaults; the displayed defaults are the deployed web-article prompts."
  },
  "media.email.summarization": {
    key: "mediaEmailSummarization",
    label: "Email summarization",
    description:
      "Controls system instructions for synchronous email analysis. Without a saved override, server defaults apply."
  },
  "notes.title.generate": {
    key: "notesTitleGenerate",
    label: "Notes title",
    description:
      "Controls the wording used by LLM-backed automatic Notes titles."
  }
} as const

const KNOWN_WORKFLOWS: Record<string, { key: string; label: string }> = {
  "chat.main.rag": { key: "mainChatRag", label: "Main chat RAG" },
  "chat.tab.rag": { key: "tabChatRag", label: "Tab chat RAG" },
  "chat.document.rag": {
    key: "documentChatRag",
    label: "Document chat RAG"
  },
  "chat.sidepanel.rag": { key: "sidepanelRag", label: "Sidepanel RAG" },
  "chat.main.web_search": {
    key: "mainChatWebSearch",
    label: "Main chat web search"
  },
  "chat.compare.web_search": {
    key: "compareWebSearch",
    label: "Compare web search"
  },
  "chat.title.generation": {
    key: "automaticConversationTitles",
    label: "Automatic conversation titles"
  },
  "image.prompt.refinement": {
    key: "imagePromptRefinement",
    label: "Image prompt refinement"
  },
  "media.text.translation": {
    key: "textTranslation",
    label: "Text translation"
  },
  "media.document.summarization": {
    key: "documentSummarization",
    label: "Synchronous document analysis"
  },
  "media.pdf.summarization": {
    key: "pdfSummarization",
    label: "Synchronous PDF analysis"
  },
  "media.ebook.summarization": {
    key: "ebookSummarization",
    label: "Synchronous EPUB analysis"
  },
  "media.email.summarization": {
    key: "emailSummarization",
    label: "Synchronous email analysis"
  },
  "media.audio.analysis": {
    key: "audioAnalysis",
    label: "Synchronous audio analysis"
  },
  "media.video.summarization": {
    key: "videoSummarization",
    label: "Synchronous video analysis"
  },
  "media.web.summarization": {
    key: "webSummarization",
    label: "Synchronous web scraping and ingestion"
  },
  "notes.title.generate": {
    key: "automaticNotesTitles",
    label: "Automatic Notes titles"
  }
}

const KNOWN_PARTS: Record<string, { key: string; label: string }> = {
  template: { key: "template", label: "Template" },
  system: { key: "system", label: "System instructions" },
  user: { key: "user", label: "User instructions" },
  final_summary: { key: "finalSummary", label: "Final-summary instructions" },
  system_semantics: {
    key: "systemSemantics",
    label: "Refinement guidance"
  },
  rewrite_semantics: {
    key: "rewriteSemantics",
    label: "Rewrite guidance"
  },
  title_instruction: { key: "titleInstruction", label: "Title instruction" },
  user_template: { key: "userTemplate", label: "User template" }
}

const isKnownDefinitionId = (id: string): id is KnownServicePromptId =>
  Object.prototype.hasOwnProperty.call(KNOWN_DEFINITIONS, id)

const isAbortError = (error: unknown): boolean =>
  (error as { name?: unknown } | null)?.name === "AbortError"

const getDefinitionText = (
  definition: ServicePromptCatalogItem,
  field: "label" | "description",
  t: ReturnType<typeof useTranslation>["t"]
): string => {
  if (!isKnownDefinitionId(definition.id)) return definition[field]
  const known = KNOWN_DEFINITIONS[definition.id]
  return t(`servicePrompts.definitions.${known.key}.${field}`, {
    defaultValue: known[field]
  })
}

const getWorkflowLabel = (
  id: string,
  fallback: string,
  t: ReturnType<typeof useTranslation>["t"]
): string => {
  const known = KNOWN_WORKFLOWS[id]
  return known
    ? t(`servicePrompts.workflows.${known.key}`, { defaultValue: known.label })
    : fallback
}

const getPartLabel = (
  key: string,
  fallback: string,
  t: ReturnType<typeof useTranslation>["t"]
): string => {
  const known = KNOWN_PARTS[key]
  return known
    ? t(`servicePrompts.parts.${known.key}`, { defaultValue: known.label })
    : fallback
}

const getPromptSearchId = (searchParams: URLSearchParams): string | null => {
  const value = searchParams.get("prompt")?.trim()
  return value || null
}

const getHistoryIndex = (state: unknown): number | null => {
  if (!state || typeof state !== "object") return null
  const historyState = state as {
    idx?: unknown
    servicePromptHistoryIndex?: unknown
  }
  if (typeof historyState.servicePromptHistoryIndex === "number") {
    return historyState.servicePromptHistoryIndex
  }
  return typeof historyState.idx === "number" ? historyState.idx : null
}

const getHistoryEntryToken = (state: unknown): string | null => {
  if (!state || typeof state !== "object") return null
  const token = (state as {
    servicePromptHistoryEntryToken?: unknown
  }).servicePromptHistoryEntryToken
  return typeof token === "string" && token ? token : null
}

const getHistoryForwardEntryToken = (state: unknown): string | null => {
  if (!state || typeof state !== "object") return null
  const token = (state as {
    servicePromptHistoryForwardEntryToken?: unknown
  }).servicePromptHistoryForwardEntryToken
  return typeof token === "string" && token ? token : null
}

const toDomId = (value: string): string =>
  value.replace(/[^a-zA-Z0-9_-]+/g, "-")

export const ServicePromptsSettings = () => {
  const { t } = useTranslation("settings")
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()
  const [searchParams, setSearchParams] = useSearchParams()
  const selectedId = getPromptSearchId(searchParams)

  const [scope, setScope] = React.useState<ServicePromptScope | null>(null)
  const [scopeError, setScopeError] = React.useState<unknown>(null)
  const [scopeLoading, setScopeLoading] = React.useState(true)
  const [scopeGeneration, setScopeGeneration] = React.useState(0)
  const [scopeChanged, setScopeChanged] = React.useState(false)
  const [scopeVerified, setScopeVerified] = React.useState(false)
  const [scopeVerificationError, setScopeVerificationError] =
    React.useState(false)
  const [draft, setDraft] = React.useState<Draft | null>(null)
  const [dirty, setDirty] = React.useState(false)
  const [fieldErrors, setFieldErrors] = React.useState<Record<string, string>>({})
  const [preview, setPreview] = React.useState<Record<string, string> | null>(null)
  const [conflict, setConflict] = React.useState(false)
  const [operationError, setOperationError] = React.useState<string | null>(null)
  const [operationAnnouncement, setOperationAnnouncement] = React.useState("")
  const [activeOperation, setActiveOperation] =
    React.useState<ActiveOperation | null>(null)
  const [migrationItems, setMigrationItems] = React.useState<MigrationItem[]>([])
  const [migrationError, setMigrationError] = React.useState<string | null>(null)
  const [migrationMessage, setMigrationMessage] = React.useState<string | null>(null)
  const [migrationProbeError, setMigrationProbeError] = React.useState<string | null>(null)
  const [migrationProbeGeneration, setMigrationProbeGeneration] = React.useState(0)

  const scopeRef = React.useRef<ServicePromptScope | null>(null)
  const scopeVerifiedRef = React.useRef(false)
  const selectedIdRef = React.useRef<string | null>(selectedId)
  const dirtyRef = React.useRef(false)
  const historyIndexRef = React.useRef(0)
  const historyInitializedRef = React.useRef(false)
  const historyForwardEntryTokenRef = React.useRef<string | null>(null)
  const pendingHistoryDestinationRef = React.useRef<{
    token: string
    url: string
  } | null>(null)
  const historyUrlRef = React.useRef(
    typeof window === "undefined" ? "" : window.location.href
  )
  const historyRestoreCheckedRef = React.useRef(false)
  const claimedHistoryRestoreRef = React.useRef<PendingHistoryRestore | null>(null)
  const historyFocusRef = React.useRef<{
    element: HTMLElement | null
    id: string | null
  } | null>(null)
  const suppressPopstateRef = React.useRef(false)
  const historyRestoreStateRef = React.useRef({
    conflict,
    draft,
    fieldErrors,
    operationError,
    preview
  })
  const activeOperationRef = React.useRef<ActiveOperation | null>(null)
  const operationIdentityRef = React.useRef(0)
  const migrationProbedScopeRef = React.useRef<string | null>(null)
  const detailFocusRef = React.useRef<HTMLElement | null>(null)
  const pendingFocusDefinitionRef = React.useRef<string | null>(null)
  const scopeReconcileControllerRef = React.useRef<AbortController | null>(null)

  const abortActiveOperation = React.useCallback(() => {
    activeOperationRef.current?.controller.abort()
    activeOperationRef.current = null
    setActiveOperation(null)
    setOperationAnnouncement("")
  }, [])

  const startOperation = React.useCallback((
    kind: OperationKind,
    scopeKey: string,
    definitionId?: string
  ): ActiveOperation => {
    const operation = {
      controller: new AbortController(),
      definitionId,
      identity: operationIdentityRef.current + 1,
      kind,
      scopeKey
    }
    operationIdentityRef.current = operation.identity
    activeOperationRef.current = operation
    setActiveOperation(operation)
    return operation
  }, [])

  const isCurrentOperation = React.useCallback((operation: ActiveOperation) =>
    !operation.controller.signal.aborted &&
    scopeVerifiedRef.current &&
    activeOperationRef.current?.identity === operation.identity &&
    scopeRef.current?.scopeKey === operation.scopeKey &&
    (!operation.definitionId || selectedIdRef.current === operation.definitionId), [])

  const finishOperation = React.useCallback((operation: ActiveOperation) => {
    if (activeOperationRef.current?.identity === operation.identity) {
      activeOperationRef.current = null
    }
    if (operation.controller.signal.aborted) return
    setActiveOperation((current) =>
      current?.identity === operation.identity ? null : current
    )
  }, [])

  const activeKind = activeOperation?.kind ?? null
  const isSaving = activeKind === "save"
  const isResetting = activeKind === "reset"

  const replaceHistoryForwardEntryToken = React.useCallback(
    (token: string | null) => {
      const existing = window.history.state ?? {}
      const next = { ...existing }
      delete next.servicePromptHistoryForwardDestination
      if (token === null) {
        delete next.servicePromptHistoryForwardEntryToken
      } else {
        next.servicePromptHistoryForwardEntryToken = token
      }
      window.history.replaceState(next, "", window.location.href)
      historyForwardEntryTokenRef.current = token
    },
    []
  )

  const stampPendingHistoryDestination = React.useCallback(() => {
    const pending = pendingHistoryDestinationRef.current
    if (!pending || window.location.href !== pending.url) return
    const next = { ...(window.history.state ?? {}) }
    delete next.servicePromptHistoryForwardDestination
    delete next.servicePromptHistoryForwardEntryToken
    next.servicePromptHistoryEntryToken = pending.token
    window.history.replaceState(next, "", window.location.href)
    pendingHistoryDestinationRef.current = null
  }, [])

  const prepareHistoryNavigation = React.useCallback((destination: string) => {
    if (dirtyRef.current) {
      const leave = window.confirm(t("servicePrompts.unsaved.leave", {
        defaultValue: "Discard unsaved workflow prompt changes?"
      }))
      if (!leave) return false
      dirtyRef.current = false
      setDirty(false)
    }
    clearPendingHistoryRestore()
    claimedHistoryRestoreRef.current = null
    const token = createHistoryEntryToken()
    pendingHistoryDestinationRef.current = { token, url: destination }
    replaceHistoryForwardEntryToken(token)
    queueMicrotask(stampPendingHistoryDestination)
    return true
  }, [replaceHistoryForwardEntryToken, stampPendingHistoryDestination, t])

  React.useEffect(() => {
    scopeRef.current = scope
  }, [scope])

  React.useEffect(() => {
    selectedIdRef.current = selectedId
  }, [selectedId])

  React.useEffect(() => {
    dirtyRef.current = dirty
  }, [dirty])

  React.useEffect(() => {
    historyRestoreStateRef.current = {
      conflict,
      draft,
      fieldErrors,
      operationError,
      preview
    }
  }, [conflict, draft, fieldErrors, operationError, preview])

  React.useEffect(() => {
    if (historyRestoreCheckedRef.current || !selectedId) return
    historyRestoreCheckedRef.current = true
    const restore = claimPendingHistoryRestore(
      window.location.href,
      selectedId
    )
    claimedHistoryRestoreRef.current = restore
    if (restore) dirtyRef.current = true
  }, [selectedId])

  React.useEffect(() => {
    const restore = claimedHistoryRestoreRef.current
    if (!restore || !selectedId || !scope) return
    claimedHistoryRestoreRef.current = null
    if (restore.url !== window.location.href ||
      restore.definitionId !== selectedId || restore.scopeKey !== scope.scopeKey) {
      dirtyRef.current = false
      setDirty(false)
      return
    }
    historyFocusRef.current = { element: null, id: restore.focusId }
    dirtyRef.current = true
    setDraft({ ...restore.draft, parts: { ...restore.draft.parts } })
    setDirty(true)
    setFieldErrors({ ...restore.fieldErrors })
    setPreview(restore.preview ? { ...restore.preview } : null)
    setConflict(restore.conflict)
    setOperationError(restore.operationError)
  }, [scope, selectedId])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    const existing = window.history.state ?? {}
    if (!historyInitializedRef.current) {
      const existingIndex = getHistoryIndex(existing)
      if (existingIndex !== null) historyIndexRef.current = existingIndex
      historyInitializedRef.current = true
    }
    historyUrlRef.current = window.location.href
    historyForwardEntryTokenRef.current = getHistoryForwardEntryToken(existing)
    const next = {
      ...existing,
      servicePromptHistoryIndex: historyIndexRef.current
    }
    delete next.servicePromptHistoryForwardDestination
    window.history.replaceState(next, "", window.location.href)
  }, [selectedId])

  React.useEffect(() => {
    const controller = new AbortController()
    scopeVerifiedRef.current = false
    setScopeVerified(false)
    setScopeVerificationError(false)
    setScopeLoading(true)
    setScopeError(null)
    void resolveServicePromptScope({ signal: controller.signal })
      .then((resolved) => {
        if (controller.signal.aborted) return
        setScope(resolved)
        scopeVerifiedRef.current = true
        setScopeVerified(true)
        setScopeLoading(false)
      })
      .catch((error) => {
        if (controller.signal.aborted || isAbortError(error)) return
        setScopeError(error)
        setScopeLoading(false)
      })
    return () => controller.abort()
  }, [scopeGeneration])

  const invalidateScope = React.useCallback(() => {
    const oldScope = scopeRef.current
    scopeReconcileControllerRef.current?.abort()
    scopeReconcileControllerRef.current = null
    clearPendingHistoryRestore()
    claimedHistoryRestoreRef.current = null
    pendingHistoryDestinationRef.current = null
    dirtyRef.current = false
    abortActiveOperation()
    if (oldScope) {
      const queryKey = ["service-prompts", oldScope.scopeKey]
      void queryClient.cancelQueries({ queryKey })
      void queryClient.invalidateQueries({ queryKey, refetchType: "none" })
    }
    migrationProbedScopeRef.current = null
    pendingFocusDefinitionRef.current = null
    setMigrationItems([])
    setMigrationError(null)
    setMigrationMessage(null)
    setMigrationProbeError(null)
    setScope(null)
    scopeVerifiedRef.current = false
    setScopeVerified(false)
    setScopeVerificationError(false)
    setScopeError(null)
    setScopeLoading(true)
    setScopeChanged(true)
    setDirty(false)
    setFieldErrors({})
    setPreview(null)
    setConflict(false)
    setOperationError(null)
    const next = new URLSearchParams(searchParams)
    next.delete("prompt")
    setSearchParams(next, { replace: true })
    setScopeGeneration((value) => value + 1)
  }, [abortActiveOperation, queryClient, searchParams, setSearchParams])

  const reconcileScope = React.useCallback(() => {
    const expectedScope = scopeRef.current
    if (!expectedScope) return
    scopeVerifiedRef.current = false
    setScopeVerified(false)
    setScopeVerificationError(false)
    abortActiveOperation()
    scopeReconcileControllerRef.current?.abort()
    const controller = new AbortController()
    scopeReconcileControllerRef.current = controller
    void resolveServicePromptScope({ signal: controller.signal })
      .then((currentScope) => {
        if (controller.signal.aborted ||
          scopeRef.current?.scopeKey !== expectedScope.scopeKey) {
          return
        }
        if (currentScope.scopeKey !== expectedScope.scopeKey) {
          invalidateScope()
          return
        }
        setScope(currentScope)
        scopeVerifiedRef.current = true
        setScopeVerified(true)
      })
      .catch((error) => {
        if (controller.signal.aborted ||
          scopeRef.current?.scopeKey !== expectedScope.scopeKey) {
          return
        }
        if (isServicePromptScopeUnresolvedError(error)) {
          invalidateScope()
          return
        }
        setScopeVerificationError(true)
      })
      .finally(() => {
        if (scopeReconcileControllerRef.current === controller) {
          scopeReconcileControllerRef.current = null
        }
      })
  }, [abortActiveOperation, invalidateScope])

  React.useEffect(() => {
    const unsubscribe = subscribeToServicePromptConfigChanges(reconcileScope)
    window.addEventListener("tldw:config-updated", reconcileScope)
    window.addEventListener("tldw:auth-credentials-changed", reconcileScope)
    return () => {
      scopeReconcileControllerRef.current?.abort()
      scopeReconcileControllerRef.current = null
      unsubscribe()
      window.removeEventListener("tldw:config-updated", reconcileScope)
      window.removeEventListener("tldw:auth-credentials-changed", reconcileScope)
    }
  }, [reconcileScope])

  const handleRequestScopeChanged = React.useCallback((error: unknown) => {
    if (!(error instanceof ServicePromptApiError) ||
      error.code !== "request_config_scope_changed") {
      return false
    }
    invalidateScope()
    return true
  }, [invalidateScope])

  const catalogKey = [
    "service-prompts",
    scope?.scopeKey ?? "unresolved",
    "catalog"
  ] as const
  const catalogQuery = useQuery({
    queryKey: catalogKey,
    enabled: Boolean(scope),
    queryFn: ({ signal }) => tldwClient.listServicePrompts({
      signal,
      requestScope: scope!
    })
  })

  React.useEffect(() => {
    if (catalogQuery.error) handleRequestScopeChanged(catalogQuery.error)
  }, [catalogQuery.error, handleRequestScopeChanged])

  React.useEffect(() => {
    if (!scope || !catalogQuery.data || migrationProbedScopeRef.current === scope.scopeKey) {
      return
    }
    migrationProbedScopeRef.current = scope.scopeKey
    const controller = new AbortController()
    setMigrationProbeError(null)
    void readLegacyServicePromptCandidates({ signal: controller.signal })
      .then((candidates) => {
        if (controller.signal.aborted || scopeRef.current?.scopeKey !== scope.scopeKey) {
          return
        }
        setMigrationItems(candidates)
      })
      .catch((error) => {
        if (!controller.signal.aborted && !isAbortError(error) &&
          scopeRef.current?.scopeKey === scope.scopeKey) {
          setMigrationProbeError(t("servicePrompts.migration.readFailed", {
            defaultValue: "Unable to read browser-local workflow prompts."
          }))
        }
      })
    return () => controller.abort()
  }, [catalogQuery.data, migrationProbeGeneration, scope, t])

  const selectedDefinition = React.useMemo(
    () => catalogQuery.data?.find((item) => item.id === selectedId) ?? null,
    [catalogQuery.data, selectedId]
  )
  const detailKey = [
    "service-prompts",
    scope?.scopeKey ?? "unresolved",
    "detail",
    selectedId ?? "unselected"
  ] as const
  const detailQuery = useQuery({
    queryKey: detailKey,
    enabled: Boolean(scope && selectedDefinition),
    queryFn: ({ signal }) =>
      tldwClient.getServicePrompt(selectedDefinition!.id, {
        signal,
        requestScope: scope!
      })
  })
  React.useEffect(() => {
    if (detailQuery.error) handleRequestScopeChanged(detailQuery.error)
  }, [detailQuery.error, handleRequestScopeChanged])
  const detailFocusReady = Boolean(
    detailQuery.data && draft && scope && selectedDefinition &&
    draft.scopeKey === scope.scopeKey &&
    draft.definitionId === selectedDefinition.id &&
    detailQuery.data.id === selectedDefinition.id
  )

  React.useEffect(() => {
    const detail = detailQuery.data
    if (!detail || !scope || dirty || dirtyRef.current) return
    setDraft({
      scopeKey: scope.scopeKey,
      definitionId: detail.id,
      parts: { ...detail.effective_parts },
      revision: detail.revision
    })
    setFieldErrors({})
    setPreview(null)
    setConflict(false)
    setOperationError(null)
  }, [detailQuery.data, dirty, scope])

  React.useEffect(() => {
    if (!selectedId || pendingFocusDefinitionRef.current !== selectedId ||
      !detailQuery.isError) {
      return
    }
    const target = detailFocusRef.current
    if (!target) return
    const timeout = window.setTimeout(() => {
      if (pendingFocusDefinitionRef.current !== selectedId ||
        selectedIdRef.current !== selectedId || !target.isConnected) {
        return
      }
      pendingFocusDefinitionRef.current = null
      target.focus()
    }, 0)
    return () => window.clearTimeout(timeout)
  }, [detailQuery.isError, selectedId])

  React.useEffect(() => {
    const selectionFocusPending = pendingFocusDefinitionRef.current === selectedId
    const historyFocusPending = historyFocusRef.current !== null
    if (!selectedId || (!selectionFocusPending && !historyFocusPending) ||
      !detailFocusReady) {
      return
    }
    const target = detailFocusRef.current
    if (!target) return
    const frame = window.requestAnimationFrame(() => {
      if (selectedIdRef.current !== selectedId || !target.isConnected) {
        return
      }
      const historyFocus = historyFocusRef.current
      if (historyFocus) {
        const historyTarget = historyFocus.id
          ? document.getElementById(historyFocus.id)
          : null
        const restored = historyTarget ?? (historyFocus.element?.isConnected
          ? historyFocus.element
          : target)
        historyFocusRef.current = null
        restored.focus()
        return
      }
      if (pendingFocusDefinitionRef.current === selectedId) {
        pendingFocusDefinitionRef.current = null
        target.focus()
      }
    })
    return () => window.cancelAnimationFrame(frame)
  }, [detailFocusReady, selectedId])

  React.useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirtyRef.current) return
      event.preventDefault()
      event.returnValue = ""
    }
    const anchorClick = (event: MouseEvent) => {
      if (event.defaultPrevented || event.button !== 0 ||
        event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return
      }
      const element = event.target instanceof Element
        ? event.target.closest("a")
        : null
      if (!(element instanceof HTMLAnchorElement) || element.target ||
        element.hasAttribute("download")) {
        return
      }
      const destination = resolveSettingsNavigationUrl(
        element.href,
        window.location.href
      )
      if (!destination) return
      if (!prepareHistoryNavigation(destination)) event.preventDefault()
    }
    const settingsNavigationRequest = (event: Event) => {
      const requestedDestination = (
        event as CustomEvent<SettingsNavigationRequestDetail>
      ).detail?.destination
      const destination = typeof requestedDestination === "string"
        ? resolveSettingsNavigationUrl(requestedDestination, window.location.href)
        : null
      if (!destination || !prepareHistoryNavigation(destination)) {
        event.preventDefault()
      }
    }
    const popstate = (event: PopStateEvent) => {
      if (suppressPopstateRef.current) {
        suppressPopstateRef.current = false
        const index = getHistoryIndex(event.state)
        if (index !== null) historyIndexRef.current = index
        historyForwardEntryTokenRef.current = getHistoryForwardEntryToken(event.state)
        const focusTarget = historyFocusRef.current
        const restoreFocus = (allowFallback: boolean) => {
          if (!focusTarget || historyFocusRef.current !== focusTarget) return
          const restoredById = focusTarget.id
            ? document.getElementById(focusTarget.id)
            : null
          const restored = restoredById ?? (focusTarget.element?.isConnected
            ? focusTarget.element
            : allowFallback && detailFocusRef.current?.isConnected
              ? detailFocusRef.current
              : null)
          if (restored) {
            historyFocusRef.current = null
            restored.focus()
          }
        }
        restoreFocus(false)
        if (historyFocusRef.current === focusTarget) {
          window.requestAnimationFrame(() => restoreFocus(true))
        }
        return
      }
      if (!dirtyRef.current) {
        const index = getHistoryIndex(event.state)
        if (index !== null) historyIndexRef.current = index
        historyForwardEntryTokenRef.current = getHistoryForwardEntryToken(event.state)
        return
      }
      const leave = window.confirm(t("servicePrompts.unsaved.leave", {
        defaultValue: "Discard unsaved workflow prompt changes?"
      }))
      if (leave) {
        clearPendingHistoryRestore()
        claimedHistoryRestoreRef.current = null
        dirtyRef.current = false
        setDirty(false)
        const index = getHistoryIndex(event.state)
        if (index !== null) historyIndexRef.current = index
        historyForwardEntryTokenRef.current = getHistoryForwardEntryToken(event.state)
        return
      }
      const targetIndex = getHistoryIndex(event.state)
      const forwardEntryToken = historyForwardEntryTokenRef.current
      const delta = targetIndex === null
        ? (forwardEntryToken &&
          getHistoryEntryToken(event.state) === forwardEntryToken ? -1 : 1)
        : historyIndexRef.current - targetIndex
      const activeElement = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
      const restoreState = historyRestoreStateRef.current
      const restoreDraft = restoreState.draft
      if (restoreDraft && scopeRef.current?.scopeKey === restoreDraft.scopeKey &&
        selectedIdRef.current === restoreDraft.definitionId) {
        stagePendingHistoryRestore({
          conflict: restoreState.conflict,
          definitionId: restoreDraft.definitionId,
          draft: { ...restoreDraft, parts: { ...restoreDraft.parts } },
          fieldErrors: { ...restoreState.fieldErrors },
          focusId: activeElement?.id || null,
          operationError: restoreState.operationError,
          preview: restoreState.preview ? { ...restoreState.preview } : null,
          scopeKey: restoreDraft.scopeKey,
          url: historyUrlRef.current
        })
      }
      historyFocusRef.current = {
        element: activeElement ?? detailFocusRef.current,
        id: activeElement?.id || null
      }
      suppressPopstateRef.current = true
      window.history.go(delta || 1)
    }

    window.addEventListener("beforeunload", beforeUnload)
    window.addEventListener("popstate", popstate)
    window.addEventListener(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      settingsNavigationRequest
    )
    document.addEventListener("click", anchorClick, true)
    return () => {
      stampPendingHistoryDestination()
      window.removeEventListener("beforeunload", beforeUnload)
      window.removeEventListener("popstate", popstate)
      window.removeEventListener(
        SETTINGS_NAVIGATION_REQUEST_EVENT,
        settingsNavigationRequest
      )
      document.removeEventListener("click", anchorClick, true)
    }
  }, [
    prepareHistoryNavigation,
    stampPendingHistoryDestination,
    t
  ])

  React.useEffect(() => () => {
    activeOperationRef.current?.controller.abort()
    activeOperationRef.current = null
  }, [])

  const definitionLabel = React.useCallback(
    (definition: ServicePromptCatalogItem) =>
      getDefinitionText(definition, "label", t),
    [t]
  )

  const selectDefinition = (id: string) => {
    if (id === selectedId) return
    if (dirty && !window.confirm(t("servicePrompts.unsaved.changePrompt", {
      defaultValue: "Discard unsaved changes and open another workflow prompt?"
    }))) {
      return
    }
    clearPendingHistoryRestore()
    claimedHistoryRestoreRef.current = null
    pendingHistoryDestinationRef.current = null
    dirtyRef.current = false
    abortActiveOperation()
    if (scope && selectedId) {
      const oldDetailKey = [
        "service-prompts",
        scope.scopeKey,
        "detail",
        selectedId
      ] as const
      void queryClient.cancelQueries({ queryKey: oldDetailKey })
      void queryClient.invalidateQueries({
        queryKey: oldDetailKey,
        refetchType: "none"
      })
    }
    pendingFocusDefinitionRef.current = id
    setDirty(false)
    setScopeChanged(false)
    setFieldErrors({})
    setPreview(null)
    setConflict(false)
    setOperationError(null)
    const next = new URLSearchParams(searchParams)
    next.set("prompt", id)
    replaceHistoryForwardEntryToken(null)
    historyIndexRef.current += 1
    setSearchParams(next)
  }

  const updatePart = (key: string, value: string) => {
    setDraft((current) => current ? {
      ...current,
      parts: { ...current.parts, [key]: value }
    } : current)
    setDirty(true)
    setFieldErrors((current) => {
      const next = { ...current }
      delete next[key]
      return next
    })
    setPreview(null)
    setConflict(false)
    setOperationError(null)
  }

  const draftIsCurrent = Boolean(
    draft && scope && selectedDefinition &&
    draft.scopeKey === scope.scopeKey &&
    draft.definitionId === selectedDefinition.id
  )

  const previewDraft = () => {
    if (!selectedDefinition || !draftIsCurrent || !draft) return
    const errors = validateServicePromptParts(selectedDefinition, draft.parts)
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) return
    const rendered: Record<string, string> = {}
    for (const part of selectedDefinition.parts) {
      const values = Object.fromEntries(
        part.required_variables.map((name) => [name, `[${name}]`])
      )
      rendered[part.key] = renderServicePromptPart(
        selectedDefinition,
        part.key,
        draft.parts[part.key],
        values
      )
    }
    setPreview(rendered)
  }

  const saveDraft = async () => {
    if (!selectedDefinition || !scope || !scopeVerifiedRef.current ||
      !draftIsCurrent || !draft) return
    const errors = validateServicePromptParts(selectedDefinition, draft.parts)
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) return
    const operationScope = scope.scopeKey
    const definitionId = selectedDefinition.id
    const operation = startOperation("save", operationScope, definitionId)
    const operationDetailKey = [
      "service-prompts",
      operationScope,
      "detail",
      definitionId
    ] as const
    setConflict(false)
    setOperationError(null)
    setOperationAnnouncement(t("servicePrompts.operations.saving", {
      defaultValue: "Saving workflow prompt…"
    }))
    try {
      const saved = await tldwClient.saveServicePrompt(
        definitionId,
        {
          parts: { ...draft.parts },
          expected_revision: draft.revision
        },
        {
          signal: operation.controller.signal,
          requestScope: scope
        }
      )
      if (!isCurrentOperation(operation)) return
      queryClient.setQueryData(operationDetailKey, saved)
      setDraft({
        scopeKey: operationScope,
        definitionId: saved.id,
        parts: { ...saved.effective_parts },
        revision: saved.revision
      })
      setDirty(false)
      setPreview(null)
      setOperationAnnouncement(t("servicePrompts.operations.saved", {
        defaultValue: "Workflow prompt saved."
      }))
      await queryClient.invalidateQueries({
        queryKey: operationDetailKey,
        refetchType: "none"
      })
    } catch (error) {
      if (!isCurrentOperation(operation) ||
        handleRequestScopeChanged(error) ||
        isAbortError(error)) return
      setOperationAnnouncement("")
      const validationEntries = error instanceof ServicePromptApiError &&
        error.status === 422 &&
        error.code === "service_prompt_validation_failed" &&
        error.fieldErrors
        ? Object.entries(error.fieldErrors)
        : []
      const partKeys = new Set(selectedDefinition.parts.map((part) => part.key))
      if (validationEntries.length > 0 &&
        validationEntries.every(([key, message]) =>
          partKeys.has(key) && message.trim().length > 0
        )
      ) {
        setFieldErrors(Object.fromEntries(validationEntries))
      } else if (error instanceof ServicePromptApiError && error.status === 409) {
        setConflict(true)
      } else {
        setOperationError(t("servicePrompts.errors.saveFailed", {
          defaultValue: "Unable to save this workflow prompt."
        }))
      }
    } finally {
      finishOperation(operation)
    }
  }

  const reloadServerValue = async () => {
    if (!selectedDefinition || !scope || !scopeVerifiedRef.current) return
    const operationScope = scope.scopeKey
    const definitionId = selectedDefinition.id
    const operation = startOperation("reload", operationScope, definitionId)
    const operationDetailKey = [
      "service-prompts",
      operationScope,
      "detail",
      definitionId
    ] as const
    setFieldErrors({})
    setOperationError(null)
    setOperationAnnouncement("")
    try {
      const result = await tldwClient.getServicePrompt(definitionId, {
        signal: operation.controller.signal,
        requestScope: scope
      })
      if (!isCurrentOperation(operation)) return
      queryClient.setQueryData(operationDetailKey, result)
      setDraft({
        scopeKey: operationScope,
        definitionId: result.id,
        parts: { ...result.effective_parts },
        revision: result.revision
      })
      setDirty(false)
      setConflict(false)
      setOperationError(null)
    } catch (error) {
      if (!isCurrentOperation(operation) ||
        handleRequestScopeChanged(error) ||
        isAbortError(error)) return
      setDirty(true)
      setConflict(true)
      setOperationError(t("servicePrompts.errors.reloadFailed", {
        defaultValue: "Unable to reload the server value."
      }))
    } finally {
      finishOperation(operation)
    }
  }

  const resetPrompt = async (revision: string | null, corrupt = false) => {
    if (!selectedDefinition || !scope || !scopeVerifiedRef.current ||
      (!corrupt && !draftIsCurrent) ||
      activeOperationRef.current !== null) {
      return
    }
    const operationScope = scope.scopeKey
    const definitionId = selectedDefinition.id
    const label = definitionLabel(selectedDefinition)
    const operation = startOperation("reset", operationScope, definitionId)
    const operationDetailKey = [
      "service-prompts",
      operationScope,
      "detail",
      definitionId
    ] as const
    try {
      let confirmed = false
      try {
        confirmed = await confirmDanger({
          title: t("servicePrompts.reset.title", {
            defaultValue: "Reset {{name}}?",
            name: label
          }),
          content: t("servicePrompts.reset.content", {
            defaultValue:
              "This will permanently remove the saved customization. There is no history or undo.",
            name: label
          }),
          okText: t("servicePrompts.actions.resetConfirm", { defaultValue: "Reset" })
        })
      } catch {
        if (isCurrentOperation(operation)) {
          setOperationError(t("servicePrompts.errors.resetFailed", {
            defaultValue: "Unable to reset this workflow prompt."
          }))
        }
        return
      }
      if (!confirmed || !isCurrentOperation(operation)) return

      setOperationError(null)
      setOperationAnnouncement(t("servicePrompts.operations.resetting", {
        defaultValue: "Resetting workflow prompt…"
      }))
      const requestScope = scope
      try {
        const reset = await tldwClient.resetServicePrompt(
          definitionId,
          revision,
          {
            signal: operation.controller.signal,
            requestScope
          }
        )
        if (!isCurrentOperation(operation)) return
        queryClient.setQueryData(operationDetailKey, reset)
        setDraft({
          scopeKey: operationScope,
          definitionId: reset.id,
          parts: { ...reset.effective_parts },
          revision: null
        })
        setDirty(false)
        setFieldErrors({})
        setPreview(null)
        setConflict(false)
        setOperationAnnouncement(t("servicePrompts.operations.reset", {
          defaultValue: "Workflow prompt reset to the server default."
        }))
        await queryClient.invalidateQueries({
          queryKey: operationDetailKey,
          refetchType: "none"
        })
      } catch (error) {
        if (!isCurrentOperation(operation) ||
          handleRequestScopeChanged(error) ||
          isAbortError(error)) return
        setOperationAnnouncement("")
        if (error instanceof ServicePromptApiError && error.status === 409) {
          if (corrupt) {
            try {
              const refreshed = await queryClient.fetchQuery({
                queryKey: operationDetailKey,
                queryFn: () => tldwClient.getServicePrompt(definitionId, {
                  signal: operation.controller.signal,
                  requestScope
                }),
                retry: false,
                staleTime: 0
              })
              if (!isCurrentOperation(operation)) return
              setDraft({
                scopeKey: operationScope,
                definitionId: refreshed.id,
                parts: { ...refreshed.effective_parts },
                revision: refreshed.revision
              })
              setDirty(false)
              setConflict(false)
            } catch (refreshError) {
              if (!isCurrentOperation(operation) ||
                handleRequestScopeChanged(refreshError) ||
                isAbortError(refreshError)) return
              if (!(refreshError instanceof ServicePromptApiError &&
                refreshError.code === "service_prompt_corrupt_override" &&
                refreshError.canReset === true &&
                typeof refreshError.revision === "string")) {
                setOperationError(t("servicePrompts.errors.resetFailed", {
                  defaultValue: "Unable to reset this workflow prompt."
                }))
              } else {
                const message = t("servicePrompts.corrupt.rebound", {
                  defaultValue:
                    "The saved customization changed. The latest revision was loaded. Retry reset."
                })
                setOperationError(message)
              }
            }
          } else {
            setConflict(true)
          }
        } else {
          setOperationError(t("servicePrompts.errors.resetFailed", {
            defaultValue: "Unable to reset this workflow prompt."
          }))
        }
      }
    } finally {
      finishOperation(operation)
    }
  }

  const remainingMigrationMessage = (count: number) =>
    t("servicePrompts.migration.remaining", {
      defaultValue: "{{count}} browser-local prompt still needs attention.",
      count
    })

  const importMigration = async () => {
    if (!scope || !scopeVerifiedRef.current || migrationItems.length === 0 ||
      activeKind !== null) return
    const operationScope = scope.scopeKey
    const operation = startOperation("migration-import", operationScope)
    setOperationAnnouncement("")
    setMigrationError(null)
    setMigrationMessage(null)
    const details = new Map<string, ServicePromptDetail>()
    const nextItems = migrationItems.map((item) => ({ ...item, error: undefined }))
    try {
      for (const item of nextItems) {
        const definition = catalogQuery.data?.find(
          (candidate) => candidate.id === item.definitionId
        )
        if (!definition) continue
        const errors = validateServicePromptParts(definition, {
          template: item.value
        })
        if (errors.template) item.error = errors.template
      }
      if (nextItems.some((item) => item.error)) {
        if (isCurrentOperation(operation)) setMigrationItems(nextItems)
        return
      }

      for (const item of nextItems) {
        const detail = await tldwClient.getServicePrompt(item.definitionId, {
          signal: operation.controller.signal,
          requestScope: scope
        })
        if (!isCurrentOperation(operation)) return
        details.set(item.definitionId, detail)
      }

      const replacements = nextItems.filter((item) => {
        const detail = details.get(item.definitionId)
        return detail?.source === "user" &&
          detail.effective_parts[item.partKey] !== item.value
      })
      if (replacements.length > 0) {
        const names = replacements.map((item) => {
          const definition = catalogQuery.data?.find(
            (candidate) => candidate.id === item.definitionId
          )
          return definition ? definitionLabel(definition) : item.definitionId
        }).join(", ")
        const confirmed = await confirmDanger({
          title: t("servicePrompts.migration.replaceTitle", {
            defaultValue: "Replace saved customizations?"
          }),
          content: t("servicePrompts.migration.replaceContent", {
            defaultValue: "Importing will replace saved customizations for: {{names}}.",
            names
          }),
          okText: t("servicePrompts.migration.replaceAction", {
            defaultValue: "Replace and import"
          })
        })
        if (!isCurrentOperation(operation) || !confirmed) return
      }

      let remaining = [...nextItems]
      for (const item of nextItems) {
        try {
          const saved = await importLegacyServicePromptCandidate(
            item,
            details.get(item.definitionId)!,
            {
              signal: operation.controller.signal,
              requestScope: scope
            }
          )
          if (!isCurrentOperation(operation)) return
          queryClient.setQueryData([
            "service-prompts",
            operationScope,
            "detail",
            item.definitionId
          ], saved)
          remaining = remaining.filter(
            (candidate) => candidate.definitionId !== item.definitionId
          )
          setMigrationItems(remaining)
          void queryClient.invalidateQueries({
            queryKey: ["service-prompts", operationScope],
            refetchType: "none"
          })
        } catch (error) {
          if (!isCurrentOperation(operation) ||
            handleRequestScopeChanged(error) ||
            isAbortError(error)) return
          remaining = remaining.map((candidate) =>
            candidate.definitionId === item.definitionId
              ? {
                  ...candidate,
                  error: t("servicePrompts.migration.importFailed", {
                    defaultValue: "Import failed. The browser-local value was preserved."
                  })
                }
              : candidate
          )
          setMigrationItems(remaining)
        }
      }
      if (remaining.length > 0) {
        setMigrationMessage(remainingMigrationMessage(remaining.length))
      }
    } catch (error) {
      if (!isCurrentOperation(operation) ||
        handleRequestScopeChanged(error) ||
        isAbortError(error)) return
      setMigrationItems(nextItems)
      setMigrationError(t("servicePrompts.migration.prepareFailed", {
        defaultValue:
          "Unable to prepare this import. The browser-local values were preserved."
      }))
      setMigrationMessage(remainingMigrationMessage(nextItems.length))
    } finally {
      finishOperation(operation)
    }
  }

  const discardMigration = async () => {
    if (!scope || !scopeVerifiedRef.current || migrationItems.length === 0 ||
      activeKind !== null ||
      activeOperationRef.current !== null) {
      return
    }
    const operationScope = scope.scopeKey
    const operation = startOperation("migration-discard", operationScope)
    try {
      let confirmed = false
      try {
        confirmed = await confirmDanger({
          title: t("servicePrompts.migration.discardTitle", {
            defaultValue: "Discard browser-local workflow prompts?"
          }),
          content: t("servicePrompts.migration.discardContent", {
            defaultValue:
              "This permanently removes only the three mapped browser-local values."
          }),
          okText: t("servicePrompts.migration.discardAction", {
            defaultValue: "Discard"
          })
        })
      } catch {
        if (isCurrentOperation(operation)) {
          setMigrationError(t("servicePrompts.migration.discardFailed", {
            defaultValue: "Discard failed. The browser-local value was preserved."
          }))
        }
        return
      }
      if (!confirmed || !isCurrentOperation(operation)) return

      setOperationAnnouncement("")
      setMigrationError(null)
      setMigrationMessage(null)
      let remaining = [...migrationItems]
      for (const item of migrationItems) {
        if (!isCurrentOperation(operation)) return
        try {
          await clearLegacyServicePromptCandidate(item.definitionId)
          if (!isCurrentOperation(operation)) return
          remaining = remaining.filter(
            (candidate) => candidate.definitionId !== item.definitionId
          )
          setMigrationItems(remaining)
        } catch (error) {
          if (!isCurrentOperation(operation) || isAbortError(error)) return
          remaining = remaining.map((candidate) =>
            candidate.definitionId === item.definitionId
              ? {
                  ...candidate,
                  error: t("servicePrompts.migration.discardFailed", {
                    defaultValue: "Discard failed. The browser-local value was preserved."
                  })
                }
              : candidate
          )
          setMigrationItems(remaining)
        }
      }
      if (isCurrentOperation(operation) && remaining.length > 0) {
        setMigrationMessage(remainingMigrationMessage(remaining.length))
      }
    } finally {
      finishOperation(operation)
    }
  }

  const retryScope = () => setScopeGeneration((value) => value + 1)
  const retryMigrationProbe = () => {
    migrationProbedScopeRef.current = null
    setMigrationProbeError(null)
    setMigrationProbeGeneration((value) => value + 1)
  }
  const catalogError = catalogQuery.error
  const unsupported = catalogError instanceof ServicePromptApiError &&
    catalogError.status === 404
  const corruptRevision = detailQuery.error instanceof ServicePromptApiError &&
    detailQuery.error.code === "service_prompt_corrupt_override" &&
    detailQuery.error.canReset === true &&
    typeof detailQuery.error.revision === "string"
    ? detailQuery.error.revision
    : null

  return (
    <div className="flex min-w-0 flex-col gap-5">
      <p className="sr-only" role="status" aria-live="polite">
        {operationAnnouncement}
      </p>
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-[70ch]">
          <p className="text-sm text-text-muted">
            {t("servicePrompts.description", {
              defaultValue:
                "Review and customize the instructions used by supported content workflows."
            })}
          </p>
        </div>
        <Link
          to="/prompts"
          className="w-fit text-sm font-medium text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          {t("servicePrompts.actions.openLibrary", {
            defaultValue: "Open reusable Prompts workspace"
          })}
        </Link>
      </div>

      {scopeChanged ? (
        <Alert variant="warning" title={t("servicePrompts.scope.changedTitle", {
          defaultValue: "Server or account changed"
        })}>
          {t("servicePrompts.scope.changedDescription", {
            defaultValue:
              "Choose a workflow prompt again after the new server and account scope loads. The previous draft cannot be saved."
          })}
        </Alert>
      ) : null}

      {scope && !scopeVerified ? (
        <Alert
          variant="warning"
          title={scopeVerificationError
            ? t("servicePrompts.scope.verificationFailedTitle", {
                defaultValue: "Server and account could not be verified"
              })
            : t("servicePrompts.scope.verifyingTitle", {
                defaultValue: "Verifying server and account…"
              })}
        >
          {scopeVerificationError ? (
            <Button size="small" onClick={reconcileScope}>
              {t("servicePrompts.actions.retry", { defaultValue: "Retry" })}
            </Button>
          ) : null}
        </Alert>
      ) : null}

      {scopeLoading ? (
        <div
          role="status"
          aria-live="polite"
          aria-busy="true"
          aria-label={t("servicePrompts.states.loadingScope", {
            defaultValue: "Loading server and account scope…"
          })}
        >
          <p className="mb-2 text-sm text-text-muted">
            {t("servicePrompts.states.loadingScope", {
              defaultValue: "Loading server and account scope…"
            })}
          </p>
          <Skeleton paragraph={{ rows: 5 }} />
        </div>
      ) : scopeError ? (
        <RecoveryCallout
          state="unavailable"
          role="alert"
          title={t("servicePrompts.errors.scopeTitle", {
            defaultValue: "Unable to resolve the connected server and account."
          })}
          message={t("servicePrompts.errors.scopeDescription", {
            defaultValue: "Check the connection and credentials, then try again."
          })}
          primaryAction={{
            label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
            onClick: retryScope
          }}
        />
      ) : scope && !scopeVerified ? (
        null
      ) : catalogQuery.isPending ? (
        <div
          role="status"
          aria-live="polite"
          aria-busy="true"
          aria-label={t("servicePrompts.states.loadingCatalog", {
            defaultValue: "Loading workflow prompts…"
          })}
        >
          <Skeleton paragraph={{ rows: 6 }} />
        </div>
      ) : unsupported ? (
        <RecoveryCallout
          state="unavailable"
          title={t("servicePrompts.unsupported.title", {
            defaultValue: "Workflow prompts require a server update"
          })}
          message={t("servicePrompts.unsupported.description", {
            defaultValue:
              "This older server does not support server-synced editing. Existing browser-local prompt behavior remains active until the server is updated."
          })}
        />
      ) : catalogQuery.isError ? (
        <RecoveryCallout
          state="error"
          role="alert"
          title={t("servicePrompts.errors.catalogTitle", {
            defaultValue: "Unable to load workflow prompts"
          })}
          message={t("servicePrompts.errors.catalogDescription", {
            defaultValue: "The connected server returned an error. No local values were changed."
          })}
          primaryAction={{
            label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
            onClick: () => void catalogQuery.refetch()
          }}
        />
      ) : (
        <>
          {scope && migrationProbeError && migrationItems.length === 0 ? (
            <RecoveryCallout
              state="error"
              role="alert"
              title={migrationProbeError}
              message={t("servicePrompts.migration.readFailedDescription", {
                defaultValue:
                  "No browser-local values were changed. Retry the browser storage check."
              })}
              primaryAction={{
                label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
                onClick: retryMigrationProbe
              }}
            />
          ) : null}
          {scope && migrationItems.length > 0 ? (
            <RecoveryCallout
              state="blocked"
              title={t("servicePrompts.migration.title", {
                defaultValue: "Browser-local workflow prompts found"
              })}
              message={t("servicePrompts.migration.description", {
                defaultValue:
                  "Review these browser-local values before using Chat with this server. Imported overrides are saved to the connected server and account. Portable backups do not include Service Prompt overrides."
              })}
            >
              <dl className="grid gap-1 text-sm sm:grid-cols-[8rem_minmax(0,1fr)]">
                <dt className="font-medium text-text-muted">
                  {t("servicePrompts.scope.server", { defaultValue: "Server" })}
                </dt>
                <dd className="break-all text-text">{scope.config.serverUrl}</dd>
                <dt className="font-medium text-text-muted">
                  {t("servicePrompts.scope.account", { defaultValue: "Account scope" })}
                </dt>
                <dd><code className="break-all text-xs text-text">{scope.scopeKey}</code></dd>
              </dl>
              <div className="mt-3 divide-y divide-border rounded-md border border-border">
                {migrationItems.map((item) => {
                  const definition = catalogQuery.data?.find(
                    (candidate) => candidate.id === item.definitionId
                  )
                  const label = definition
                    ? definitionLabel(definition)
                    : item.definitionId
                  const fieldId = `migration-${toDomId(item.definitionId)}`
                  const errorId = `${fieldId}-error`
                  return (
                    <div key={item.definitionId} className="p-3">
                      <label
                        htmlFor={fieldId}
                        className="text-sm font-semibold text-text"
                      >
                        {label}
                      </label>
                      <Input.TextArea
                        id={fieldId}
                        aria-label={t("servicePrompts.migration.repairLabel", {
                          defaultValue: "Repair {{name}}",
                          name: label
                        })}
                        aria-invalid={Boolean(item.error)}
                        aria-describedby={item.error ? errorId : undefined}
                        className="mt-2 font-mono"
                        disabled={activeKind !== null}
                        autoSize={{ minRows: 3, maxRows: 10 }}
                        value={item.value}
                        onChange={(event) => setMigrationItems((items) =>
                          items.map((candidate) =>
                            candidate.definitionId === item.definitionId
                              ? { ...candidate, value: event.target.value, error: undefined }
                              : candidate
                          )
                        )}
                      />
                      {item.error ? (
                        <p
                          id={errorId}
                          className="mt-1 text-sm text-danger"
                          role="alert"
                        >
                          {item.error}
                        </p>
                      ) : null}
                    </div>
                  )
                })}
              </div>
              {migrationError ? (
                <Alert className="mt-3" variant="error" title={migrationError} />
              ) : null}
              {migrationMessage ? (
                <p className="mt-3 text-sm text-warn" role="status">{migrationMessage}</p>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Button
                  type="primary"
                  loading={activeKind === "migration-import"}
                  disabled={!scopeVerified || activeKind !== null}
                  onClick={() => void importMigration()}
                >
                  {t("servicePrompts.migration.importAction", {
                    defaultValue: "Import to this server"
                  })}
                </Button>
                <Button
                  danger
                  loading={activeKind === "migration-discard"}
                  disabled={!scopeVerified || activeKind !== null}
                  onClick={() => void discardMigration()}
                >
                  {t("servicePrompts.migration.discardButton", {
                    defaultValue: "Discard local values"
                  })}
                </Button>
              </div>
            </RecoveryCallout>
          ) : null}

          <div className="grid min-w-0 grid-cols-1 gap-5 lg:grid-cols-[minmax(14rem,18rem)_minmax(0,1fr)]">
            <nav aria-label={t("servicePrompts.listLabel", {
              defaultValue: "Workflow prompt definitions"
            })}>
              <ul className="divide-y divide-border rounded-lg border border-border bg-surface">
                {(catalogQuery.data ?? []).map((definition) => {
                  const label = definitionLabel(definition)
                  const selected = definition.id === selectedId
                  return (
                    <li key={definition.id} data-testid="service-prompt-list-item">
                      <button
                        type="button"
                        aria-label={label}
                        aria-current={selected ? "page" : undefined}
                        onClick={() => selectDefinition(definition.id)}
                        className={`w-full px-3 py-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus ${
                          selected
                            ? "bg-primary/10 font-semibold text-primary"
                            : "text-text hover:bg-surface2"
                        }`}
                      >
                        <span className="block text-sm">{label}</span>
                        <span className="mt-0.5 block line-clamp-2 text-xs font-normal text-text-muted">
                          {getDefinitionText(definition, "description", t)}
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            </nav>

            {!selectedDefinition ? (
              <section className="min-w-0 rounded-lg border border-border bg-surface p-5">
                <h2 className="text-base font-semibold text-text">
                  {t("servicePrompts.states.chooseTitle", {
                    defaultValue: "Choose a workflow prompt"
                  })}
                </h2>
                <p className="mt-1 text-sm text-text-muted">
                  {t("servicePrompts.states.chooseDescription", {
                    defaultValue: "Select a definition to review its affected workflows and instructions."
                  })}
                </p>
              </section>
            ) : (
              <section
                ref={detailFocusRef}
                className="min-w-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                role="region"
                aria-label={t("servicePrompts.detailRegionLabel", {
                  defaultValue: "Workflow prompt details"
                })}
                tabIndex={-1}
              >
                {detailQuery.isPending ? (
                  <div
                    role="status"
                    aria-live="polite"
                    aria-busy="true"
                    aria-label={t("servicePrompts.states.loadingDetail", {
                      defaultValue: "Loading workflow prompt…"
                    })}
                  >
                    <Skeleton paragraph={{ rows: 7 }} />
                  </div>
            ) : corruptRevision ? (
              <RecoveryCallout
                state="error"
                title={t("servicePrompts.corrupt.title", {
                  defaultValue: "Saved customization is unavailable"
                })}
                message={t("servicePrompts.corrupt.description", {
                  defaultValue:
                    "The saved value cannot be read safely. Reset it to restore the server default."
                })}
                secondaryActions={[
                  {
                    label: t("servicePrompts.corrupt.resetAction", {
                      defaultValue: "Reset corrupt customization"
                    }),
                    onClick: () => void resetPrompt(corruptRevision, true),
                    loading: isResetting,
                    disabled: !scopeVerified || activeKind !== null
                  }
                ]}
              >
                {operationError ? (
                  <Alert variant="error" title={operationError} />
                ) : null}
              </RecoveryCallout>
            ) : detailQuery.isError && !(draft && draftIsCurrent && dirty) ? (
              <RecoveryCallout
                state="error"
                title={t("servicePrompts.errors.detailTitle", {
                  defaultValue: "Unable to load this workflow prompt"
                })}
                primaryAction={{
                  label: t("servicePrompts.actions.retry", { defaultValue: "Retry" }),
                  onClick: () => void detailQuery.refetch()
                }}
              />
            ) : draft && draftIsCurrent && detailQuery.data ? (
              <div
                className="min-w-0 rounded-lg border border-border bg-surface p-4 sm:p-5"
                role="region"
                aria-label={t("servicePrompts.editorLabel", {
                  defaultValue: "{{name}} editor",
                  name: definitionLabel(selectedDefinition)
                })}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h2 className="text-lg font-semibold text-text">
                      {definitionLabel(selectedDefinition)}
                    </h2>
                    <p className="mt-1 max-w-[70ch] text-sm text-text-muted">
                      {getDefinitionText(selectedDefinition, "description", t)}
                    </p>
                    {selectedDefinition.id === "chat.title.generation" ? (
                      <p className="mt-2 text-sm text-text-muted">
                        {t("servicePrompts.titleGeneration.note", {
                          defaultValue:
                            "Automatic title generation is enabled or disabled in Chat settings."
                        })}{" "}
                        <Link
                          className="font-medium text-primary hover:underline"
                          to="/settings/chat"
                        >
                          {t("servicePrompts.titleGeneration.openChatSettings", {
                            defaultValue: "Open Chat settings"
                          })}
                        </Link>
                      </p>
                    ) : null}
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Badge
                      variant={detailQuery.data.source === "user" ? "primary" : "secondary"}
                    >
                      {detailQuery.data.source === "user"
                        ? t("servicePrompts.states.customized", { defaultValue: "Customized" })
                        : t("servicePrompts.states.packaged", { defaultValue: "Server default" })}
                    </Badge>
                    {dirty ? (
                      <Badge variant="warning">
                        {t("servicePrompts.states.dirty", { defaultValue: "Unsaved" })}
                      </Badge>
                    ) : null}
                  </div>
                </div>

                {scope ? (
                  <dl className="mt-4 grid gap-x-4 gap-y-1 border-y border-border py-3 text-sm sm:grid-cols-[8rem_minmax(0,1fr)]">
                    <dt className="font-medium text-text-muted">
                      {t("servicePrompts.scope.server", { defaultValue: "Server" })}
                    </dt>
                    <dd className="break-all text-text">{scope.config.serverUrl}</dd>
                    <dt className="font-medium text-text-muted">
                      {t("servicePrompts.scope.account", { defaultValue: "Account scope" })}
                    </dt>
                    <dd><code className="break-all text-xs text-text">{scope.scopeKey}</code></dd>
                    <dt className="font-medium text-text-muted">
                      {t("servicePrompts.scope.workflows", { defaultValue: "Affected workflows" })}
                    </dt>
                    <dd>
                      <ul className="flex flex-wrap gap-x-3 gap-y-1">
                        {selectedDefinition.affected_workflows.map((workflow) => (
                          <li key={workflow.id}>
                            {getWorkflowLabel(workflow.id, workflow.label, t)}
                          </li>
                        ))}
                      </ul>
                    </dd>
                  </dl>
                ) : null}

                {conflict ? (
                  <Alert
                    className="mt-4"
                    variant="warning"
                    title={t("servicePrompts.conflict.title", {
                      defaultValue: "This prompt changed on the server."
                    })}
                    action={{
                      label: t("servicePrompts.conflict.reload", {
                        defaultValue: "Reload server value"
                      }),
                      onClick: () => void reloadServerValue(),
                      loading: activeKind === "reload",
                      disabled: !scopeVerified || activeKind !== null
                    }}
                  >
                    {t("servicePrompts.conflict.description", {
                      defaultValue:
                        "Your complete local draft is preserved. Reload only when you are ready to replace it."
                    })}
                  </Alert>
                ) : null}
                {operationError ? (
                  <Alert className="mt-4" variant="error" title={operationError} />
                ) : null}

                <Form className="mt-5" layout="vertical" onFinish={() => void saveDraft()}>
                  <div className="flex flex-col gap-5">
                    {selectedDefinition.parts.map((part) => {
                      const partLabel = getPartLabel(part.key, part.label, t)
                      const fieldId = `service-prompt-${toDomId(
                        selectedDefinition.id
                      )}-${toDomId(part.key)}`
                      const errorId = `${fieldId}-error`
                      const fieldError = fieldErrors[part.key]
                      return (
                        <section key={part.key} className="min-w-0">
                          <Form.Item
                            label={partLabel}
                            validateStatus={fieldError ? "error" : undefined}
                            help={fieldError
                              ? <span id={errorId}>{fieldError}</span>
                              : undefined}
                          >
                            <Input.TextArea
                              id={fieldId}
                              aria-label={partLabel}
                              aria-invalid={Boolean(fieldError)}
                              aria-describedby={fieldError ? errorId : undefined}
                              className="font-mono"
                              disabled={activeKind !== null}
                              autoSize={{ minRows: 6, maxRows: 18 }}
                              value={draft.parts[part.key]}
                              onChange={(event) => updatePart(part.key, event.target.value)}
                            />
                          </Form.Item>
                          {part.mode === "template" ? (
                            <div className="flex flex-wrap items-center gap-2" aria-label={t(
                              "servicePrompts.variables.label",
                              { defaultValue: "Required variables" }
                            )}>
                              <span className="text-xs font-medium text-text-muted">
                                {t("servicePrompts.variables.label", {
                                  defaultValue: "Required variables"
                                })}
                              </span>
                              {part.required_variables.map((variable) => (
                                <Badge key={variable} size="sm" outline>
                                  {`{${variable}}`}
                                </Badge>
                              ))}
                            </div>
                          ) : null}
                        </section>
                      )
                    })}
                  </div>

                  {preview ? (
                    <section
                      className="mt-5 min-w-0 rounded-lg border border-border bg-surface2 p-3"
                      aria-label={t("servicePrompts.preview.label", {
                        defaultValue: "Prompt preview"
                      })}
                    >
                      <h3 className="text-sm font-semibold text-text">
                        {t("servicePrompts.preview.title", { defaultValue: "Local preview" })}
                      </h3>
                      <div className="mt-3 flex flex-col gap-3">
                        {selectedDefinition.parts.map((part) => (
                          <div key={part.key} className="min-w-0">
                            <p className="text-xs font-medium text-text-muted">
                              {getPartLabel(part.key, part.label, t)}
                            </p>
                            <pre
                              className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface p-3 text-sm text-text"
                              tabIndex={0}
                            >
                              <code role="code">{preview[part.key]}</code>
                            </pre>
                          </div>
                        ))}
                      </div>
                    </section>
                  ) : null}

                  <div className="mt-5 flex flex-wrap items-center gap-2">
                    <Button onClick={previewDraft} disabled={activeKind !== null}>
                      {t("servicePrompts.actions.preview", { defaultValue: "Preview" })}
                    </Button>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={isSaving}
                      disabled={!dirty || !draftIsCurrent || !scopeVerified ||
                        activeKind !== null}
                    >
                      {t("servicePrompts.actions.save", { defaultValue: "Save changes" })}
                    </Button>
                    <Button
                      danger
                      loading={isResetting}
                      disabled={detailQuery.data.source !== "user" || !draftIsCurrent ||
                        !scopeVerified || activeKind !== null}
                      onClick={() => void resetPrompt(draft.revision)}
                    >
                      {t("servicePrompts.actions.reset", {
                        defaultValue: "Reset to default"
                      })}
                    </Button>
                  </div>
                </Form>
              </div>
                ) : null}
              </section>
            )}
          </div>
        </>
      )}
    </div>
  )
}

export default ServicePromptsSettings
