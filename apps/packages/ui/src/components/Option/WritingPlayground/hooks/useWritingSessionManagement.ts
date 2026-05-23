/**
 * Hook: useWritingSessionManagement
 *
 * Manages session CRUD, save/load, active session tracking, debounced auto-save,
 * version conflict handling, and session usage history.
 */

import React from "react"
import type { TFunction } from "i18next"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type { JSONContent } from "@tiptap/react"
import type { MessageInstance } from "antd/es/message/interface"
import {
  cloneWritingSession,
  createWritingSession,
  deleteWritingSession,
  getWritingSession,
  listWritingSessions,
  updateWritingSession,
  type WritingSessionListItem,
  type WritingSessionListResponse
} from "@/services/writing-playground"
import {
  areSettingsEqual,
  cloneDefaultSettings,
  DEFAULT_SETTINGS,
  getChatModeFromPayload,
  getPromptFromPayload,
  getPromptRichFromPayload,
  getSettingsFromPayload,
  getTemplateNameFromPayload,
  getThemeNameFromPayload,
  isVersionConflictError,
  mergePendingPayloadIntoSession,
  normalizeStringArrayValue,
  SAVE_DEBOUNCE_MS,
  type PendingSave,
  type SessionUsageMap,
  type WritingSessionPayload,
  type WritingSessionSettings
} from "./utils"
import {
  getImportedSessionModelHint,
  getImportedSessionProviderHint
} from "../writing-session-import-utils"
import { formatLogitBiasValue } from "../writing-logit-bias-utils"

export interface UseWritingSessionManagementDeps {
  isOnline: boolean
  hasWriting: boolean
  activeSessionId: string | null
  activeSessionName: string | null
  setActiveSessionId: (id: string | null) => void
  setActiveSessionName: (name: string | null) => void
  sessionUsageMap: SessionUsageMap | undefined
  setSessionUsageMap: (map: SessionUsageMap) => void
  selectedModel: string | undefined
  setSelectedModel: (model: string) => Promise<void> | void
  apiProviderOverride: string | undefined
  setApiProvider: (provider: string) => void
  isGenerating: boolean
  t: TFunction
}

type PromptSelection = { start: number; end: number }
type PromptUpdateOptions =
  | PromptSelection
  | {
      selection?: PromptSelection
      promptRich?: JSONContent | null
    }

const isPromptSelection = (
  options: PromptUpdateOptions | undefined
): options is PromptSelection =>
  Boolean(options) &&
  typeof (options as PromptSelection).start === "number" &&
  typeof (options as PromptSelection).end === "number"

const getPromptSelection = (
  options: PromptUpdateOptions | undefined
): PromptSelection | undefined => {
  if (!options) return undefined
  return isPromptSelection(options) ? options : options.selection
}

const getPromptRichOption = (
  options: PromptUpdateOptions | undefined
): JSONContent | null | undefined => {
  if (!options || isPromptSelection(options)) return undefined
  if (!Object.prototype.hasOwnProperty.call(options, "promptRich")) {
    return undefined
  }
  return options.promptRich ?? null
}

const serializePromptRich = (promptRich: JSONContent | null): string | null =>
  promptRich ? JSON.stringify(promptRich) : null

export function useWritingSessionManagement(deps: UseWritingSessionManagementDeps) {
  const {
    isOnline,
    hasWriting,
    activeSessionId,
    activeSessionName,
    setActiveSessionId,
    setActiveSessionName,
    sessionUsageMap,
    setSessionUsageMap,
    selectedModel,
    setSelectedModel,
    apiProviderOverride,
    setApiProvider,
    isGenerating,
    t
  } = deps

  const queryClient = useQueryClient()

  // --- Local UI state ---
  const [createModalOpen, setCreateModalOpen] = React.useState(false)
  const [newSessionName, setNewSessionName] = React.useState("")
  const [renameModalOpen, setRenameModalOpen] = React.useState(false)
  const [renameSessionName, setRenameSessionName] = React.useState("")
  const [renameTarget, setRenameTarget] =
    React.useState<WritingSessionListItem | null>(null)
  const [isDirty, setIsDirty] = React.useState(false)
  const [lastSavedAt, setLastSavedAt] = React.useState<number | null>(null)

  // --- Editor state that is session-coupled ---
  const [editorText, setEditorText] = React.useState("")
  const [settings, setSettings] =
    React.useState<WritingSessionSettings>(() => cloneDefaultSettings())
  const [stopStringsInput, setStopStringsInput] = React.useState("")
  const [bannedTokensInput, setBannedTokensInput] = React.useState("")
  const [drySequenceBreakersInput, setDrySequenceBreakersInput] = React.useState("")
  const [logitBiasInput, setLogitBiasInput] = React.useState("")
  const [logitBiasError, setLogitBiasError] = React.useState<string | null>(null)
  const [logitBiasTokenInput, setLogitBiasTokenInput] = React.useState("")
  const [logitBiasValueInput, setLogitBiasValueInput] = React.useState<number | null>(null)
  const [selectedTemplateName, setSelectedTemplateName] =
    React.useState<string | null>(null)
  const [selectedThemeName, setSelectedThemeName] =
    React.useState<string | null>(null)
  const [chatMode, setChatMode] = React.useState(false)

  // --- Refs ---
  const saveTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingSaveMapRef = React.useRef<Record<string, WritingSessionPayload>>({})
  const pendingQueueRef = React.useRef<string[]>([])
  const saveInFlightRef = React.useRef(false)
  const savingSessionIdRef = React.useRef<string | null>(null)
  const sessionVersionRef = React.useRef<Record<string, number>>({})
  const sessionSchemaVersionRef = React.useRef<Record<string, number>>({})
  const lastLoadedSessionIdRef = React.useRef<string | null>(null)
  const lastSavedPromptRef = React.useRef<Record<string, string>>({})
  const lastSavedPromptRichRef = React.useRef<Record<string, string | null>>({})
  const lastSavedSettingsRef =
    React.useRef<Record<string, WritingSessionSettings>>({})
  const lastSavedTemplateNameRef = React.useRef<Record<string, string | null>>({})
  const lastSavedThemeNameRef = React.useRef<Record<string, string | null>>({})
  const lastSavedChatModeRef = React.useRef<Record<string, boolean>>({})
  const editorPromptRichRef = React.useRef<JSONContent | null>(null)

  // --- Queries ---
  const {
    data: sessionsData,
    isLoading: sessionsLoading,
    isFetching: sessionsFetching,
    error: sessionsError
  } = useQuery({
    queryKey: ["writing-sessions"],
    queryFn: () => listWritingSessions({ limit: 200 }),
    enabled: isOnline && hasWriting,
    staleTime: 30 * 1000
  })
  const sessions = sessionsData?.sessions ?? []

  const {
    data: activeSessionDetail,
    isLoading: activeSessionLoading,
    error: activeSessionError
  } = useQuery({
    queryKey: ["writing-session", activeSessionId],
    queryFn: () => getWritingSession(activeSessionId ?? ""),
    enabled: isOnline && hasWriting && Boolean(activeSessionId),
    staleTime: 30 * 1000
  })

  // --- Helpers ---
  const refreshSessionData = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["writing-sessions"] })
    if (activeSessionId) {
      queryClient.invalidateQueries({
        queryKey: ["writing-session", activeSessionId]
      })
    }
  }, [activeSessionId, queryClient])

  const handleVersionConflict = React.useCallback(() => {
    refreshSessionData()
  }, [refreshSessionData])

  // --- Mutations ---
  const createSessionMutation = useMutation({
    mutationFn: (name: string) =>
      createWritingSession({
        name,
        payload: {
          prompt: "",
          settings: cloneDefaultSettings(),
          template_name: null,
          theme_name: null,
          chat_mode: false
        },
        schema_version: 1
      }),
    onSuccess: (session) => {
      queryClient.invalidateQueries({ queryKey: ["writing-sessions"] })
      setCreateModalOpen(false)
      setNewSessionName("")
      setActiveSessionId(session.id)
      setActiveSessionName(session.name)
      const nextUsage = {
        ...(sessionUsageMap || {}),
        [session.id]: { name: session.name, lastUsedAt: Date.now() }
      }
      setSessionUsageMap(nextUsage)
    }
  })

  const renameSessionMutation = useMutation({
    mutationFn: (payload: { session: WritingSessionListItem; name: string }) =>
      updateWritingSession(
        payload.session.id,
        { name: payload.name },
        payload.session.version
      ),
    onSuccess: (session, payload) => {
      queryClient.invalidateQueries({ queryKey: ["writing-sessions"] })
      setRenameModalOpen(false)
      setRenameTarget(null)
      setRenameSessionName("")
      if (activeSessionId === payload.session.id) {
        setActiveSessionName(session.name)
      }
      const nextUsage = { ...(sessionUsageMap || {}) }
      if (nextUsage[payload.session.id]) {
        nextUsage[payload.session.id] = {
          ...nextUsage[payload.session.id],
          name: session.name
        }
        setSessionUsageMap(nextUsage)
      }
    },
    onError: (err) => {
      if (isVersionConflictError(err)) {
        handleVersionConflict()
        setRenameModalOpen(false)
        setRenameTarget(null)
        setRenameSessionName("")
      }
    }
  })

  const deleteSessionMutation = useMutation({
    mutationFn: (payload: { session: WritingSessionListItem }) =>
      deleteWritingSession(payload.session.id, payload.session.version),
    onSuccess: (_data, payload) => {
      queryClient.invalidateQueries({ queryKey: ["writing-sessions"] })
      if (activeSessionId === payload.session.id) {
        setActiveSessionId(null)
        setActiveSessionName(null)
      }
      const nextUsage = { ...(sessionUsageMap || {}) }
      if (nextUsage[payload.session.id]) {
        delete nextUsage[payload.session.id]
        setSessionUsageMap(nextUsage)
      }
    },
    onError: (err) => {
      if (isVersionConflictError(err)) {
        handleVersionConflict()
      }
    }
  })

  const cloneSessionMutation = useMutation({
    mutationFn: (payload: { session: WritingSessionListItem }) =>
      cloneWritingSession(payload.session.id),
    onSuccess: (session) => {
      queryClient.invalidateQueries({ queryKey: ["writing-sessions"] })
      setActiveSessionId(session.id)
      setActiveSessionName(session.name)
      const nextUsage = {
        ...(sessionUsageMap || {}),
        [session.id]: { name: session.name, lastUsedAt: Date.now() }
      }
      setSessionUsageMap(nextUsage)
    }
  })

  const saveSessionMutation = useMutation({
    mutationFn: (payload: PendingSave & { expectedVersion: number }) =>
      updateWritingSession(
        payload.sessionId,
        {
          payload: payload.payload,
          schema_version:
            sessionSchemaVersionRef.current[payload.sessionId] ?? 1
        },
        payload.expectedVersion
      ),
    onMutate: () => {
      saveInFlightRef.current = true
    },
    onSuccess: (session, payload) => {
      saveInFlightRef.current = false
      if (savingSessionIdRef.current === session.id) {
        savingSessionIdRef.current = null
      }
      sessionVersionRef.current[session.id] = session.version
      sessionSchemaVersionRef.current[session.id] = session.schema_version
      lastSavedPromptRef.current[session.id] = getPromptFromPayload(session.payload)
      lastSavedPromptRichRef.current[session.id] =
        serializePromptRich(getPromptRichFromPayload(session.payload))
      lastSavedSettingsRef.current[session.id] = getSettingsFromPayload(session.payload)
      lastSavedTemplateNameRef.current[session.id] =
        getTemplateNameFromPayload(session.payload)
      lastSavedThemeNameRef.current[session.id] =
        getThemeNameFromPayload(session.payload)
      lastSavedChatModeRef.current[session.id] = getChatModeFromPayload(
        session.payload
      )
      queryClient.setQueryData(
        ["writing-session", session.id],
        session
      )
      queryClient.setQueryData<WritingSessionListResponse | undefined>(
        ["writing-sessions"],
        (prev) => {
          if (!prev) return prev
          return {
            ...prev,
            sessions: prev.sessions.map((item) =>
              item.id === session.id
                ? {
                    ...item,
                    name: session.name,
                    last_modified: session.last_modified,
                    version: session.version
                  }
                : item
            )
          }
        }
      )
      const pendingPayload = pendingSaveMapRef.current[session.id]
      if (!pendingPayload || pendingPayload === payload.payload) {
        delete pendingSaveMapRef.current[session.id]
      }
      if (activeSessionId === session.id) {
        if (!pendingPayload || pendingPayload === payload.payload) {
          editorPromptRichRef.current = getPromptRichFromPayload(session.payload)
        }
        setLastSavedAt(Date.now())
        if (!pendingSaveMapRef.current[session.id]) {
          setIsDirty(false)
        }
      }
      if (!saveInFlightRef.current) {
        flushNextSave()
      }
    },
    onError: (err) => {
      saveInFlightRef.current = false
      savingSessionIdRef.current = null
      if (isVersionConflictError(err)) {
        handleVersionConflict()
      }
    }
  })

  // --- Save infrastructure ---
  const flushNextSave = React.useCallback(() => {
    if (saveInFlightRef.current) return
    const queue = pendingQueueRef.current
    while (queue.length > 0) {
      const sessionId = queue.shift()
      if (!sessionId) continue
      const payload = pendingSaveMapRef.current[sessionId]
      if (!payload) continue
      const expectedVersion = sessionVersionRef.current[sessionId]
      if (expectedVersion == null) {
        delete pendingSaveMapRef.current[sessionId]
        continue
      }
      savingSessionIdRef.current = sessionId
      saveSessionMutation.mutate({
        sessionId,
        payload,
        expectedVersion
      })
      return
    }
  }, [saveSessionMutation])

  const scheduleSave = React.useCallback(
    (sessionId: string, payload: WritingSessionPayload) => {
      pendingSaveMapRef.current[sessionId] = payload
      if (!pendingQueueRef.current.includes(sessionId)) {
        pendingQueueRef.current.push(sessionId)
      }
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
      }
      saveTimerRef.current = setTimeout(() => {
        saveTimerRef.current = null
        flushNextSave()
      }, SAVE_DEBOUNCE_MS)
    },
    [flushNextSave]
  )

  const clearPendingSave = React.useCallback((sessionId: string) => {
    delete pendingSaveMapRef.current[sessionId]
    pendingQueueRef.current = pendingQueueRef.current.filter(
      (queuedId) => queuedId !== sessionId
    )
    if (pendingQueueRef.current.length === 0 && saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
      saveTimerRef.current = null
    }
  }, [])

  const computeDirty = React.useCallback(
    (
      sessionId: string,
      prompt: string,
      promptRich: JSONContent | null,
      nextSettings: WritingSessionSettings,
      templateName: string | null,
      themeName: string | null,
      nextChatMode: boolean
    ) => {
      const lastPrompt = lastSavedPromptRef.current[sessionId] ?? ""
      const lastPromptRich = lastSavedPromptRichRef.current[sessionId] ?? null
      const lastSettings =
        lastSavedSettingsRef.current[sessionId] ?? DEFAULT_SETTINGS
      const lastTemplate = lastSavedTemplateNameRef.current[sessionId] ?? null
      const lastTheme = lastSavedThemeNameRef.current[sessionId] ?? null
      const lastChatMode = lastSavedChatModeRef.current[sessionId] ?? false
      return (
        prompt !== lastPrompt ||
        serializePromptRich(promptRich) !== lastPromptRich ||
        !areSettingsEqual(nextSettings, lastSettings) ||
        templateName !== lastTemplate ||
        themeName !== lastTheme ||
        nextChatMode !== lastChatMode
      )
    },
    []
  )

  // --- Session select ---
  const sortedSessions = React.useMemo(() => {
    const usage = sessionUsageMap || {}
    return sessions
      .map((session, index) => ({
        session,
        index,
        lastUsedAt: usage[session.id]?.lastUsedAt ?? 0
      }))
      .sort((a, b) => {
        if (a.lastUsedAt !== b.lastUsedAt) {
          return b.lastUsedAt - a.lastUsedAt
        }
        return a.index - b.index
      })
  }, [sessions, sessionUsageMap])

  const handleSelectSession = React.useCallback(
    (session: WritingSessionListItem) => {
      if (isGenerating) return
      setActiveSessionId(session.id)
      setActiveSessionName(session.name)
      const nextUsage = {
        ...(sessionUsageMap || {}),
        [session.id]: { name: session.name, lastUsedAt: Date.now() }
      }
      setSessionUsageMap(nextUsage)
    },
    [
      isGenerating,
      setActiveSessionId,
      setActiveSessionName,
      sessionUsageMap,
      setSessionUsageMap
    ]
  )

  const openRenameModal = React.useCallback(
    (session: WritingSessionListItem) => {
      setRenameTarget(session)
      setRenameSessionName(session.name)
      setRenameModalOpen(true)
    },
    []
  )

  const activeSession = activeSessionId
    ? sessions.find((session) => session.id === activeSessionId)
    : null

  const canCreateSession = newSessionName.trim().length > 0
  const canRenameSession =
    renameSessionName.trim().length > 0 &&
    renameTarget != null &&
    renameSessionName.trim() !== renameTarget.name

  // --- Session data sync effects ---
  React.useEffect(() => {
    if (!activeSessionId) return
    const match = sessions.find((session) => session.id === activeSessionId)
    if (match && match.name !== activeSessionName) {
      setActiveSessionName(match.name)
    }
  }, [activeSessionId, activeSessionName, sessions, setActiveSessionName])

  React.useEffect(() => {
    if (!activeSessionId) return
    if (sessionsFetching) return
    const exists = sessions.some((session) => session.id === activeSessionId)
    if (!exists) {
      setActiveSessionId(null)
      setActiveSessionName(null)
    }
  }, [
    activeSessionId,
    sessions,
    sessionsFetching,
    setActiveSessionId,
    setActiveSessionName
  ])

  React.useEffect(() => {
    if (!activeSessionDetail) {
      if (lastLoadedSessionIdRef.current === null) {
        editorPromptRichRef.current = null
        return
      }
      setEditorText("")
      editorPromptRichRef.current = null
      setSettings(cloneDefaultSettings())
      setStopStringsInput("")
      setBannedTokensInput("")
      setDrySequenceBreakersInput("")
      setLogitBiasInput("")
      setLogitBiasError(null)
      setLogitBiasTokenInput("")
      setLogitBiasValueInput(null)
      setSelectedTemplateName(null)
      setSelectedThemeName(null)
      setChatMode(false)
      setIsDirty(false)
      lastLoadedSessionIdRef.current = null
      return
    }
    sessionVersionRef.current[activeSessionDetail.id] = activeSessionDetail.version
    sessionSchemaVersionRef.current[activeSessionDetail.id] =
      activeSessionDetail.schema_version
    const nextPrompt = getPromptFromPayload(activeSessionDetail.payload)
    const nextPromptRich = getPromptRichFromPayload(activeSessionDetail.payload)
    const nextSettings = getSettingsFromPayload(activeSessionDetail.payload)
    const nextTemplateName = getTemplateNameFromPayload(activeSessionDetail.payload)
    const nextThemeName = getThemeNameFromPayload(activeSessionDetail.payload)
    const nextChatMode = getChatModeFromPayload(activeSessionDetail.payload)
    const nextModelHint = getImportedSessionModelHint(activeSessionDetail.payload)
    const nextProviderHint = getImportedSessionProviderHint(
      activeSessionDetail.payload
    )
    const lastLoadedId = lastLoadedSessionIdRef.current
    if (activeSessionDetail.id !== lastLoadedId) {
      setEditorText(nextPrompt)
      editorPromptRichRef.current = nextPromptRich
      setSettings(nextSettings)
      setStopStringsInput(nextSettings.stop.join("\n"))
      setBannedTokensInput(
        normalizeStringArrayValue(nextSettings.advanced_extra_body.banned_tokens).join("\n")
      )
      setDrySequenceBreakersInput(
        normalizeStringArrayValue(
          nextSettings.advanced_extra_body.dry_sequence_breakers
        ).join("\n")
      )
      setLogitBiasInput(
        formatLogitBiasValue(nextSettings.advanced_extra_body.logit_bias)
      )
      setLogitBiasError(null)
      setLogitBiasTokenInput("")
      setLogitBiasValueInput(null)
      setSelectedTemplateName(nextTemplateName)
      setSelectedThemeName(nextThemeName)
      setChatMode(nextChatMode)
      setIsDirty(false)
      setLastSavedAt(Date.now())
      lastSavedPromptRef.current[activeSessionDetail.id] = nextPrompt
      lastSavedPromptRichRef.current[activeSessionDetail.id] =
        serializePromptRich(nextPromptRich)
      lastSavedSettingsRef.current[activeSessionDetail.id] = nextSettings
      lastSavedTemplateNameRef.current[activeSessionDetail.id] = nextTemplateName
      lastSavedThemeNameRef.current[activeSessionDetail.id] = nextThemeName
      lastSavedChatModeRef.current[activeSessionDetail.id] = nextChatMode
      if (nextModelHint && nextModelHint !== selectedModel) {
        void setSelectedModel(nextModelHint)
      }
      if (nextProviderHint && nextProviderHint !== apiProviderOverride) {
        setApiProvider(nextProviderHint)
      }
      lastLoadedSessionIdRef.current = activeSessionDetail.id
      return
    }
    if (!isDirty) {
      if (editorText !== nextPrompt) {
        setEditorText(nextPrompt)
      }
      editorPromptRichRef.current = nextPromptRich
      if (!areSettingsEqual(settings, nextSettings)) {
        setSettings(nextSettings)
        setStopStringsInput(nextSettings.stop.join("\n"))
        setBannedTokensInput(
          normalizeStringArrayValue(
            nextSettings.advanced_extra_body.banned_tokens
          ).join("\n")
        )
        setDrySequenceBreakersInput(
          normalizeStringArrayValue(
            nextSettings.advanced_extra_body.dry_sequence_breakers
          ).join("\n")
        )
        setLogitBiasInput(
          formatLogitBiasValue(nextSettings.advanced_extra_body.logit_bias)
        )
        setLogitBiasError(null)
        setLogitBiasTokenInput("")
        setLogitBiasValueInput(null)
      }
      if (selectedTemplateName !== nextTemplateName) {
        setSelectedTemplateName(nextTemplateName)
      }
      if (selectedThemeName !== nextThemeName) {
        setSelectedThemeName(nextThemeName)
      }
      if (chatMode !== nextChatMode) {
        setChatMode(nextChatMode)
      }
      setLastSavedAt(Date.now())
      lastSavedPromptRef.current[activeSessionDetail.id] = nextPrompt
      lastSavedPromptRichRef.current[activeSessionDetail.id] =
        serializePromptRich(nextPromptRich)
      lastSavedSettingsRef.current[activeSessionDetail.id] = nextSettings
      lastSavedTemplateNameRef.current[activeSessionDetail.id] = nextTemplateName
      lastSavedThemeNameRef.current[activeSessionDetail.id] = nextThemeName
      lastSavedChatModeRef.current[activeSessionDetail.id] = nextChatMode
    }
  }, [
    activeSessionDetail,
    chatMode,
    editorText,
    isDirty,
    apiProviderOverride,
    selectedTemplateName,
    selectedThemeName,
    selectedModel,
    setApiProvider,
    setSelectedModel,
    settings
  ])

  React.useEffect(() => {
    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
      }
    }
  }, [])

  // --- Prompt/settings change helpers ---
  const applySessionPayloadPatch = React.useCallback(
    (patcher: (payload: WritingSessionPayload) => WritingSessionPayload) => {
      if (!activeSessionDetail) return
      const basePayload = mergePendingPayloadIntoSession(
        activeSessionDetail.payload,
        pendingSaveMapRef.current[activeSessionDetail.id],
        editorText,
        settings,
        selectedTemplateName,
        selectedThemeName,
        chatMode,
        { promptRich: editorPromptRichRef.current }
      )
      const nextPayload = patcher(basePayload)
      pendingSaveMapRef.current[activeSessionDetail.id] = nextPayload
      setIsDirty(true)
      scheduleSave(activeSessionDetail.id, nextPayload)
    },
    [
      activeSessionDetail,
      chatMode,
      editorText,
      scheduleSave,
      selectedTemplateName,
      selectedThemeName,
      settings
    ]
  )

  const applyPromptValue = React.useCallback(
    (
      nextValue: string,
      options?: PromptUpdateOptions
    ) => {
      const selection = getPromptSelection(options)
      const promptRichOption = getPromptRichOption(options)
      const promptRich =
        promptRichOption === undefined ? editorPromptRichRef.current : promptRichOption
      setEditorText(nextValue)
      editorPromptRichRef.current = promptRich
      if (!activeSessionDetail) return selection
      const nextPayload = mergePendingPayloadIntoSession(
        activeSessionDetail.payload,
        pendingSaveMapRef.current[activeSessionDetail.id],
        nextValue,
        settings,
        selectedTemplateName,
        selectedThemeName,
        chatMode,
        { promptRich }
      )
      const isDirtyNext = computeDirty(
        activeSessionDetail.id,
        nextValue,
        promptRich,
        settings,
        selectedTemplateName,
        selectedThemeName,
        chatMode
      )
      setIsDirty(isDirtyNext)
      if (!isDirtyNext) {
        clearPendingSave(activeSessionDetail.id)
      } else {
        scheduleSave(activeSessionDetail.id, nextPayload)
      }
      return selection
    },
    [
      activeSessionDetail,
      chatMode,
      clearPendingSave,
      computeDirty,
      scheduleSave,
      selectedTemplateName,
      selectedThemeName,
      settings
    ]
  )

  const handleSettingsChange = React.useCallback(
    (
      nextSettings: WritingSessionSettings,
      nextStopInput?: string | null
    ) => {
      if (!activeSessionDetail) return
      setSettings(nextSettings)
      if (typeof nextStopInput === "string") {
        setStopStringsInput(nextStopInput)
      }
      const nextPayload = mergePendingPayloadIntoSession(
        activeSessionDetail.payload,
        pendingSaveMapRef.current[activeSessionDetail.id],
        editorText,
        nextSettings,
        selectedTemplateName,
        selectedThemeName,
        chatMode,
        { promptRich: editorPromptRichRef.current }
      )
      const isDirtyNext = computeDirty(
        activeSessionDetail.id,
        editorText,
        editorPromptRichRef.current,
        nextSettings,
        selectedTemplateName,
        selectedThemeName,
        chatMode
      )
      setIsDirty(isDirtyNext)
      if (!isDirtyNext) {
        clearPendingSave(activeSessionDetail.id)
        return
      }
      scheduleSave(activeSessionDetail.id, nextPayload)
    },
    [
      activeSessionDetail,
      chatMode,
      clearPendingSave,
      computeDirty,
      editorText,
      scheduleSave,
      selectedTemplateName,
      selectedThemeName
    ]
  )

  const updateSetting = React.useCallback(
    (partial: Partial<WritingSessionSettings>, nextStopInput?: string) => {
      const nextSettings = { ...settings, ...partial }
      handleSettingsChange(nextSettings, nextStopInput)
    },
    [handleSettingsChange, settings]
  )

  const handleTemplateChange = React.useCallback(
    (nextTemplateName: string | null) => {
      setSelectedTemplateName(nextTemplateName)
      if (!activeSessionDetail) return
      const nextPayload = mergePendingPayloadIntoSession(
        activeSessionDetail.payload,
        pendingSaveMapRef.current[activeSessionDetail.id],
        editorText,
        settings,
        nextTemplateName,
        selectedThemeName,
        chatMode,
        { promptRich: editorPromptRichRef.current }
      )
      const isDirtyNext = computeDirty(
        activeSessionDetail.id,
        editorText,
        editorPromptRichRef.current,
        settings,
        nextTemplateName,
        selectedThemeName,
        chatMode
      )
      setIsDirty(isDirtyNext)
      if (!isDirtyNext) {
        clearPendingSave(activeSessionDetail.id)
        return
      }
      scheduleSave(activeSessionDetail.id, nextPayload)
    },
    [
      activeSessionDetail,
      chatMode,
      clearPendingSave,
      computeDirty,
      editorText,
      scheduleSave,
      settings,
      selectedThemeName
    ]
  )

  const handleThemeChange = React.useCallback(
    (nextThemeName: string | null) => {
      setSelectedThemeName(nextThemeName)
      if (!activeSessionDetail) return
      const nextPayload = mergePendingPayloadIntoSession(
        activeSessionDetail.payload,
        pendingSaveMapRef.current[activeSessionDetail.id],
        editorText,
        settings,
        selectedTemplateName,
        nextThemeName,
        chatMode,
        { promptRich: editorPromptRichRef.current }
      )
      const isDirtyNext = computeDirty(
        activeSessionDetail.id,
        editorText,
        editorPromptRichRef.current,
        settings,
        selectedTemplateName,
        nextThemeName,
        chatMode
      )
      setIsDirty(isDirtyNext)
      if (!isDirtyNext) {
        clearPendingSave(activeSessionDetail.id)
        return
      }
      scheduleSave(activeSessionDetail.id, nextPayload)
    },
    [
      activeSessionDetail,
      chatMode,
      clearPendingSave,
      computeDirty,
      editorText,
      scheduleSave,
      selectedTemplateName,
      settings
    ]
  )

  const handleChatModeChange = React.useCallback(
    (nextChatMode: boolean) => {
      setChatMode(nextChatMode)
      if (!activeSessionDetail) return
      const nextPayload = mergePendingPayloadIntoSession(
        activeSessionDetail.payload,
        pendingSaveMapRef.current[activeSessionDetail.id],
        editorText,
        settings,
        selectedTemplateName,
        selectedThemeName,
        nextChatMode,
        { promptRich: editorPromptRichRef.current }
      )
      const isDirtyNext = computeDirty(
        activeSessionDetail.id,
        editorText,
        editorPromptRichRef.current,
        settings,
        selectedTemplateName,
        selectedThemeName,
        nextChatMode
      )
      setIsDirty(isDirtyNext)
      if (!isDirtyNext) {
        clearPendingSave(activeSessionDetail.id)
        return
      }
      scheduleSave(activeSessionDetail.id, nextPayload)
    },
    [
      activeSessionDetail,
      clearPendingSave,
      computeDirty,
      editorText,
      scheduleSave,
      selectedTemplateName,
      selectedThemeName,
      settings
    ]
  )

  return {
    // queries
    sessions,
    sessionsLoading,
    sessionsFetching,
    sessionsError,
    activeSessionDetail,
    activeSessionLoading,
    activeSessionError,
    activeSession,
    sortedSessions,
    // state
    createModalOpen, setCreateModalOpen,
    newSessionName, setNewSessionName,
    renameModalOpen, setRenameModalOpen,
    renameSessionName, setRenameSessionName,
    renameTarget, setRenameTarget,
    isDirty, setIsDirty,
    lastSavedAt, setLastSavedAt,
    editorText, setEditorText,
    settings, setSettings,
    stopStringsInput, setStopStringsInput,
    bannedTokensInput, setBannedTokensInput,
    drySequenceBreakersInput, setDrySequenceBreakersInput,
    logitBiasInput, setLogitBiasInput,
    logitBiasError, setLogitBiasError,
    logitBiasTokenInput, setLogitBiasTokenInput,
    logitBiasValueInput, setLogitBiasValueInput,
    selectedTemplateName, setSelectedTemplateName,
    selectedThemeName, setSelectedThemeName,
    chatMode, setChatMode,
    // mutations
    createSessionMutation,
    renameSessionMutation,
    deleteSessionMutation,
    cloneSessionMutation,
    saveSessionMutation,
    // refs
    savingSessionIdRef,
    sessionVersionRef,
    sessionSchemaVersionRef,
    // callbacks
    refreshSessionData,
    handleVersionConflict,
    handleSelectSession,
    openRenameModal,
    applyPromptValue,
    handleSettingsChange,
    updateSetting,
    handleTemplateChange,
    handleThemeChange,
    handleChatModeChange,
    applySessionPayloadPatch,
    scheduleSave,
    clearPendingSave,
    computeDirty,
    flushNextSave,
    // derived
    canCreateSession,
    canRenameSession
  }
}
