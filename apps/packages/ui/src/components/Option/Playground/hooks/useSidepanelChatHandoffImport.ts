import React from "react"
import { useLocation, useNavigate } from "react-router-dom"
import { useStoreMessageOption, type State } from "@/store/option"
import { fetchChatModels } from "@/services/tldw-server"
import { buildChatModelUsability } from "@/utils/chat-model-availability"
import { resolveModelSelectorSelection } from "@/hooks/playground/modelSelectorUtils"

import {
  consumeSidepanelChatHandoff,
  readSidepanelChatHandoff,
  type SidepanelChatHandoffPackage,
  type SidepanelChatHandoffPageContext,
} from "@/services/sidepanel-chat-handoff"

import { decodeSidepanelChatWebUiHandoff, isEncodedSidepanelChatWebUiHandoff } from "@/services/tldw/sidepanel-chat-webui-handoff"

const literalOptionKeys = [
  "selectedModel",
  "chatMode",
  "webSearch",
  "toolChoice",
  "useOCR"
] as const
type LiteralOptions = Partial<Pick<State, (typeof literalOptionKeys)[number]>>
type LiteralOptionControls = {
  current: Pick<State, (typeof literalOptionKeys)[number]>
  setSelectedModel: (model: string | null) => void | Promise<void>
  setChatMode: State["setChatMode"]
  setWebSearch: State["setWebSearch"]
  setToolChoice: State["setToolChoice"]
  setUseOCR: State["setUseOCR"]
  canApply?: () => boolean
}

type HandoffRoute = { id: string; field: "search" | "hash"; webUi: boolean }
type HandoffDraft = Pick<SidepanelChatHandoffPackage, "id" | "draft" | "pageContext"> & {
  hasText?: boolean
  literalOptions?: LiteralOptions
  unsupportedSettings?: boolean
  skippedParts?: Array<"settings" | "history" | "attachments">
}

const routeParams = (value: string) => new URLSearchParams(value.replace(/^[?#]/, ""))
const getHandoffRoute = (location: { search: string; hash: string }): HandoffRoute | null => {
  const fragment = routeParams(location.hash).get("handoff")
  if (fragment !== null) return { id: fragment, field: "hash", webUi: true }
  const id = routeParams(location.search).get("handoff")
  return id ? { id, field: "search", webUi: isEncodedSidepanelChatWebUiHandoff(id) } : null
}
const matchesHandoffRoute = (location: { search: string; hash: string }, route: HandoffRoute) =>
  routeParams(location[route.field]).get("handoff") === route.id &&
  (route.field === "hash" || !routeParams(location.hash).has("handoff"))
const readHandoffDraft = async (route: HandoffRoute, strict = false): Promise<HandoffDraft | null> => {
  if (!route.webUi) return readSidepanelChatHandoff(route.id, { throwOnReadError: strict })
  const payload = decodeSidepanelChatWebUiHandoff(route.id)
  if (!payload) return null
  const has = (fields: Array<keyof typeof payload>) => fields.some(field => Object.prototype.hasOwnProperty.call(payload, field))
  const skippedParts: NonNullable<HandoffDraft["skippedParts"]> = []
  if (has(["selectedModel", "selectedSystemPrompt", "selectedQuickPrompt", "chatMode", "webSearch", "toolChoice", "temporaryChat", "useOCR"])) skippedParts.push("settings")
  if (has(["historyId", "serverChatId", "title"])) skippedParts.push("history")
  if (has(["chatDocumentDraftId", "ragMediaIds", "fileRetrievalEnabled"])) skippedParts.push("attachments")
  const literalOptions = Object.fromEntries(literalOptionKeys.filter(key => Object.prototype.hasOwnProperty.call(payload, key)).map(key => [key, payload[key]])) as LiteralOptions
  return { id: route.id, draft: { text: payload.draft ?? "" }, hasText: typeof payload.draft === "string", skippedParts,
    literalOptions, unsupportedSettings: has(["selectedSystemPrompt", "selectedQuickPrompt", "temporaryChat"]) }
}

type SetMessageValue = (
  value: string,
  options?: { collapseLarge?: boolean; forceCollapse?: boolean },
) => void

type NotificationApi = {
  warning?: (args: { message: string; description?: string }) => void
}

export type SidepanelChatHandoffConflict = {
  handoff: HandoffDraft
  route: HandoffRoute
}

type UseSidepanelChatHandoffImportDeps = {
  ready: boolean
  literalOptions?: LiteralOptionControls
  destinationKey: string
  captureGuard: () => () => boolean
  draftValue: string
  getDraftRevision: () => number
  setMessageValue: SetMessageValue
  notificationApi?: NotificationApi
  t: (key: string, defaultValueOrOptions?: any, options?: any) => string
}

const appendImportedDraft = (currentDraft: string, importedDraft: string) => {
  if (!currentDraft) return importedDraft
  if (!importedDraft) return currentDraft
  return `${currentDraft}\n\n${importedDraft}`
}

export function useSidepanelChatHandoffImport({
  ready,
  literalOptions,
  destinationKey,
  captureGuard,
  draftValue,
  getDraftRevision,
  setMessageValue,
  notificationApi,
  t,
}: UseSidepanelChatHandoffImportDeps) {
  const location = useLocation()
  const navigate = useNavigate()
  const [importedContext, setImportedContext] =
    React.useState<SidepanelChatHandoffPageContext | null>(null)
  const [conflict, setConflict] =
    React.useState<SidepanelChatHandoffConflict | null>(null)
  const [includeLiteralOptions, setIncludeOptions] = React.useState(false)
  const consentRevision = React.useRef(0)
  const includeOptionsRef = React.useRef(false)
  const renderedConsentRevision = consentRevision.current
  const optionRevision = React.useRef(0)
  const renderedOptionRevision = optionRevision.current
  const [importPending, setImportPending] = React.useState(false)
  const [conflictFeedback, setConflictFeedback] = React.useState<string | null>(
    null
  )
  const [importReceipt, setImportReceipt] = React.useState<{
    description: string
    modelPending?: boolean
  } | null>(null)
  React.useLayoutEffect(
    () =>
      useStoreMessageOption.subscribe((current, previous) => {
        if (
          current.selectedModelRevision === previous.selectedModelRevision &&
          !literalOptionKeys.some((key) => current[key] !== previous[key])
        ) return
        optionRevision.current += 1
        setImportReceipt((receipt) => (receipt?.modelPending ? null : receipt))
      }),
    []
  )
  const skippedFeedbackFor = React.useCallback(
    (handoff: HandoffDraft, optionsApplied = false) => {
      if (!handoff.skippedParts?.length) return null
      const labels = handoff.skippedParts
        .filter(
          (part) =>
            part !== "settings" ||
            !optionsApplied ||
            handoff.unsupportedSettings
        )
        .map((part) =>
          part === "settings"
            ? optionsApplied
              ? t(
                  "playground:sidepanelHandoff.skippedCommandsPrivacy",
                  "Prompts and chat privacy"
                )
              : t("playground:sidepanelHandoff.skippedSettings", "Settings")
            : part === "history"
              ? t(
                  "playground:sidepanelHandoff.skippedHistory",
                  "Conversation history"
                )
              : t(
                  "playground:sidepanelHandoff.skippedAttachments",
                  "Attachments and retrieval sources"
                )
        )
      if (!labels.length) return null
      return `${t("playground:sidepanelHandoff.notImported", "Not imported")}: ${labels.join(", ")}. ${t("playground:sidepanelHandoff.chooseHere", "Choose these in the destination chat.")}`
    },
    [t]
  )

  const pendingChoiceRef = React.useRef<{ location: typeof location; draft: string; revision: number; isOwned: () => boolean } | null>(null)
  const locationRef = React.useRef(location)
  locationRef.current = location
  const draftValueRef = React.useRef(draftValue)
  draftValueRef.current = draftValue
  const handledHandoffIdsRef = React.useRef(new Map<string, { destinationKey: string; settled: boolean }>())
  const conflictOwnerRef = React.useRef<{ conflict: SidepanelChatHandoffConflict; isOwned: () => boolean } | null>(null)
  const readyRef = React.useRef(ready)
  readyRef.current = ready

  const retireConflict = React.useCallback(() => {
    const conflictId = conflictOwnerRef.current?.conflict.handoff.id
    const retired = conflictId ? handledHandoffIdsRef.current.get(conflictId) : undefined
    if (retired) retired.settled = false
    pendingChoiceRef.current = null
    conflictOwnerRef.current = null
    setImportPending(false)
    setConflict(null)
    setConflictFeedback(null)
    includeOptionsRef.current = false
    consentRevision.current += 1
    setIncludeOptions(false)
  }, [])

  React.useLayoutEffect(() => {
    retireConflict()
    setImportedContext(null)
    setImportReceipt(null)
  }, [destinationKey, ready, retireConflict])

  React.useLayoutEffect(() => {
    if (location.pathname !== "/chat") setImportReceipt(null)
    retireConflict()
    return retireConflict
  }, [destinationKey, location, retireConflict])

  const cleanRouteHandoff = React.useCallback((route: HandoffRoute) => {
    const current = locationRef.current
    if (!matchesHandoffRoute(current, route)) return
    const cleanField = (field: "search" | "hash") => {
      const value = current[field]
      const raw = field === "hash" ? value.replace(/^#/, "") : value
      const queryIndex = field === "hash" ? raw.indexOf("?") : -1
      const hashPath = queryIndex >= 0 ? raw.slice(0, queryIndex) : ""
      const params = routeParams(queryIndex >= 0 ? raw.slice(queryIndex + 1) : raw)
      const original = params.toString()
      if (field === route.field) params.delete("handoff")
      // A literal copy cannot authorize a character card or session after its token leaves.
      const modes = params.getAll("mode")
      if (modes.some(mode => mode.trim().toLowerCase() === "character")) {
        params.delete("mode")
        modes.filter(mode => mode.trim().toLowerCase() !== "character").forEach(mode => params.append("mode", mode))
      }
      for (const pointer of ["characterId", "character_id", "chatId", "chat_id", "serverChatId", "server_chat_id"]) params.delete(pointer)
      const remaining = params.toString()
      if (remaining === original) return value
      if (field === "search") return remaining ? `?${remaining}` : ""
      const fragment = hashPath + (remaining ? `${queryIndex >= 0 ? "?" : ""}${remaining}` : "")
      return fragment ? `#${fragment}` : ""
    }
    navigate(
      {
        pathname: current.pathname,
        search: cleanField("search"),
        hash: cleanField("hash"),
      },
      { replace: true },
    )
  }, [navigate])

  const showInvalidFeedback = React.useCallback(() => {
    notificationApi?.warning?.({
      message: t(
        "playground:sidepanelHandoff.unavailableTitle",
        "Sidepanel handoff unavailable",
      ),
      description: t(
        "playground:sidepanelHandoff.unavailableCopyDescription",
        "The copy expired or could not be read. Your destination draft remains here. Copy again from the sidepanel to try again.",
      ),
    })
  }, [notificationApi, t])

  const consumeQuietly = React.useCallback((handoffId: string, isOwned: () => boolean) => {
    void consumeSidepanelChatHandoff(handoffId, isOwned).catch(() => undefined)
  }, [])

  const applyImportedHandoff = React.useCallback(
    (
      handoff: HandoffDraft,
      route: HandoffRoute,
      nextDraft: string,
      options: {
        isOwned: () => boolean
        settingsApplied?: boolean
        modelPersisting?: boolean
        optionsOnly?: boolean
      }
    ) => {
      if (
        !readyRef.current ||
        !options.isOwned() ||
        !matchesHandoffRoute(locationRef.current, route)
      )
        return
      if (handoff.hasText !== false && !options.optionsOnly) {
        setMessageValue(nextDraft, { collapseLarge: true })
        setImportedContext(handoff.pageContext ?? null)
      }
      const receipt = {
        modelPending: options.modelPersisting,
        description: [
          handoff.hasText !== false && !options.optionsOnly
            ? handoff.draft.text === "" && handoff.pageContext
              ? t(
                  "playground:sidepanelHandoff.pageReferencesImported",
                  "Page references imported."
                )
              : t(
                  "playground:sidepanelHandoff.draftImported",
                  "Draft text imported."
                )
            : null,
          options.settingsApplied
            ? t(
                "playground:sidepanelHandoff.settingsApplied",
                "Supported settings applied here."
              )
            : null,
          options.modelPersisting
            ? t(
                "playground:sidepanelHandoff.modelSaving",
                "Saving the model preference…"
              )
            : null,
          skippedFeedbackFor(handoff, options.settingsApplied)
        ]
          .filter(Boolean)
          .join(" ")
      }
      setImportReceipt(receipt)
      setConflict(null)
      conflictOwnerRef.current = null
      cleanRouteHandoff(route)
      handledHandoffIdsRef.current.set(handoff.id, {
        destinationKey,
        settled: true
      })
      queueMicrotask(() => {
        if (!route.webUi && options.isOwned())
          consumeQuietly(handoff.id, options.isOwned)
      })
      return receipt
    },
    [
      cleanRouteHandoff,
      consumeQuietly,
      destinationKey,
      setMessageValue,
      skippedFeedbackFor,
      t
    ]
  )

  React.useEffect(() => {
    const route = getHandoffRoute(location)
    const handoffId = route?.id
    const previous = handoffId ? handledHandoffIdsRef.current.get(handoffId) : undefined
    if (!ready || location.pathname !== "/chat" || !route || !handoffId || (previous?.destinationKey === destinationKey && previous.settled && (!conflictOwnerRef.current || conflictOwnerRef.current.isOwned()))) return
    const isOwned = captureGuard()
    if (!isOwned()) return
    setImportReceipt(null)
    if (conflictOwnerRef.current && !conflictOwnerRef.current.isOwned()) retireConflict()
    const attempt = { destinationKey, settled: false }
    handledHandoffIdsRef.current.set(handoffId, attempt)

    let cancelled = false

    void (async () => {
      const handoff = await readHandoffDraft(route)
      if (cancelled || handledHandoffIdsRef.current.get(handoffId) !== attempt || !readyRef.current || !isOwned()) return
      attempt.settled = true

      if (!handoff) {
        setConflict(null)
        showInvalidFeedback()
        cleanRouteHandoff(route)
        return
      }

      // A readable literal copy has no source-account proof, even in an empty draft.
      const nextConflict = { handoff, route }
      conflictOwnerRef.current = { conflict: nextConflict, isOwned }
      setConflict(nextConflict)
    })()

    return () => {
      cancelled = true
    }
  }, [captureGuard, cleanRouteHandoff, destinationKey, location, ready, retireConflict, showInvalidFeedback])

  const removeImportedContext = React.useCallback(() => {
    setImportedContext(current => current === importedContext ? null : current)
  }, [importedContext])

  const isConflictCurrent = React.useCallback(() => readyRef.current && conflict !== null &&
    conflictOwnerRef.current?.conflict === conflict && conflictOwnerRef.current.isOwned() &&
    matchesHandoffRoute(locationRef.current, conflict.route), [conflict])

  const setIncludeLiteralOptions = React.useCallback(
    (include: boolean) => {
      if (
        !isConflictCurrent() ||
        pendingChoiceRef.current ||
        renderedConsentRevision !== consentRevision.current
      )
        return
      if (includeOptionsRef.current === include) return
      includeOptionsRef.current = include
      consentRevision.current += 1
      setIncludeOptions(include)
    },
    [isConflictCurrent, renderedConsentRevision]
  )
  const optionLabels = {
    selectedModel: t(
      "playground:sidepanelHandoff.optionModel",
      "Model and provider"
    ),
    chatMode: t("playground:sidepanelHandoff.optionMode", "Chat mode"),
    webSearch: t("playground:sidepanelHandoff.optionWebSearch", "Web search"),
    toolChoice: t("playground:sidepanelHandoff.optionTools", "Tool choice"),
    useOCR: t("playground:sidepanelHandoff.optionOCR", "OCR")
  }
  const literalOptionsReview =
    literalOptions && conflict?.handoff.literalOptions
      ? literalOptionKeys
          .filter((key) =>
            Object.prototype.hasOwnProperty.call(
              conflict.handoff.literalOptions,
              key
            )
          )
          .map((key) => ({
            key,
            label: optionLabels[key],
            current: literalOptions.current[key],
            copied: conflict.handoff.literalOptions![key]
          }))
      : []

  const acceptHandoffDraft = React.useCallback(
    async (action: "insert" | "replace" | "options") => {
      if (
        !isConflictCurrent() ||
        !conflict ||
        pendingChoiceRef.current ||
        (action !== "options" && conflict.handoff.hasText === false) ||
        (action === "options" &&
          (!includeLiteralOptions || !literalOptionsReview.length))
      )
        return
      if (renderedConsentRevision !== consentRevision.current) return
      const applyOptions =
        includeLiteralOptions && Boolean(literalOptionsReview.length)
      const attempt = {
        location: locationRef.current,
        draft: draftValueRef.current,
        revision: getDraftRevision(),
        optionsRevision: renderedOptionRevision,
        isOwned: captureGuard()
      }
      if (!attempt.isOwned()) return
      pendingChoiceRef.current = attempt
      setImportPending(true)
      setConflictFeedback(null)
      const isCurrent = () =>
        readyRef.current &&
        attempt.isOwned() &&
        pendingChoiceRef.current === attempt &&
        locationRef.current.key === attempt.location.key &&
        locationRef.current.pathname === attempt.location.pathname &&
        locationRef.current.search === attempt.location.search &&
        locationRef.current.hash === attempt.location.hash &&
        matchesHandoffRoute(locationRef.current, conflict.route)
      try {
        const handoff = await readHandoffDraft(conflict.route, true)
        if (!isCurrent()) return
        if (!handoff) {
          setConflict(null)
          showInvalidFeedback()
          cleanRouteHandoff(conflict.route)
          return
        }
        const unchanged = () =>
          getDraftRevision() === attempt.revision &&
          draftValueRef.current === attempt.draft &&
          (!applyOptions || optionRevision.current === attempt.optionsRevision)
        const showChanged = () =>
          setConflictFeedback(
            t(
              applyOptions
                ? "playground:sidepanelHandoff.destinationChanged"
                : "playground:sidepanelHandoff.draftChanged",
              applyOptions
                ? "Your draft or settings changed while the copy was checked. Review them, then choose again."
                : "Your draft changed while the copy was checked. Review it, then choose Insert or Replace again."
            )
          )
        if (!unchanged()) {
          showChanged()
          return
        }
        let modelChoice: string | undefined
        if (applyOptions) {
          if (!literalOptions || literalOptions.canApply?.() === false) {
            setConflictFeedback(
              t(
                "playground:sidepanelHandoff.settingsUnavailable",
                "Wait for this conversation to finish loading, or uncheck settings to import text only."
              )
            )
            return
          }
          if (
            Object.prototype.hasOwnProperty.call(
              handoff.literalOptions,
              "selectedModel"
            )
          ) {
            const models = await fetchChatModels({ requireFresh: true })
            if (!isCurrent()) return
            if (!unchanged()) {
              showChanged()
              return
            }
            const choice = resolveModelSelectorSelection(
              models,
              handoff.literalOptions!.selectedModel
            )
            if (
              !choice.model ||
              !choice.key ||
              !choice.provider ||
              !buildChatModelUsability({
                selectedModel: choice.key,
                availableModels: models
              }).canSend
            ) {
              setConflictFeedback(
                t(
                  "playground:sidepanelHandoff.modelUnavailable",
                  "The copied model and provider could not be verified. Choose them here, or uncheck settings to import text only."
                )
              )
              return
            }
            modelChoice = choice.key
          }
        }
        if (!isCurrent()) return
        if (!unchanged()) {
          showChanged()
          return
        }
        if (applyOptions && literalOptions?.canApply?.() === false) {
          setConflictFeedback(
            t(
              "playground:sidepanelHandoff.settingsUnavailable",
              "Wait for this conversation to finish loading, or uncheck settings to import text only."
            )
          )
          return
        }
        if (
          conflict.route.webUi &&
          !decodeSidepanelChatWebUiHandoff(conflict.route.id)
        ) {
          setConflict(null)
          showInvalidFeedback()
          cleanRouteHandoff(conflict.route)
          return
        }
        let modelPersistence: Promise<void> | undefined
        if (applyOptions && literalOptions) {
          const values = handoff.literalOptions!
          if (values.chatMode !== undefined)
            literalOptions.setChatMode(values.chatMode)
          if (values.webSearch !== undefined)
            literalOptions.setWebSearch(values.webSearch)
          if (values.toolChoice !== undefined)
            literalOptions.setToolChoice(values.toolChoice)
          if (values.useOCR !== undefined)
            literalOptions.setUseOCR(values.useOCR)
          if (modelChoice !== undefined) {
            try {
              modelPersistence = Promise.resolve(
                literalOptions.setSelectedModel(modelChoice)
              )
            } catch (error) {
              modelPersistence = Promise.reject(error)
            }
          }
        }
        const committedOptionRevision = optionRevision.current
        const receipt = applyImportedHandoff(
          handoff,
          conflict.route,
          action === "insert"
            ? appendImportedDraft(draftValueRef.current, handoff.draft.text)
            : handoff.draft.text,
          {
            isOwned: attempt.isOwned,
            settingsApplied: applyOptions,
            modelPersisting: Boolean(modelPersistence),
            optionsOnly: action === "options"
          }
        )
        if (modelPersistence) {
          try {
            await modelPersistence
            if (
              receipt &&
              attempt.isOwned() &&
              optionRevision.current === committedOptionRevision
            )
              setImportReceipt((current) =>
                current === receipt
                  ? {
                      description: receipt.description.replace(
                        t(
                          "playground:sidepanelHandoff.modelSaving",
                          "Saving the model preference…"
                        ),
                        t(
                          "playground:sidepanelHandoff.modelSaved",
                          "Model preference saved."
                        )
                      )
                    }
                  : current
              )
          } catch {
            if (
              receipt &&
              attempt.isOwned() &&
              optionRevision.current === committedOptionRevision
            )
              setImportReceipt((current) =>
                current === receipt
                  ? {
                      description: receipt.description.replace(
                        t(
                          "playground:sidepanelHandoff.modelSaving",
                          "Saving the model preference…"
                        ),
                        t(
                          "playground:sidepanelHandoff.modelSaveFailed",
                          "The model preference could not be saved. Select the model again to retry saving."
                        )
                      )
                    }
                  : current
              )
          }
        }
      } catch {
        if (isCurrent())
          setConflictFeedback(
            t(
              applyOptions
                ? "playground:sidepanelHandoff.optionsCheckFailed"
                : "playground:sidepanelHandoff.readFailed",
              applyOptions
                ? "The copied settings could not be checked. Your draft and settings remain here. Try again, or uncheck settings to import text only."
                : "The copy could not be checked. Your draft remains here. Try Insert or Replace again."
            )
          )
      } finally {
        if (pendingChoiceRef.current === attempt) {
          pendingChoiceRef.current = null
          setImportPending(false)
        }
      }
    },
    [
      applyImportedHandoff,
      captureGuard,
      cleanRouteHandoff,
      conflict,
      getDraftRevision,
      isConflictCurrent,
      showInvalidFeedback,
      t,
      includeLiteralOptions,
      literalOptions,
      literalOptionsReview.length,
      renderedOptionRevision,
      renderedConsentRevision
    ]
  )

  const applyHandoffOptions = React.useCallback(
    () => acceptHandoffDraft("options"),
    [acceptHandoffDraft]
  )

  const insertHandoffDraft = React.useCallback(
    () => acceptHandoffDraft("insert"), [acceptHandoffDraft],
  )
  const replaceWithHandoffDraft = React.useCallback(
    () => acceptHandoffDraft("replace"), [acceptHandoffDraft],
  )

  const cancelHandoffImport = React.useCallback(() => {
    if (!isConflictCurrent() || !conflict) return
    const isOwned = conflictOwnerRef.current!.isOwned
    conflictOwnerRef.current = null
    pendingChoiceRef.current = null
    setImportPending(false)
    setConflictFeedback(null)
    setConflict(null)
    cleanRouteHandoff(conflict.route)
    handledHandoffIdsRef.current.set(conflict.handoff.id, { destinationKey, settled: true })
    if (!conflict.route.webUi) consumeQuietly(conflict.handoff.id, isOwned)
  }, [cleanRouteHandoff, conflict, consumeQuietly, destinationKey, isConflictCurrent])

  const dismissImportReceipt = React.useCallback(() => {
    setImportReceipt(current => current === importReceipt ? null : current)
  }, [importReceipt])

  return {
    literalOptionsReview,
    includeLiteralOptions,
    setIncludeLiteralOptions,
    applyHandoffOptions,
    importReceipt,
    dismissImportReceipt,
    skippedFeedback: conflict ? skippedFeedbackFor(conflict.handoff, includeLiteralOptions && literalOptionsReview.length > 0) : null,
    importedContext,
    removeImportedContext,
    conflict,
    importPending,
    conflictFeedback,
    insertHandoffDraft,
    replaceWithHandoffDraft,
    cancelHandoffImport,
  }
}
