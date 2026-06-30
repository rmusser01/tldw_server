import React from "react"
import {
  composeConversationContext,
  type ConversationContextPrimitiveClient
} from "@/services/conversation-context/conversationContextComposer"
import {
  buildConversationContextSettingsPatch,
  resolveConversationContextSelection
} from "@/services/conversation-context/conversationContextSettings"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type {
  ConversationContextComposition,
  ConversationContextSelection,
  ConversationContextSettingsPatch
} from "@/types/conversation-context"
import type { ChatHistory } from "@/store/option"

export type ConversationContextCompositionStatus =
  | "idle"
  | "loading"
  | "ready"
  | "error"

export type ConversationContextSendOverrides = {
  historyForModel: ChatHistory
  messageForModel: string
}

export type ConversationContextSendComposition = {
  composition: ConversationContextComposition
  authoredMessage: string
  requestOverrides: ConversationContextSendOverrides
}

export type UseConversationContextCompositionParams = {
  draftMessage: string
  selection: ConversationContextSelection
  settings?: Record<string, unknown> | null
  inheritedWorldBookIds?: Array<number | string | null | undefined>
  primitives?: ConversationContextPrimitiveClient
  enabled?: boolean
  debounceMs?: number
  updateSettings?: (
    patch: ConversationContextSettingsPatch
  ) => Promise<unknown>
}

type RawContextIdList = Array<number | string | null | undefined>

const parseContextIdKey = (key: string): RawContextIdList =>
  JSON.parse(key) as RawContextIdList

const defaultPrimitives: ConversationContextPrimitiveClient = {
  processDictionary: (request, options) =>
    tldwClient.processDictionary(request, options),
  processWorldBookContext: (request, options) =>
    tldwClient.processWorldBookContext(request, options)
}

export const buildConversationContextSendOverrides = ({
  composition,
  history
}: {
  composition: ConversationContextComposition
  history: ChatHistory
}): ConversationContextSendOverrides => {
  const contextMessages = composition.providerMessages
    .filter((message) => message.role === "system")
    .map((message) => ({
      role: "system" as const,
      content: message.content
    }))

  return {
    historyForModel: [...history, ...contextMessages],
    messageForModel: composition.transformedInputText
  }
}

export const useConversationContextComposition = ({
  draftMessage,
  selection,
  settings,
  inheritedWorldBookIds,
  primitives = defaultPrimitives,
  enabled = true,
  debounceMs = 0,
  updateSettings
}: UseConversationContextCompositionParams) => {
  const [composition, setComposition] =
    React.useState<ConversationContextComposition | null>(null)
  const [status, setStatus] =
    React.useState<ConversationContextCompositionStatus>("idle")
  const [error, setError] = React.useState<unknown>(null)
  const selectionWorldBookKey = React.useMemo(
    () => JSON.stringify(selection.worldBookIds ?? []),
    [selection.worldBookIds]
  )
  const selectionDictionaryKey = React.useMemo(
    () => JSON.stringify(selection.dictionaryIds ?? []),
    [selection.dictionaryIds]
  )
  const inheritedWorldBookKey = React.useMemo(
    () => JSON.stringify(inheritedWorldBookIds ?? []),
    [inheritedWorldBookIds]
  )
  const selectionWorldBookIds = React.useMemo(
    () => parseContextIdKey(selectionWorldBookKey),
    [selectionWorldBookKey]
  )
  const selectionDictionaryIds = React.useMemo(
    () => parseContextIdKey(selectionDictionaryKey),
    [selectionDictionaryKey]
  )
  const resolvedInheritedWorldBookIds = React.useMemo(
    () => parseContextIdKey(inheritedWorldBookKey),
    [inheritedWorldBookKey]
  )

  const resolvedSelection = React.useMemo(
    () =>
      resolveConversationContextSelection({
        settings,
        seed: {
          chatId: selection.chatId,
          characterId: selection.characterId,
          worldBookIds: selectionWorldBookIds,
          dictionaryIds: selectionDictionaryIds,
          workspaceId: selection.workspaceId,
          providerId: selection.providerId,
          modelId: selection.modelId
        }
      }),
    [
      selection.chatId,
      selection.characterId,
      selection.modelId,
      selection.providerId,
      selectionDictionaryIds,
      selectionWorldBookIds,
      selection.workspaceId,
      settings
    ]
  )

  const compose = React.useCallback(
    (message: string, signal?: AbortSignal) =>
      composeConversationContext({
        inputText: message,
        selection: resolvedSelection,
        inheritedWorldBookIds: resolvedInheritedWorldBookIds,
        primitives,
        signal
      }),
    [primitives, resolvedInheritedWorldBookIds, resolvedSelection]
  )

  const refresh = React.useCallback(async () => {
    if (!enabled) {
      setStatus("idle")
      setComposition(null)
      setError(null)
      return null
    }
    setStatus("loading")
    setError(null)
    try {
      const next = await compose(draftMessage)
      setComposition(next)
      setStatus("ready")
      return next
    } catch (nextError) {
      setComposition(null)
      setError(nextError)
      setStatus("error")
      return null
    }
  }, [compose, draftMessage, enabled])

  React.useEffect(() => {
    let cancelled = false
    const controller = new AbortController()
    if (!enabled) {
      setStatus("idle")
      setComposition(null)
      setError(null)
      return
    }

    const runComposition = () => {
      compose(draftMessage, controller.signal)
        .then((next) => {
          if (cancelled) return
          setComposition(next)
          setStatus("ready")
        })
        .catch((nextError) => {
          if (cancelled || controller.signal.aborted) return
          setComposition(null)
          setError(nextError)
          setStatus("error")
        })
    }

    setStatus("loading")
    setError(null)
    const timeout =
      debounceMs > 0 ? setTimeout(runComposition, debounceMs) : null
    if (timeout === null) runComposition()

    return () => {
      cancelled = true
      controller.abort()
      if (timeout !== null) clearTimeout(timeout)
    }
  }, [compose, debounceMs, draftMessage, enabled])

  const composeForSend = React.useCallback(
    async ({
      message,
      history
    }: {
      message: string
      history: ChatHistory
    }): Promise<ConversationContextSendComposition> => {
      const activeComposition =
        composition &&
        status === "ready" &&
        error == null &&
        message === draftMessage &&
        composition.inputText === message
          ? composition
          : await compose(message)
      return {
        composition: activeComposition,
        authoredMessage: message,
        requestOverrides: buildConversationContextSendOverrides({
          composition: activeComposition,
          history
        })
      }
    },
    [compose, composition, draftMessage, error, status]
  )

  const saveSelection = React.useCallback(
    async (
      nextSelection: Pick<
        ConversationContextSelection,
        "worldBookIds" | "dictionaryIds"
      >
    ) => {
      if (!updateSettings) return null
      return await updateSettings(
        buildConversationContextSettingsPatch({
          worldBookIds: nextSelection.worldBookIds,
          dictionaryIds: nextSelection.dictionaryIds
        })
      )
    },
    [updateSettings]
  )

  return {
    composition,
    status,
    error,
    selection: resolvedSelection,
    refresh,
    composeForSend,
    saveSelection
  }
}
