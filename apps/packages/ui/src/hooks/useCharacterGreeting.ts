import React from "react"
import { generateID } from "@/db/dexie/helpers"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ChatHistory, Message } from "@/store/option/types"
import type { Character } from "@/types/character"
import {
  buildGreetingOptionsFromEntries,
  collectGreetingEntries,
  resolveGreetingSelection,
  type GreetingOption
} from "@/utils/character-greetings"
import { replaceUserDisplayNamePlaceholders } from "@/utils/chat-display-name"
import { useStorage } from "@plasmohq/storage/hook"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import {
  SELECTED_CHARACTER_STORAGE_KEY,
  selectedCharacterStorage,
  parseSelectedCharacterValue
} from "@/utils/selected-character-storage"

type UseCharacterGreetingOptions = {
  playgroundReady: boolean
  selectedCharacter: Character | null
  serverChatId: string | number | null
  historyId: string | null
  messagesLength: number
  setMessages: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])
  ) => void
  setHistory: (
    historyOrUpdater: ChatHistory | ((prev: ChatHistory) => ChatHistory)
  ) => void
  setSelectedCharacter: (next: Character | null) => void
}

export const useCharacterGreeting = ({
  playgroundReady,
  selectedCharacter,
  serverChatId,
  historyId,
  messagesLength,
  setMessages,
  setHistory,
  setSelectedCharacter
}: UseCharacterGreetingOptions) => {
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  const resolvedServerChatId =
    serverChatId != null ? String(serverChatId) : null
  const { settings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId: resolvedServerChatId
  })
  const greetingSelectionId =
    typeof settings?.greetingSelectionId === "string"
      ? settings.greetingSelectionId
      : null
  const greetingsChecksum =
    typeof settings?.greetingsChecksum === "string"
      ? settings.greetingsChecksum
      : null
  const useCharacterDefault = Boolean(settings?.useCharacterDefault)
  const greetingEnabled = settings?.greetingEnabled ?? true
  const greetingInjectedRef = React.useRef<string | null>(null)
  const greetingFetchRef = React.useRef<string | null>(null)
  const greetingFetchedRef = React.useRef<string | null>(null)
  const greetingTemplateRef = React.useRef<{
    characterId: string
    greeting: string
    rendered: string
    avatarUrl: string | null
    selectionId: string | null
    checksum: string | null
  } | null>(null)
  const chatWasEmptyRef = React.useRef(false)
  const selectedCharacterIdRef = React.useRef<string | null>(null)
  const lastCharacterIdRef = React.useRef<string | null>(null)
  const greetingSettingsRef = React.useRef({
    greetingSelectionId,
    greetingsChecksum,
    useCharacterDefault,
    greetingEnabled
  })

  React.useEffect(() => {
    greetingSettingsRef.current = {
      greetingSelectionId,
      greetingsChecksum,
      useCharacterDefault,
      greetingEnabled
    }
  }, [
    greetingEnabled,
    greetingSelectionId,
    greetingsChecksum,
    useCharacterDefault
  ])

  React.useEffect(() => {
    if (!playgroundReady) return
    if (serverChatId != null) return
    let cancelled = false
    const syncSelection = async () => {
      try {
        const storedRaw = await selectedCharacterStorage.get(
          SELECTED_CHARACTER_STORAGE_KEY
        )
        const stored = parseSelectedCharacterValue<Character>(storedRaw)
        if (!stored?.id || cancelled) return
        const storedId = String(stored.id)
        const currentId = selectedCharacter?.id
          ? String(selectedCharacter.id)
          : null
        if (storedId !== currentId) {
          setSelectedCharacter(stored)
        }
      } catch {
        // ignore
      }
    }
    void syncSelection()
    return () => {
      cancelled = true
    }
  }, [playgroundReady, selectedCharacter?.id, serverChatId, setSelectedCharacter])

  React.useEffect(() => {
    const isEmpty = messagesLength === 0
    if (isEmpty && !chatWasEmptyRef.current) {
      greetingInjectedRef.current = null
      greetingTemplateRef.current = null
    }
    chatWasEmptyRef.current = isEmpty
  }, [messagesLength])

  React.useEffect(() => {
    greetingFetchRef.current = null
    greetingFetchedRef.current = null
    greetingTemplateRef.current = null
  }, [selectedCharacter?.id])

  React.useEffect(() => {
    if (!playgroundReady) return
    if (serverChatId != null) {
      selectedCharacterIdRef.current = null
      greetingFetchRef.current = null
      greetingTemplateRef.current = null
      return
    }
    if (!selectedCharacter?.id) {
      selectedCharacterIdRef.current = null
      return
    }

    const characterId = String(selectedCharacter.id)
    selectedCharacterIdRef.current = characterId
    if (
      lastCharacterIdRef.current &&
      lastCharacterIdRef.current !== characterId
    ) {
      void updateSettings({
        greetingSelectionId: null,
        greetingsChecksum: null,
        useCharacterDefault: false
      })
      greetingTemplateRef.current = null
      greetingInjectedRef.current = null
    }
    lastCharacterIdRef.current = characterId
    const characterName = selectedCharacter.name || "Assistant"
    const characterAvatarUrl = selectedCharacter.avatar_url ?? null
    const isCurrentSelection = () =>
      selectedCharacterIdRef.current === characterId

    const upsertGreeting = (
      greetingValue: string,
      avatarUrl?: string | null,
      meta?: { selectionId?: string | null; checksum?: string | null }
    ) => {
      if (!isCurrentSelection()) return
      const rendered = replaceUserDisplayNamePlaceholders(
        greetingValue,
        userDisplayName
      )
      const trimmed = rendered.trim()
      if (!trimmed) return
      const selectionId = meta?.selectionId ?? null
      const checksum = meta?.checksum ?? null
      const normalizedAvatarUrl = avatarUrl ?? null
      const cached = greetingTemplateRef.current
      if (
        greetingInjectedRef.current === characterId &&
        cached?.characterId === characterId &&
        cached.greeting === greetingValue &&
        cached.rendered === trimmed &&
        cached.avatarUrl === normalizedAvatarUrl &&
        cached.selectionId === selectionId &&
        cached.checksum === checksum
      ) {
        return
      }

      const createdAt = Date.now()
      const messageId = generateID()
      let updated = false

      React.startTransition(() => {
        setMessages((prev) => {
          if (!isCurrentSelection()) return prev
          const onlyGreetings =
            prev.length > 0 &&
            prev.every(
              (message) => message.messageType === "character:greeting"
            )
          const singleAssistant = prev.length === 1 && prev[0]?.isBot
          const canReplace =
            prev.length === 0 || onlyGreetings || singleAssistant
          if (!canReplace) return prev
          updated = true
          if (
            prev.length === 1 &&
            prev[0]?.messageType === "character:greeting"
          ) {
            return [
              {
                ...prev[0],
                name: characterName,
                role: "assistant",
                message: trimmed,
                modelName: characterName,
                modelImage: avatarUrl ?? prev[0]?.modelImage
              }
            ]
          }
          return [
            {
              isBot: true,
              name: characterName,
              role: "assistant",
              message: trimmed,
              messageType: "character:greeting",
              sources: [],
              createdAt,
              id: messageId,
              modelName: characterName,
              modelImage: avatarUrl ?? undefined
            }
          ]
        })
      })

      if (!updated) return
      greetingInjectedRef.current = characterId
      greetingTemplateRef.current = {
        characterId,
        greeting: greetingValue,
        rendered: trimmed,
        avatarUrl: normalizedAvatarUrl,
        selectionId,
        checksum
      }

      if (greetingSettingsRef.current.greetingEnabled) {
        React.startTransition(() => {
          setHistory((prev) => {
            if (!isCurrentSelection()) return prev
            const onlyGreetings =
              prev.length > 0 &&
              prev.every((entry) => entry.messageType === "character:greeting")
            const singleAssistant =
              prev.length === 1 && prev[0]?.role === "assistant"
            const canReplace =
              prev.length === 0 || onlyGreetings || singleAssistant
            if (!canReplace) return prev
            if (
              prev.length === 1 &&
              prev[0]?.messageType === "character:greeting"
            ) {
              return [
                {
                  ...prev[0],
                  content: trimmed
                }
              ]
            }
            return [
              {
                role: "assistant",
                content: trimmed,
                messageType: "character:greeting"
              }
            ]
          })
        })
      }
    }

    const resolveAndPersistGreeting = (
      options: GreetingOption[],
      avatarUrl: string | null
    ) => {
      const currentSettings = greetingSettingsRef.current
      const storedSelectionId = currentSettings.greetingSelectionId
      const storedChecksum = currentSettings.greetingsChecksum
      const useCharacterDefaultSetting = currentSettings.useCharacterDefault
      const { option: selectedOption, checksum } = resolveGreetingSelection({
        options,
        greetingSelectionId: storedSelectionId,
        greetingsChecksum: storedChecksum,
        useCharacterDefault: useCharacterDefaultSetting,
        fallback: "random"
      })

      if (!selectedOption) {
        if (storedSelectionId || storedChecksum) {
          void updateSettings({
            greetingSelectionId: null,
            greetingsChecksum: null
          })
        }
        return
      }

      const cached = greetingTemplateRef.current
      if (
        cached?.characterId === characterId &&
        cached.selectionId === selectedOption.id &&
        cached.checksum === checksum
      ) {
        upsertGreeting(cached.greeting, avatarUrl, {
          selectionId: cached.selectionId,
          checksum: cached.checksum
        })
        return
      }

      if (
        storedSelectionId !== selectedOption.id ||
        storedChecksum !== checksum
      ) {
        void updateSettings({
          greetingSelectionId: selectedOption.id,
          greetingsChecksum: checksum
        })
      }

      upsertGreeting(selectedOption.text, avatarUrl, {
        selectionId: selectedOption.id,
        checksum
      })
    }

    const greetingEntries = collectGreetingEntries(selectedCharacter)
    const greetingOptions = buildGreetingOptionsFromEntries(greetingEntries)
    if (greetingOptions.length > 0) {
      resolveAndPersistGreeting(greetingOptions, characterAvatarUrl)
      if (greetingOptions.length > 1) {
        return
      }
    }

    const fallbackGreeting = greetingOptions[0]?.text?.trim() || ""
    if (
      greetingFetchRef.current !== characterId &&
      greetingFetchedRef.current !== characterId
    ) {
      greetingFetchRef.current = characterId
      void (async () => {
        try {
          await tldwClient.initialize().catch(() => null)
          if (
            !isCurrentSelection() ||
            greetingFetchRef.current !== characterId
          ) {
            return
          }
          const full = await tldwClient.getCharacter(characterId)
          greetingFetchedRef.current = characterId
          if (
            !isCurrentSelection() ||
            greetingFetchRef.current !== characterId
          ) {
            return
          }
          const fetchedEntries = collectGreetingEntries(full)
          const resolvedEntries =
            fetchedEntries.length > 0 ? fetchedEntries : greetingEntries
          const resolvedOptions = buildGreetingOptionsFromEntries(
            resolvedEntries
          )
          if (resolvedOptions.length > 0) {
            resolveAndPersistGreeting(resolvedOptions, characterAvatarUrl)
          } else if (
            greetingSettingsRef.current.greetingSelectionId ||
            greetingSettingsRef.current.greetingsChecksum
          ) {
            void updateSettings({
              greetingSelectionId: null,
              greetingsChecksum: null
            })
          }
          const nextAvatar =
            full?.avatar_url ?? selectedCharacter.avatar_url ?? null
          const mergedCharacter = full
            ? {
                ...selectedCharacter,
                ...full,
                avatar_url: nextAvatar
              }
            : {
                ...selectedCharacter,
                avatar_url: nextAvatar
              }
          setSelectedCharacter(mergedCharacter)
        } catch {
          greetingFetchedRef.current = characterId
          if (fallbackGreeting) {
            resolveAndPersistGreeting(
              buildGreetingOptionsFromEntries([
                {
                  text: fallbackGreeting,
                  sourceKey: "greeting",
                  sourceLabel: "Greeting"
                }
              ]),
              characterAvatarUrl
            )
          }
        } finally {
          if (greetingFetchRef.current === characterId) {
            greetingFetchRef.current = null
          }
        }
      })()
    }
  }, [
    playgroundReady,
    selectedCharacter,
    serverChatId,
    historyId,
    setHistory,
    setMessages,
    setSelectedCharacter,
    userDisplayName,
    greetingSelectionId,
    greetingsChecksum,
    useCharacterDefault,
    greetingEnabled,
    updateSettings
  ])
}
