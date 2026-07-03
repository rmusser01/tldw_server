import React from "react"
import { Check, RefreshCcw } from "lucide-react"
import { notification, Select, Switch } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import type { Character } from "@/types/character"
import type { ChatHistory, Message } from "@/store/option/types"
import { generateID } from "@/db/dexie/helpers"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import {
  buildGreetingOptionsFromEntries,
  buildGreetingsChecksumFromOptions,
  collectGreetingEntries,
  isGreetingMessageType,
  parseGreetingSelectionIndex
} from "@/utils/character-greetings"
import { replaceUserDisplayNamePlaceholders } from "@/utils/chat-display-name"

type Props = {
  selectedCharacter: Character | null
  messages: Message[]
  historyId: string | null
  serverChatId: string | null
  setMessages?: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[])
  ) => void
  setHistory?: (
    historyOrUpdater: ChatHistory | ((prev: ChatHistory) => ChatHistory)
  ) => void
  className?: string
}

export const ChatGreetingPicker: React.FC<Props> = ({
  selectedCharacter,
  messages,
  historyId,
  serverChatId,
  setMessages,
  setHistory,
  className
}) => {
  const { t } = useTranslation(["sidepanel", "common", "playground"])
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  const { settings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const [draftSelectionId, setDraftSelectionId] = React.useState<string | null>(
    null
  )
  const greetingPersistInFlightRef = React.useRef(false)

  const hasNonGreetingMessages = React.useMemo(
    () =>
      messages.some(
        (message) => !isGreetingMessageType(message?.messageType)
      ),
    [messages]
  )

  const greetingEntries = React.useMemo(
    () => collectGreetingEntries(selectedCharacter),
    [selectedCharacter]
  )
  const greetingOptions = React.useMemo(
    () => buildGreetingOptionsFromEntries(greetingEntries),
    [greetingEntries]
  )
  const checksum = React.useMemo(
    () =>
      greetingOptions.length > 0
        ? buildGreetingsChecksumFromOptions(greetingOptions)
        : null,
    [greetingOptions]
  )
  const storedSelectionId =
    typeof settings?.greetingSelectionId === "string"
      ? settings.greetingSelectionId
      : null
  const storedChecksum =
    typeof settings?.greetingsChecksum === "string"
      ? settings.greetingsChecksum
      : null
  const greetingEnabled = settings?.greetingEnabled ?? true
  const useCharacterDefault = settings?.useCharacterDefault ?? false

  React.useEffect(() => {
    setDraftSelectionId(null)
  }, [selectedCharacter?.id, storedSelectionId, storedChecksum, useCharacterDefault])

  if (!selectedCharacter?.id) return null
  if (hasNonGreetingMessages) return null
  if (greetingOptions.length === 0) return null

  const resolvedSelection = (() => {
    if (storedChecksum && checksum && storedChecksum !== checksum) {
      return null
    }
    const exactMatch = greetingOptions.find(
      (option) => option.id === storedSelectionId
    )
    if (exactMatch) return exactMatch
    const selectedIndex = parseGreetingSelectionIndex(storedSelectionId)
    if (selectedIndex == null) return null
    return greetingOptions[selectedIndex] || null
  })()
  const draftSelection =
    !useCharacterDefault && draftSelectionId
      ? greetingOptions.find((option) => option.id === draftSelectionId) || null
      : null
  const selectedOption =
    useCharacterDefault && greetingOptions.length > 0
      ? greetingOptions[0]
      : draftSelection || resolvedSelection || greetingOptions[0]
  const selectedOptionId = selectedOption?.id

  const previewText = selectedOption?.text
    ? replaceUserDisplayNamePlaceholders(
        selectedOption.text,
        userDisplayName
      )
    : ""

  const handleReroll = async () => {
    if (greetingOptions.length < 2) return
    const currentId = selectedOption?.id
    const candidates = greetingOptions.filter(
      (option) => option.id !== currentId
    )
    const next =
      candidates[Math.floor(Math.random() * candidates.length)] ||
      greetingOptions[0]
    setDraftSelectionId(next.id)
    await updateSettings({
      greetingSelectionId: next.id,
      greetingsChecksum: checksum,
      useCharacterDefault: false
    })
  }

  const handleSelectGreeting = async (value: string) => {
    setDraftSelectionId(value)
    await updateSettings({
      greetingSelectionId: value,
      greetingsChecksum: checksum,
      useCharacterDefault: false
    })
  }

  const handleUseDefault = async (checked: boolean) => {
    const defaultId = greetingOptions[0]?.id ?? null
    setDraftSelectionId(checked ? null : selectedOption?.id ?? null)
    await updateSettings({
      useCharacterDefault: checked,
      greetingSelectionId: checked
        ? defaultId
        : resolvedSelection?.id ??
          selectedOption?.id ??
          storedSelectionId ??
          null,
      greetingsChecksum: checksum
    })
  }

  const handleToggle = async (checked: boolean) => {
    await updateSettings({ greetingEnabled: checked })
  }

  const handleSelectFirstMessage = async () => {
    if (!selectedOption?.text || !setMessages) return
    if (greetingPersistInFlightRef.current) return
    const rendered = replaceUserDisplayNamePlaceholders(
      selectedOption.text,
      userDisplayName
    )
    const trimmed = rendered.trim()
    if (!trimmed) return

    const createdAt = Date.now()
    const messageId = generateID()
    const characterName = selectedCharacter.name || "Assistant"
    const characterAvatarUrl = selectedCharacter.avatar_url ?? undefined
    const greetingMessage: Message = {
      isBot: true,
      name: characterName,
      role: "assistant",
      message: trimmed,
      messageType: "character:greeting",
      sources: [],
      createdAt,
      id: messageId,
      modelName: characterName,
      modelImage: characterAvatarUrl
    }

    setMessages((prev) => {
      const onlyGreetings =
        prev.length > 0 &&
        prev.every((message) => isGreetingMessageType(message.messageType))
      const singleAssistant = prev.length === 1 && prev[0]?.isBot
      const canReplace = prev.length === 0 || onlyGreetings || singleAssistant
      if (!canReplace) return prev
      if (
        prev.length === 1 &&
        isGreetingMessageType(prev[0]?.messageType)
      ) {
        return [
          {
            ...prev[0],
            name: characterName,
            role: "assistant",
            message: trimmed,
            messageType: "character:greeting",
            modelName: characterName,
            modelImage: characterAvatarUrl ?? prev[0]?.modelImage
          }
        ]
      }
      return [greetingMessage]
    })

    if (greetingEnabled && setHistory) {
      setHistory((prev) => {
        const onlyGreetings =
          prev.length > 0 &&
          prev.every((entry) => isGreetingMessageType(entry.messageType))
        const singleAssistant =
          prev.length === 1 && prev[0]?.role === "assistant"
        const canReplace = prev.length === 0 || onlyGreetings || singleAssistant
        if (!canReplace) return prev
        if (
          prev.length === 1 &&
          isGreetingMessageType(prev[0]?.messageType)
        ) {
          return [
            {
              ...prev[0],
              role: "assistant",
              content: trimmed,
              messageType: "character:greeting"
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
    }

    const existingServerGreeting = messages.find(
      (message) =>
        message?.isBot &&
        isGreetingMessageType(message.messageType) &&
        typeof message.message === "string" &&
        message.message.trim() === trimmed &&
        message.serverMessageId
    )
    if (!serverChatId || existingServerGreeting) return

    greetingPersistInFlightRef.current = true
    try {
      await tldwClient.initialize().catch(() => null)
      const createdGreeting = (await tldwClient.addChatMessage(serverChatId, {
        role: "assistant",
        content: trimmed
      })) as { id?: string | number; version?: number } | null
      if (createdGreeting?.id == null) return
      const serverMessageId = String(createdGreeting.id)
      const serverMessageVersion = createdGreeting.version
      setMessages((prev) => {
        const updated = [...prev]
        for (let index = 0; index < updated.length; index += 1) {
          const message = updated[index]
          if (
            message?.isBot &&
            isGreetingMessageType(message.messageType) &&
            typeof message.message === "string" &&
            message.message.trim() === trimmed &&
            !message.serverMessageId
          ) {
            updated[index] = {
              ...message,
              serverMessageId,
              serverMessageVersion
            }
            break
          }
        }
        return updated
      })
    } catch (error) {
      notification.error({
        message: t("sidepanel:greetingPicker.syncFailed", {
          defaultValue: "Greeting sync failed"
        }),
        description: t("sidepanel:greetingPicker.syncFailedDescription", {
          defaultValue:
            "The greeting was added locally but could not be saved to the server chat."
        })
      })
      console.warn("Failed to persist selected character greeting:", error)
    } finally {
      greetingPersistInFlightRef.current = false
    }
  }

  return (
    <div
      className={`w-full max-w-2xl rounded-2xl border border-border/60 bg-surface/80 p-3 text-xs text-text shadow-sm backdrop-blur ${className || ""}`}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
          {t("sidepanel:greetingPicker.title", { defaultValue: "Greeting" })}
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleReroll}
            disabled={greetingOptions.length < 2}
            className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-surface2 px-2 py-1 text-[11px] text-text-muted transition hover:border-primary/50 hover:text-text disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCcw className="h-3 w-3" />
            {t("sidepanel:greetingPicker.reroll", { defaultValue: "Reroll" })}
          </button>
          <button
            type="button"
            aria-label={t("sidepanel:greetingPicker.selectAria", {
              defaultValue: "Select greeting"
            })}
            onClick={() => {
              void handleSelectFirstMessage()
            }}
            disabled={!selectedOption?.text || !setMessages}
            className="inline-flex items-center gap-1 rounded-full border border-primary/40 bg-primary px-2.5 py-1 text-[11px] font-semibold text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:border-border/70 disabled:bg-surface2 disabled:text-text-muted disabled:opacity-60"
          >
            <Check className="h-3 w-3" />
            {t("sidepanel:greetingPicker.select", { defaultValue: "Select" })}
          </button>
        </div>
      </div>
      <div className="mt-2">
        <div className="mb-1 text-[10px] uppercase tracking-wide text-text-muted">
          {t("sidepanel:greetingPicker.pickLabel", {
            defaultValue: "Pick from list"
          })}
        </div>
        <Select
          value={selectedOptionId}
          onChange={handleSelectGreeting}
          disabled={useCharacterDefault}
          className="w-full"
          size="small"
          optionLabelProp="label"
          options={greetingOptions.map((option) => ({
            value: option.id,
            label: option.text,
            title: option.text,
            option
          }))}
          popupRender={(menu) => <div className="p-1">{menu}</div>}
          optionRender={(option) => {
            const data = (option.data as any)?.option || option.data
            const sourceLabel = data?.sourceLabel
              ? data.sourceLabel
              : t("sidepanel:greetingPicker.sourceUnknown", {
                  defaultValue: "Greeting"
                })
            const lengthLabel = t("sidepanel:greetingPicker.charCount", {
              defaultValue: "{{count}} chars",
              count: data?.text?.length || 0
            })
            return (
              <div className="flex flex-col gap-1">
                <div className="text-[10px] uppercase tracking-wide text-text-muted">
                  {sourceLabel} • {lengthLabel}
                </div>
                <div className="text-xs text-text line-clamp-2">
                  {data?.text}
                </div>
              </div>
            )
          }}
        />
        <div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-text-muted">
          <div>
            {t("sidepanel:greetingPicker.useDefault", {
              defaultValue: "Use character default"
            })}
          </div>
          <Switch
            size="small"
            checked={useCharacterDefault}
            onChange={handleUseDefault}
          />
        </div>
      </div>
      {previewText && (
        <div className="mt-2 rounded-lg border border-border/40 bg-surface2/60 p-2 text-[12px] text-text">
          {previewText}
        </div>
      )}
      <div className="mt-2 flex items-center justify-between gap-3 text-[11px] text-text-muted">
        <div>
          {t("sidepanel:greetingPicker.includeInContext", {
            defaultValue: "Include greeting in context"
          })}
        </div>
        <Switch
          size="small"
          checked={greetingEnabled}
          onChange={handleToggle}
        />
      </div>
    </div>
  )
}
