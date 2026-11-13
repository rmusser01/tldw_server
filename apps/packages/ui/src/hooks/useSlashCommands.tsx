import React from "react"
import { useTranslation } from "react-i18next"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useStorage } from "@plasmohq/storage/hook"
import {
  buildServerSlashCommands,
  mergeSlashCommands,
  normalizeSlashCommandName,
  useServerSlashCommandPayload,
  type SlashCommandMeta
} from "@/hooks/chat/server-slash-commands"
import type { SlashCommandItem } from "@/types/slash-command"

const SLASH_MENU_REGEX = /^\s*\/([\w-]*)$/
const SLASH_COMMAND_REGEX = /^\/+([^\s/]+)(?:\s+([\s\S]*))?$/

export type { SlashCommandItem }

interface UseSlashCommandsOptions {
  chatMode: string
  webSearch: boolean
  setChatMode: (mode: string) => void
  setWebSearch: (enabled: boolean) => void
  onOpenModelSettings: () => void
  inputValue: string
  setInputValue?: (value: string) => void
}

interface UseSlashCommandsResult {
  slashCommands: SlashCommandItem[]
  filteredSlashCommands: SlashCommandItem[]
  slashMatch: RegExpMatchArray | null
  slashQuery: string
  showSlashMenu: boolean
  slashActiveIndex: number
  setSlashActiveIndex: React.Dispatch<React.SetStateAction<number>>
  slashCommandLookup: Map<string, SlashCommandItem>
  applySlashCommand: (text: string) => { handled: boolean; message: string }
  handleSlashCommandSelect: (command: SlashCommandItem) => void
}

const normalizeCommand = normalizeSlashCommandName

/**
 * Hook for managing slash command autocomplete in the chat composer.
 * Server commands are fetched from /api/v1/chat/commands and filtered by permissions.
 */
export const useSlashCommands = ({
  chatMode,
  webSearch,
  setChatMode,
  setWebSearch,
  onOpenModelSettings,
  inputValue,
  setInputValue
}: UseSlashCommandsOptions): UseSlashCommandsResult => {
  const { t } = useTranslation(["common", "sidepanel"])
  const isOnline = useServerOnline()
  const [imageBackendDefault] = useStorage("imageBackendDefault", "")
  const imageBackendDefaultTrimmed = React.useMemo(
    () => (imageBackendDefault || "").trim(),
    [imageBackendDefault]
  )
  const [slashActiveIndex, setSlashActiveIndex] = React.useState(0)
  const serverPayload = useServerSlashCommandPayload(isOnline)

  const fallbackMeta = React.useMemo<Record<string, SlashCommandMeta>>(
    () => ({
      search: {
        label: t(
          "common:commandPalette.toggleKnowledgeSearch",
          "Toggle Knowledge Search"
        ),
        description: t(
          "common:commandPalette.toggleKnowledgeSearchDesc",
          "Search your knowledge base"
        ),
        keywords: ["rag", "context", "knowledge", "search"]
      },
      web: {
        label: t("common:commandPalette.toggleWebSearch", "Toggle Web Search"),
        description: t(
          "common:commandPalette.toggleWebDesc",
          "Search the internet"
        ),
        keywords: ["web", "internet", "browse"]
      },
      vision: {
        label: t("sidepanel:controlRow.vision", "Vision"),
        description: t(
          "sidepanel:controlRow.visionTooltip",
          "Enable Vision to analyze images"
        ),
        keywords: ["image", "ocr", "vision"]
      },
      model: {
        label: t("common:commandPalette.switchModel", "Switch Model"),
        description: t(
          "common:currentChatModelSettings",
          "Open current chat settings"
        ),
        keywords: ["settings", "parameters", "temperature"]
      },
      "generate-image": {
        label: t("sidepanel:composer.generateImage", "Generate image"),
        description: imageBackendDefaultTrimmed
          ? t(
              "sidepanel:composer.generateImageDescDefault",
              "Generate an image (default: {{backend}}). Use /generate-image:<provider> to override.",
              { backend: imageBackendDefaultTrimmed }
            )
          : t(
              "sidepanel:composer.generateImageDesc",
              "Generate an image via /generate-image:<provider>"
            ),
        keywords: ["image", "generate", "flux", "zturbo"],
        insertText: imageBackendDefaultTrimmed
          ? "/generate-image "
          : "/generate-image:"
      }
    }),
    [imageBackendDefaultTrimmed, t]
  )

  const localActions = React.useMemo(
    () => ({
      search: () => setChatMode(chatMode === "rag" ? "normal" : "rag"),
      web: () => setWebSearch(!webSearch),
      vision: () => setChatMode(chatMode === "vision" ? "normal" : "vision"),
      model: onOpenModelSettings
    }),
    [chatMode, onOpenModelSettings, setChatMode, setWebSearch, webSearch]
  )

  const fallbackCommands = React.useMemo<SlashCommandItem[]>(
    () =>
      Object.entries(fallbackMeta).map(([command, meta]) => ({
        id: `slash-${command}`,
        command,
        label: meta.label,
        description: meta.description,
        keywords: meta.keywords,
        insertText: meta.insertText,
        action: localActions[command as keyof typeof localActions],
        source: "local"
      })),
    [fallbackMeta, localActions]
  )

  const serverCommands = React.useMemo<SlashCommandItem[]>(
    () =>
      buildServerSlashCommands(
        serverPayload,
        fallbackMeta,
        localActions as Record<string, (() => void) | undefined>
      ),
    [fallbackMeta, localActions, serverPayload]
  )

  const slashCommands = React.useMemo(
    () => (isOnline ? mergeSlashCommands(serverCommands, fallbackCommands) : fallbackCommands),
    [fallbackCommands, isOnline, serverCommands]
  )

  const slashCommandLookup = React.useMemo(
    () => new Map(slashCommands.map((command) => [command.command, command])),
    [slashCommands]
  )

  // Match incomplete slash commands for autocomplete (allows just "/").
  const slashMatch = React.useMemo(
    () => inputValue.match(SLASH_MENU_REGEX),
    [inputValue]
  )

  const slashQuery = slashMatch?.[1] ?? ""
  const showSlashMenu = Boolean(slashMatch)

  const filteredSlashCommands = React.useMemo(() => {
    if (!slashQuery) return slashCommands
    const q = slashQuery.toLowerCase()
    return slashCommands.filter((command) => {
      if (command.command.startsWith(q)) return true
      if (command.label.toLowerCase().includes(q)) return true
      return (command.keywords || []).some((keyword) =>
        keyword.toLowerCase().includes(q)
      )
    })
  }, [slashCommands, slashQuery])

  // Reset or clamp active index when menu state or filtered list changes.
  React.useEffect(() => {
    if (!showSlashMenu || filteredSlashCommands.length === 0) {
      setSlashActiveIndex(0)
      return
    }
    setSlashActiveIndex((prev) =>
      Math.min(prev, filteredSlashCommands.length - 1)
    )
  }, [filteredSlashCommands.length, showSlashMenu])

  const parseSlashInput = React.useCallback((text: string) => {
    const trimmed = text.trimStart()
    const match = trimmed.match(SLASH_COMMAND_REGEX)
    if (!match) return null
    return {
      command: normalizeCommand(match[1]),
      remainder: match[2] || ""
    }
  }, [])

  const applySlashCommand = React.useCallback(
    (text: string) => {
      const parsed = parseSlashInput(text)
      if (!parsed) {
        return { handled: false, message: text }
      }
      const command = slashCommandLookup.get(parsed.command)
      if (!command) {
        return { handled: false, message: text }
      }
      if (command.action) {
        try {
          command.action()
          return { handled: true, message: parsed.remainder }
        } catch (error) {
          console.error("Slash command action failed:", error)
          return { handled: false, message: text }
        }
      }
      // No local handler; let the backend interpret the slash command.
      return { handled: false, message: text }
    },
    [parseSlashInput, slashCommandLookup]
  )

  const handleSlashCommandSelect = React.useCallback(
    (command: SlashCommandItem) => {
      if (command.insertText && setInputValue) {
        setInputValue(command.insertText)
        return
      }
      if (command.action) {
        try {
          command.action()
          if (setInputValue) {
            const parsed = parseSlashInput(inputValue)
            setInputValue(parsed?.remainder || "")
          }
          return
        } catch (error) {
          console.error("Slash command action failed:", error)
          return
        }
      }
      if (setInputValue) {
        setInputValue(`/${command.command} `)
      }
    },
    [inputValue, parseSlashInput, setInputValue]
  )

  return {
    slashCommands,
    filteredSlashCommands,
    slashMatch,
    slashQuery,
    showSlashMenu,
    slashActiveIndex,
    setSlashActiveIndex,
    slashCommandLookup,
    applySlashCommand,
    handleSlashCommandSelect
  }
}

/**
 * Helper hook to sync input value for slash command detection.
 * Use this in combination with useSlashCommands.
 */
export const useSlashCommandInput = (
  inputValue: string,
  slashCommands: SlashCommandItem[]
) => {
  const slashMatch = React.useMemo(
    () => inputValue.match(SLASH_MENU_REGEX),
    [inputValue]
  )

  const slashQuery = slashMatch?.[1] ?? ""
  const showSlashMenu = Boolean(slashMatch)

  const filteredSlashCommands = React.useMemo(() => {
    if (!slashQuery) return slashCommands
    const q = slashQuery.toLowerCase()
    return slashCommands.filter((command) => {
      if (command.command.startsWith(q)) return true
      if (command.label.toLowerCase().includes(q)) return true
      return (command.keywords || []).some((keyword) =>
        keyword.toLowerCase().includes(q)
      )
    })
  }, [slashCommands, slashQuery])

  return {
    slashMatch,
    slashQuery,
    showSlashMenu,
    filteredSlashCommands
  }
}
