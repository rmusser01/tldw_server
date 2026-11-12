import React from "react"
import { useTranslation } from "react-i18next"
import type { SlashCommandItem } from "@/components/Sidepanel/Chat/SlashCommandMenu"

/**
 * A parsed composer submission. Resolving one never runs a command: the
 * composer resolves the draft on every keystroke (raw request preview,
 * role-play checks), so commands run only through `executeSlashCommand`
 * when the user actually submits.
 */
export type SlashSubmissionIntent = {
  handled: boolean
  message: string
  imageBackendOverride: string | undefined
  isImageCommand: boolean
  invalidImageCommand: boolean
  imageCommandMissingProvider: boolean
  /** The command a submitted `/command` draft asks to run, if any. */
  slashCommand?: SlashCommandItem
}

/** Run the command a submitted draft resolved to. Call only on submit. */
export const executeSlashCommand = (
  intent: { slashCommand?: SlashCommandItem } | null | undefined
) => {
  intent?.slashCommand?.action?.()
}

export type UseSlashCommandsParams = {
  chatMode: string
  setChatMode: (mode: string) => void
  webSearch: boolean
  setWebSearch: (value: boolean) => void
  handleImageUpload: () => void
  imageBackendDefaultTrimmed: string
  imageBackendLabel: string
  setOpenModelSettings: (open: boolean) => void
  currentMessage: string
}

export function useSlashCommands({
  chatMode,
  setChatMode,
  webSearch,
  setWebSearch,
  handleImageUpload,
  imageBackendDefaultTrimmed,
  imageBackendLabel,
  setOpenModelSettings,
  currentMessage
}: UseSlashCommandsParams) {
  const { t } = useTranslation(["playground", "common"])

  const slashCommands = React.useMemo<SlashCommandItem[]>(
    () => [
      {
        id: "slash-search",
        command: "search",
        label: t(
          "common:commandPalette.toggleKnowledgeSearch",
          "Toggle Search & Context"
        ),
        description: t(
          "common:commandPalette.toggleKnowledgeSearchDesc",
          "Search your knowledge base and context"
        ),
        keywords: ["rag", "context", "knowledge", "search"],
        action: () => setChatMode(chatMode === "rag" ? "normal" : "rag")
      },
      {
        id: "slash-web",
        command: "web",
        label: t(
          "common:commandPalette.toggleWebSearch",
          "Toggle Web Search"
        ),
        description: t(
          "common:commandPalette.toggleWebDesc",
          "Search the internet"
        ),
        keywords: ["web", "internet", "browse"],
        action: () => setWebSearch(!webSearch)
      },
      {
        id: "slash-vision",
        command: "vision",
        label: t("playground:actions.upload", "Attach image"),
        description: t(
          "playground:composer.slashVisionDesc",
          "Attach an image for vision"
        ),
        keywords: ["image", "ocr", "vision"],
        action: handleImageUpload
      },
      {
        id: "slash-generate-image",
        command: "generate-image",
        label: t(
          "playground:composer.slashGenerateImage",
          "Generate image"
        ),
        description: imageBackendDefaultTrimmed
          ? t(
              "playground:composer.slashGenerateImageDescDefault",
              "Generate an image (default: {{backend}}). Use /generate-image:<provider> to override.",
              { backend: imageBackendLabel }
            )
          : t(
              "playground:composer.slashGenerateImageDesc",
              "Generate an image. Use /generate-image:<provider> <prompt>."
            ),
        keywords: ["image", "image gen", "flux", "zturbo", "art"],
        insertText: imageBackendDefaultTrimmed
          ? "/generate-image "
          : "/generate-image:"
      },
      {
        id: "slash-model",
        command: "model",
        label: t("common:commandPalette.switchModel", "Switch Model"),
        description: t(
          "common:currentChatModelSettings",
          "Open current chat settings"
        ),
        keywords: ["settings", "parameters", "temperature"],
        action: () => setOpenModelSettings(true)
      }
    ],
    [
      chatMode,
      handleImageUpload,
      imageBackendDefaultTrimmed,
      imageBackendLabel,
      setChatMode,
      setWebSearch,
      t,
      webSearch,
      setOpenModelSettings
    ]
  )

  const slashCommandLookup = React.useMemo(
    () => new Map(slashCommands.map((command) => [command.command, command])),
    [slashCommands]
  )

  const slashMatch = React.useMemo(
    () => currentMessage.match(/^\s*\/([\w-]*)$/),
    [currentMessage]
  )
  const slashQuery = slashMatch?.[1] ?? ""
  const showSlashMenu = Boolean(slashMatch)
  const [slashActiveIndex, setSlashActiveIndex] = React.useState(0)

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

  React.useEffect(() => {
    if (!showSlashMenu) {
      setSlashActiveIndex(0)
      return
    }
    setSlashActiveIndex((prev) => {
      if (filteredSlashCommands.length === 0) return 0
      return Math.min(prev, filteredSlashCommands.length - 1)
    })
  }, [showSlashMenu, filteredSlashCommands.length, slashQuery])

  const parseSlashInput = React.useCallback((text: string) => {
    const trimmed = text.trimStart()
    const match = trimmed.match(/^\/(\w+)(?:\s+([\s\S]*))?$/)
    if (!match) return null
    return {
      command: match[1].toLowerCase(),
      remainder: match[2] || ""
    }
  }, [])

  const parseImageSlashCommand = React.useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed.toLowerCase().startsWith("/generate-image")) return null
      const remainder = trimmed.slice("/generate-image".length)
      const colonMatch = remainder.match(
        /^\s*:\s*([^\s]+)(?:\s+([\s\S]*))?$/i
      )
      if (colonMatch) {
        const provider = colonMatch[1]?.trim() || ""
        const prompt = (colonMatch[2] || "").trim()
        const missingProvider = provider.length === 0
        return {
          provider,
          prompt,
          invalid: missingProvider,
          missingProvider
        }
      }

      const prompt = remainder.trim()
      if (imageBackendDefaultTrimmed) {
        return {
          provider: imageBackendDefaultTrimmed,
          prompt,
          invalid: false,
          missingProvider: false
        }
      }

      return {
        provider: "",
        prompt,
        invalid: true,
        missingProvider: true
      }
    },
    [imageBackendDefaultTrimmed]
  )

  const parseSlashCommand = React.useCallback(
    (
      text: string
    ): { handled: boolean; message: string; command?: SlashCommandItem } => {
      const parsed = parseSlashInput(text)
      if (!parsed) {
        return { handled: false, message: text }
      }
      const command = slashCommandLookup.get(parsed.command)
      if (!command?.action) {
        return { handled: false, message: text }
      }
      return { handled: true, message: parsed.remainder, command }
    },
    [parseSlashInput, slashCommandLookup]
  )

  // Pure: parses the draft without running the command (see SlashSubmissionIntent).
  const resolveSubmissionIntent = React.useCallback(
    (rawMessage: string): SlashSubmissionIntent => {
      const imageCommand = parseImageSlashCommand(rawMessage)
      if (imageCommand) {
        return {
          handled: true,
          message: imageCommand.prompt,
          imageBackendOverride: imageCommand.provider,
          isImageCommand: true,
          invalidImageCommand: imageCommand.invalid,
          imageCommandMissingProvider: Boolean(imageCommand.missingProvider)
        }
      }
      const slashResult = parseSlashCommand(rawMessage)
      return {
        handled: slashResult.handled,
        message: slashResult.handled ? slashResult.message : rawMessage,
        imageBackendOverride: undefined,
        isImageCommand: false,
        invalidImageCommand: false,
        imageCommandMissingProvider: false,
        ...(slashResult.command ? { slashCommand: slashResult.command } : {})
      }
    },
    [parseSlashCommand, parseImageSlashCommand]
  )

  const activeImageCommand = React.useMemo(
    () => Boolean(parseImageSlashCommand(currentMessage)),
    [currentMessage, parseImageSlashCommand]
  )

  const handleSlashCommandSelect = React.useCallback(
    (command: SlashCommandItem, formSetFieldValue: (field: string, value: string) => void, textareaRef: React.RefObject<HTMLTextAreaElement | null>) => {
      const parsed = parseSlashInput(currentMessage)
      if (command.insertText) {
        formSetFieldValue("message", command.insertText)
        requestAnimationFrame(() => textareaRef.current?.focus())
        return
      }
      command.action?.()
      formSetFieldValue("message", parsed?.remainder || "")
      requestAnimationFrame(() => textareaRef.current?.focus())
    },
    [currentMessage, parseSlashInput]
  )

  return {
    slashCommands,
    slashCommandLookup,
    slashMatch,
    slashQuery,
    showSlashMenu,
    slashActiveIndex,
    setSlashActiveIndex,
    filteredSlashCommands,
    parseSlashInput,
    parseImageSlashCommand,
    parseSlashCommand,
    resolveSubmissionIntent,
    activeImageCommand,
    handleSlashCommandSelect
  }
}
