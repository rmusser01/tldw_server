import { useCallback, useMemo, useRef, type ChangeEvent } from "react"
import { Input, Select, Switch, Tooltip } from "antd"
import { useStorage } from "@plasmohq/storage/hook"
import { useTranslation } from "react-i18next"
import {
  DEFAULT_CHAT_SETTINGS,
  type ChatRichTextColorOption,
  type ChatRichTextFontOption,
  type ChatRichTextMode,
  type ChatRichTextStylePreset
} from "@/types/chat-settings"
import { BetaTag } from "@/components/Common/Beta"
import { SettingRow } from "@/components/Common/SettingRow"
import { useChatSettings } from "@/hooks/useChatSettings"
import { useSetting } from "@/hooks/useSetting"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { toBase64 } from "@/libs/to-base64"
import {
  CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH,
  CHAT_BACKGROUND_IMAGE_MAX_SIZE_MB,
  CHAT_BACKGROUND_IMAGE_SETTING
} from "@/services/settings/ui-settings"
import { RotateCcw, Upload } from "lucide-react"
import { DiscoSkillsSettings } from "./DiscoSkillsSettings"
import { QuickChatWorkflowGuidesSettings } from "./QuickChatWorkflowGuidesSettings"
import Markdown from "@/components/Common/Markdown"
import {
  CHAT_RICH_TEXT_STYLE_PRESETS,
  normalizeChatRichTextStylePreset
} from "@/utils/chat-rich-text-style"
import {
  normalizeQuickChatDocsMediaIds,
  QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE,
  toQuickChatDocsMediaIdsInputValue
} from "@/components/Common/QuickChatHelper/docs-rag-profile"
import { ComposerStyleSettings } from "@/components/Chat/composer/settings/ComposerStyleSettings"

const SELECT_CLASSNAME = "w-[200px]"
export const ChatSettings = () => {
  const { t } = useTranslation("settings")
  const notification = useAntdNotification()
  const [chatBackgroundImage, setChatBackgroundImage] = useSetting(
    CHAT_BACKGROUND_IMAGE_SETTING
  )
  const [quickChatStrictDocsOnly, setQuickChatStrictDocsOnly] = useStorage<boolean>(
    "quickChatStrictDocsOnly",
    true
  )
  const [quickChatDocsNamespace, setQuickChatDocsNamespace] = useStorage<string>(
    "quickChatDocsIndexNamespace",
    QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE
  )
  const [quickChatDocsMediaIdsRaw, setQuickChatDocsMediaIdsRaw] =
    useStorage<unknown>("quickChatDocsProjectMediaIds", [])
  const chatBackgroundInputRef = useRef<HTMLInputElement | null>(null)
  const quickChatStrictEnabled = quickChatStrictDocsOnly !== false
  const quickChatNamespaceInputValue =
    typeof quickChatDocsNamespace === "string" ? quickChatDocsNamespace : ""
  const quickChatEffectiveNamespace =
    quickChatNamespaceInputValue.trim() || QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE
  const quickChatMediaIdsInputValue = useMemo(
    () => toQuickChatDocsMediaIdsInputValue(quickChatDocsMediaIdsRaw),
    [quickChatDocsMediaIdsRaw]
  )
  const quickChatParsedMediaIds = useMemo(
    () => normalizeQuickChatDocsMediaIds(quickChatDocsMediaIdsRaw),
    [quickChatDocsMediaIdsRaw]
  )

  const {
    copilotResumeLastChat,
    setCopilotResumeLastChat,
    defaultChatWithWebsite,
    setDefaultChatWithWebsite,
    webUIResumeLastChat,
    setWebUIResumeLastChat,
    hideCurrentChatModelSettings,
    setHideCurrentChatModelSettings,
    hideQuickChatHelper,
    setHideQuickChatHelper,
    restoreLastChatModel,
    setRestoreLastChatModel,
    generateTitle,
    setGenerateTitle,
    checkWideMode,
    setCheckWideMode,
    stickyChatInput,
    setStickyChatInput,
    menuDensity,
    setMenuDensity,
    openReasoning,
    setOpenReasoning,
    userChatBubble,
    setUserChatBubble,
    autoCopyResponseToClipboard,
    setAutoCopyResponseToClipboard,
    useMarkdownForUserMessage,
    setUseMarkdownForUserMessage,
    renderMermaidDiagrams,
    setRenderMermaidDiagrams,
    chatRichTextMode,
    setChatRichTextMode,
    chatRichTextStylePreset,
    setChatRichTextStylePreset,
    chatRichItalicColor,
    setChatRichItalicColor,
    chatRichItalicFont,
    setChatRichItalicFont,
    chatRichBoldColor,
    setChatRichBoldColor,
    chatRichBoldFont,
    setChatRichBoldFont,
    chatRichQuoteTextColor,
    setChatRichQuoteTextColor,
    chatRichQuoteFont,
    setChatRichQuoteFont,
    chatRichQuoteBorderColor,
    setChatRichQuoteBorderColor,
    chatRichQuoteBackgroundColor,
    setChatRichQuoteBackgroundColor,
    copyAsFormattedText,
    setCopyAsFormattedText,
    allowExternalImages,
    setAllowExternalImages,
    tabMentionsEnabled,
    setTabMentionsEnabled,
    pasteLargeTextAsFile,
    setPasteLargeTextAsFile,
    sidepanelTemporaryChat,
    setSidepanelTemporaryChat,
    removeReasoningTagFromCopy,
    setRemoveReasoningTagFromCopy,
    promptSearchIncludeServer,
    setPromptSearchIncludeServer,
    userTextColor,
    setUserTextColor,
    assistantTextColor,
    setAssistantTextColor,
    userTextFont,
    setUserTextFont,
    assistantTextFont,
    setAssistantTextFont,
    userTextSize,
    setUserTextSize,
    assistantTextSize,
    setAssistantTextSize
  } = useChatSettings()

  const colorOptions = useMemo(
    () => [
      {
        value: "default",
        label: t("chatAppearance.color.default", "Default")
      },
      { value: "blue", label: t("chatAppearance.color.blue", "Blue") },
      { value: "green", label: t("chatAppearance.color.green", "Green") },
      { value: "purple", label: t("chatAppearance.color.purple", "Purple") },
      { value: "orange", label: t("chatAppearance.color.orange", "Orange") },
      { value: "red", label: t("chatAppearance.color.red", "Red") }
    ],
    [t]
  )

  const fontOptions = useMemo(
    () => [
      {
        value: "default",
        label: t("chatAppearance.font.default", "Default")
      },
      { value: "sans", label: t("chatAppearance.font.sans", "Sans serif") },
      { value: "serif", label: t("chatAppearance.font.serif", "Serif") },
      { value: "mono", label: t("chatAppearance.font.mono", "Monospace") }
    ],
    [t]
  )

  const sizeOptions = useMemo(
    () => [
      { value: "sm", label: t("chatAppearance.size.sm", "Small") },
      { value: "md", label: t("chatAppearance.size.md", "Medium") },
      { value: "lg", label: t("chatAppearance.size.lg", "Large") }
    ],
    [t]
  )

  const menuDensityOptions = useMemo(
    () => [
      {
        value: "comfortable",
        label: t(
          "generalSettings.settings.menuDensity.comfortable",
          "Comfortable"
        )
      },
      {
        value: "compact",
        label: t("generalSettings.settings.menuDensity.compact", "Compact")
      }
    ],
    [t]
  )

  const richTextModeOptions = useMemo(
    () => [
      {
        value: "safe_markdown",
        label: t(
          "generalSettings.settings.chatRichTextMode.safe",
          "Safe Markdown (default)"
        )
      },
      {
        value: "st_compat",
        label: t(
          "generalSettings.settings.chatRichTextMode.stCompat",
          "SillyTavern-compatible"
        )
      }
    ],
    [t]
  )

  const richTextStylePresetOptions = useMemo(
    () => [
      {
        value: "default",
        label: t(
          "generalSettings.settings.chatRichTextStyles.presets.default",
          "Default (recommended)"
        )
      },
      {
        value: "muted",
        label: t(
          "generalSettings.settings.chatRichTextStyles.presets.muted",
          "Muted"
        )
      },
      {
        value: "high_contrast",
        label: t(
          "generalSettings.settings.chatRichTextStyles.presets.highContrast",
          "High contrast"
        )
      },
      {
        value: "custom",
        label: t(
          "generalSettings.settings.chatRichTextStyles.presets.custom",
          "Custom"
        )
      }
    ],
    [t]
  )

  const richTextColorOptions = useMemo(
    () => [
      {
        value: "default",
        label: t("chatAppearance.color.default", "Default")
      },
      {
        value: "text",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.text",
          "Text"
        )
      },
      {
        value: "muted",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.muted",
          "Muted"
        )
      },
      {
        value: "primary",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.primary",
          "Primary"
        )
      },
      {
        value: "accent",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.accent",
          "Accent"
        )
      },
      {
        value: "success",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.success",
          "Success"
        )
      },
      {
        value: "warn",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.warn",
          "Warning"
        )
      },
      {
        value: "danger",
        label: t(
          "generalSettings.settings.chatRichTextStyles.colors.danger",
          "Danger"
        )
      }
    ],
    [t]
  )

  const richTextPreviewSample = useMemo(
    () =>
      t(
        "generalSettings.settings.chatRichTextMode.previewSample",
        "Line one\nLine two with ||inline spoiler||\n\n[spoiler]Block spoiler content[/spoiler]"
      ),
    [t]
  )

  const handleMenuDensityChange = useCallback(
    (value: string) => setMenuDensity(value as "comfortable" | "compact"),
    [setMenuDensity]
  )

  const handleUserTextSizeChange = useCallback(
    (value: string) => setUserTextSize(value as "sm" | "md" | "lg"),
    [setUserTextSize]
  )

  const handleRichTextModeChange = useCallback(
    (value: string) => setChatRichTextMode(value as ChatRichTextMode),
    [setChatRichTextMode]
  )

  const applyRichTextPreset = useCallback(
    (preset: Exclude<ChatRichTextStylePreset, "custom">) => {
      const tokens = CHAT_RICH_TEXT_STYLE_PRESETS[preset]
      if (!tokens) return
      setChatRichTextStylePreset(preset)
      setChatRichItalicColor(tokens.chatRichItalicColor)
      setChatRichItalicFont(tokens.chatRichItalicFont)
      setChatRichBoldColor(tokens.chatRichBoldColor)
      setChatRichBoldFont(tokens.chatRichBoldFont)
      setChatRichQuoteTextColor(tokens.chatRichQuoteTextColor)
      setChatRichQuoteFont(tokens.chatRichQuoteFont)
      setChatRichQuoteBorderColor(tokens.chatRichQuoteBorderColor)
      setChatRichQuoteBackgroundColor(tokens.chatRichQuoteBackgroundColor)
    },
    [
      setChatRichBoldColor,
      setChatRichBoldFont,
      setChatRichItalicColor,
      setChatRichItalicFont,
      setChatRichQuoteBackgroundColor,
      setChatRichQuoteBorderColor,
      setChatRichQuoteFont,
      setChatRichQuoteTextColor,
      setChatRichTextStylePreset
    ]
  )

  const handleRichTextPresetChange = useCallback(
    (value: string) => {
      const preset = normalizeChatRichTextStylePreset(value)
      if (preset === "custom") {
        setChatRichTextStylePreset("custom")
        return
      }
      applyRichTextPreset(preset)
    },
    [applyRichTextPreset, setChatRichTextStylePreset]
  )

  const markRichTextStyleAsCustom = useCallback(() => {
    if (chatRichTextStylePreset !== "custom") {
      setChatRichTextStylePreset("custom")
    }
  }, [chatRichTextStylePreset, setChatRichTextStylePreset])

  const handleRichTextColorChange = useCallback(
    (
      setter: (next: ChatRichTextColorOption) => void | Promise<void>,
      value: string
    ) => {
      setter(value as ChatRichTextColorOption)
      markRichTextStyleAsCustom()
    },
    [markRichTextStyleAsCustom]
  )

  const handleRichTextFontChange = useCallback(
    (
      setter: (next: ChatRichTextFontOption) => void | Promise<void>,
      value: string
    ) => {
      setter(value as ChatRichTextFontOption)
      markRichTextStyleAsCustom()
    },
    [markRichTextStyleAsCustom]
  )

  const resetRichTextStyles = useCallback(() => {
    applyRichTextPreset("default")
  }, [applyRichTextPreset])

  const handleAssistantTextSizeChange = useCallback(
    (value: string) => setAssistantTextSize(value as "sm" | "md" | "lg"),
    [setAssistantTextSize]
  )

  const resetChatBackgroundImage = useCallback(() => {
    void setChatBackgroundImage(undefined)
  }, [setChatBackgroundImage])

  const handleChatBackgroundImageUpload = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0]
      event.target.value = ""
      if (!file) return

      if (!file.type.startsWith("image/")) {
        notification.error({
          message: t(
            "systemNotifications.invalidImage",
            "Please select a valid image file"
          )
        })
        return
      }

      try {
        const base64String = await toBase64(file)
        if (base64String.length > CHAT_BACKGROUND_IMAGE_MAX_BASE64_LENGTH) {
          notification.error({
            message: t("chatBackground.tooLargeTitle", "Image too large"),
            description: t(
              "chatBackground.tooLargeDescription",
              `Please choose a smaller image (around ${CHAT_BACKGROUND_IMAGE_MAX_SIZE_MB} MB or less) for the chat background. Try compressing or resizing it and upload again.`
            )
          })
          return
        }

        await setChatBackgroundImage(base64String)
      } catch (error) {
        console.error("Error uploading chat background image:", error)
        notification.error({
          message: t("storage.writeError", "Could not save settings"),
          description: t(
            "storage.writeErrorDescription",
            "We couldn't save your settings. Please try again shortly."
          )
        })
      }
    },
    [notification, setChatBackgroundImage, t]
  )

  const getResetProps = <T extends boolean | string>(
    value: T,
    defaultValue: T,
    setter: (next: T | ((prev: T) => T)) => void | Promise<void>
  ) => ({
    modified: value !== defaultValue,
    onReset: () => void setter(defaultValue)
  })

  return (
    <div className="flex flex-col space-y-6 text-sm">
      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("chatBehavior.title", "Chat behavior")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "chatBehavior.description",
            "Control chat defaults, layout, and message handling."
          )}
        </p>
        <div className="border-b border-border mt-3" />
      </div>

      <ComposerStyleSettings />

      <div className="border-b border-border" />

      <SettingRow
        label={t("generalSettings.settings.copilotResumeLastChat.label")}
        {...getResetProps(
          copilotResumeLastChat,
          DEFAULT_CHAT_SETTINGS.copilotResumeLastChat,
          setCopilotResumeLastChat
        )}
        control={
          <Switch
            checked={copilotResumeLastChat}
            onChange={setCopilotResumeLastChat}
            aria-label={t("generalSettings.settings.copilotResumeLastChat.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.turnOnChatWithWebsite.label")}
        {...getResetProps(
          defaultChatWithWebsite,
          DEFAULT_CHAT_SETTINGS.defaultChatWithWebsite,
          setDefaultChatWithWebsite
        )}
        control={
          <Switch
            checked={defaultChatWithWebsite}
            onChange={setDefaultChatWithWebsite}
            aria-label={t("generalSettings.settings.turnOnChatWithWebsite.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.webUIResumeLastChat.label")}
        {...getResetProps(
          webUIResumeLastChat,
          DEFAULT_CHAT_SETTINGS.webUIResumeLastChat,
          setWebUIResumeLastChat
        )}
        control={
          <Switch
            checked={webUIResumeLastChat}
            onChange={setWebUIResumeLastChat}
            aria-label={t("generalSettings.settings.webUIResumeLastChat.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.hideCurrentChatModelSettings.label")}
        {...getResetProps(
          hideCurrentChatModelSettings,
          DEFAULT_CHAT_SETTINGS.hideCurrentChatModelSettings,
          setHideCurrentChatModelSettings
        )}
        control={
          <Switch
            checked={hideCurrentChatModelSettings}
            onChange={setHideCurrentChatModelSettings}
            aria-label={t(
              "generalSettings.settings.hideCurrentChatModelSettings.label"
            )}
          />
        }
      />
      <SettingRow
        label={t(
          "generalSettings.settings.hideQuickChatHelper.label",
          "Hide Quick Chat Helper button"
        )}
        {...getResetProps(
          hideQuickChatHelper,
          DEFAULT_CHAT_SETTINGS.hideQuickChatHelper,
          setHideQuickChatHelper
        )}
        control={
          <Switch
            checked={hideQuickChatHelper}
            onChange={setHideQuickChatHelper}
            aria-label={t(
              "generalSettings.settings.hideQuickChatHelper.label",
              "Hide Quick Chat Helper button"
            )}
          />
        }
      />
      <div className="rounded-md border border-border bg-surface2/40 p-3 space-y-1">
        <p className="text-xs font-semibold text-text-muted">
          {t(
            "generalSettings.settings.quickChatDocsScope.title",
            "Quick Chat Docs Q&A scope"
          )}
        </p>
        <p className="text-xs text-text-muted">
          {t(
            "generalSettings.settings.quickChatDocsScope.description",
            "Limit Docs Q&A to project documentation and optional media ID filters."
          )}
        </p>
        <SettingRow
          label={t(
            "generalSettings.settings.quickChatDocsScope.strictOnly.label",
            "Restrict Quick Chat Docs Q&A to project documentation"
          )}
          description={t(
            "generalSettings.settings.quickChatDocsScope.strictOnly.description",
            "When enabled, docs mode queries only media_db with a project-doc namespace."
          )}
          modified={quickChatStrictEnabled !== true}
          onReset={() => void setQuickChatStrictDocsOnly(true)}
          control={
            <Switch
              checked={quickChatStrictEnabled}
              onChange={(checked) => setQuickChatStrictDocsOnly(checked)}
              aria-label={t(
                "generalSettings.settings.quickChatDocsScope.strictOnly.label",
                "Restrict Quick Chat Docs Q&A to project documentation"
              )}
            />
          }
        />
        <SettingRow
          label={t(
            "generalSettings.settings.quickChatDocsScope.namespace.label",
            "Quick Chat project docs namespace"
          )}
          description={t(
            "generalSettings.settings.quickChatDocsScope.namespace.description",
            "Defaults to project_docs when blank."
          )}
          modified={
            quickChatEffectiveNamespace !==
            QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE
          }
          onReset={() => void setQuickChatDocsNamespace("")}
          control={
            <Input
              className="w-[260px]"
              value={quickChatNamespaceInputValue}
              onChange={(event) => setQuickChatDocsNamespace(event.target.value)}
              placeholder={QUICK_CHAT_DEFAULT_PROJECT_DOCS_NAMESPACE}
              disabled={!quickChatStrictEnabled}
              aria-label={t(
                "generalSettings.settings.quickChatDocsScope.namespace.label",
                "Quick Chat project docs namespace"
              )}
            />
          }
        />
        <SettingRow
          label={t(
            "generalSettings.settings.quickChatDocsScope.mediaIds.label",
            "Quick Chat project docs media IDs"
          )}
          description={t(
            "generalSettings.settings.quickChatDocsScope.mediaIds.description",
            "Optional comma-separated IDs (for example: 101, 205, 309). Invalid values are ignored."
          )}
          modified={quickChatMediaIdsInputValue.trim().length > 0}
          onReset={() => void setQuickChatDocsMediaIdsRaw([])}
          control={
            <Input
              className="w-[320px]"
              value={quickChatMediaIdsInputValue}
              onChange={(event) =>
                setQuickChatDocsMediaIdsRaw(event.target.value)
              }
              placeholder={t(
                "generalSettings.settings.quickChatDocsScope.mediaIds.placeholder",
                "Leave blank to search all project docs"
              )}
              disabled={!quickChatStrictEnabled}
              aria-label={t(
                "generalSettings.settings.quickChatDocsScope.mediaIds.label",
                "Quick Chat project docs media IDs"
              )}
            />
          }
        />
        {quickChatStrictEnabled && quickChatParsedMediaIds.length > 0 ? (
          <p className="px-1 text-[11px] text-text-muted">
            {t(
              "generalSettings.settings.quickChatDocsScope.mediaIds.applied",
              {
                defaultValue: "Applied media ID filter: {{ids}}",
                ids: quickChatParsedMediaIds.join(", ")
              }
            )}
          </p>
        ) : null}
      </div>
      <QuickChatWorkflowGuidesSettings />
      <SettingRow
        label={t("generalSettings.settings.restoreLastChatModel.label")}
        {...getResetProps(
          restoreLastChatModel,
          DEFAULT_CHAT_SETTINGS.restoreLastChatModel,
          setRestoreLastChatModel
        )}
        control={
          <Switch
            checked={restoreLastChatModel}
            onChange={setRestoreLastChatModel}
            aria-label={t("generalSettings.settings.restoreLastChatModel.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.generateTitle.label")}
        {...getResetProps(
          generateTitle,
          DEFAULT_CHAT_SETTINGS.titleGenEnabled,
          setGenerateTitle
        )}
        control={
          <Switch
            checked={generateTitle}
            onChange={setGenerateTitle}
            aria-label={t("generalSettings.settings.generateTitle.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.wideMode.label")}
        {...getResetProps(
          checkWideMode,
          DEFAULT_CHAT_SETTINGS.checkWideMode,
          setCheckWideMode
        )}
        control={
          <Switch
            checked={checkWideMode}
            onChange={setCheckWideMode}
            aria-label={t("generalSettings.settings.wideMode.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.stickyChatInput.label")}
        {...getResetProps(
          stickyChatInput,
          DEFAULT_CHAT_SETTINGS.stickyChatInput,
          setStickyChatInput
        )}
        control={
          <Switch
            checked={stickyChatInput}
            onChange={setStickyChatInput}
            aria-label={t("generalSettings.settings.stickyChatInput.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.menuDensity.label", "Menu density")}
        {...getResetProps(
          menuDensity,
          DEFAULT_CHAT_SETTINGS.menuDensity,
          setMenuDensity
        )}
        control={
          <Select
            aria-label={t(
              "generalSettings.settings.menuDensity.label",
              "Menu density"
            )}
            className={SELECT_CLASSNAME}
            value={menuDensity}
            onChange={handleMenuDensityChange}
            options={menuDensityOptions}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.openReasoning.label")}
        {...getResetProps(
          openReasoning,
          DEFAULT_CHAT_SETTINGS.openReasoning,
          setOpenReasoning
        )}
        control={
          <Switch
            checked={openReasoning}
            onChange={setOpenReasoning}
            aria-label={t("generalSettings.settings.openReasoning.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.userChatBubble.label")}
        {...getResetProps(
          userChatBubble,
          DEFAULT_CHAT_SETTINGS.userChatBubble,
          setUserChatBubble
        )}
        control={
          <Switch
            checked={userChatBubble}
            onChange={setUserChatBubble}
            aria-label={t("generalSettings.settings.userChatBubble.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.autoCopyResponseToClipboard.label")}
        {...getResetProps(
          autoCopyResponseToClipboard,
          DEFAULT_CHAT_SETTINGS.autoCopyResponseToClipboard,
          setAutoCopyResponseToClipboard
        )}
        control={
          <Switch
            checked={autoCopyResponseToClipboard}
            onChange={setAutoCopyResponseToClipboard}
            aria-label={t(
              "generalSettings.settings.autoCopyResponseToClipboard.label"
            )}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.useMarkdownForUserMessage.label")}
        {...getResetProps(
          useMarkdownForUserMessage,
          DEFAULT_CHAT_SETTINGS.useMarkdownForUserMessage,
          setUseMarkdownForUserMessage
        )}
        control={
          <Switch
            checked={useMarkdownForUserMessage}
            onChange={setUseMarkdownForUserMessage}
            aria-label={t("generalSettings.settings.useMarkdownForUserMessage.label")}
          />
        }
      />
      <SettingRow
        label={t(
          "generalSettings.settings.renderMermaidDiagrams.label",
          "Render Mermaid diagrams"
        )}
        {...getResetProps(
          renderMermaidDiagrams,
          DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams,
          setRenderMermaidDiagrams
        )}
        control={
          <Switch
            checked={renderMermaidDiagrams}
            onChange={(checked) => setRenderMermaidDiagrams(checked)}
            aria-label={t(
              "generalSettings.settings.renderMermaidDiagrams.label",
              "Render Mermaid diagrams"
            )}
          />
        }
      />
      <SettingRow
        label={t(
          "generalSettings.settings.chatRichTextMode.label",
          "Rich text rendering mode"
        )}
        {...getResetProps(
          chatRichTextMode,
          DEFAULT_CHAT_SETTINGS.chatRichTextMode,
          setChatRichTextMode
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={chatRichTextMode}
            onChange={handleRichTextModeChange}
            options={richTextModeOptions}
            aria-label={t(
              "generalSettings.settings.chatRichTextMode.label",
              "Rich text rendering mode"
            )}
          />
        }
      />
      <div className="rounded-md border border-border bg-surface2/40 p-3">
        <p className="text-xs font-semibold text-text-muted">
          {t(
            "generalSettings.settings.chatRichTextMode.previewLabel",
            "Rendering preview"
          )}
        </p>
        <p className="mt-1 text-xs text-text-muted">
          {t(
            "generalSettings.settings.chatRichTextMode.previewDescription",
            "Same sample rendered in each mode."
          )}
        </p>
        <div className="mt-2 grid gap-2 md:grid-cols-2">
          <div className="rounded-md border border-border bg-surface p-2">
            <p className="mb-1 text-[11px] font-semibold text-text-muted">
              {t(
                "generalSettings.settings.chatRichTextMode.previewSafeTitle",
                "Safe Markdown"
              )}
            </p>
            <Markdown
              message={richTextPreviewSample}
              richTextModeOverride="safe_markdown"
              className="prose prose-sm break-words dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark max-w-none"
              allowExternalImages={false}
            />
          </div>
          <div className="rounded-md border border-border bg-surface p-2">
            <p className="mb-1 text-[11px] font-semibold text-text-muted">
              {t(
                "generalSettings.settings.chatRichTextMode.previewStCompatTitle",
                "SillyTavern-compatible"
              )}
            </p>
            <Markdown
              message={richTextPreviewSample}
              richTextModeOverride="st_compat"
              className="prose prose-sm break-words dark:prose-invert prose-p:leading-relaxed prose-pre:p-0 dark:prose-dark max-w-none"
              allowExternalImages={false}
            />
          </div>
        </div>
      </div>
      <div className="rounded-md border border-border bg-surface2/40 p-3 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold text-text-muted">
              {t(
                "generalSettings.settings.chatRichTextStyles.title",
                "Rich text element styles"
              )}
            </p>
            <p className="mt-1 text-xs text-text-muted">
              {t(
                "generalSettings.settings.chatRichTextStyles.description",
                "Customize italic, bold, and quote rendering across chat."
              )}
            </p>
          </div>
          <button
            type="button"
            onClick={resetRichTextStyles}
            className="inline-flex items-center rounded-md border border-border bg-surface px-2 py-1 text-xs text-text-muted hover:bg-surface2 hover:text-text"
          >
            {t(
              "generalSettings.settings.chatRichTextStyles.reset",
              "Reset rich text styles"
            )}
          </button>
        </div>

        <SettingRow
          label={t(
            "generalSettings.settings.chatRichTextStyles.preset",
            "Style preset"
          )}
          {...getResetProps(
            chatRichTextStylePreset,
            DEFAULT_CHAT_SETTINGS.chatRichTextStylePreset,
            setChatRichTextStylePreset
          )}
          control={
            <Select
              className={SELECT_CLASSNAME}
              value={chatRichTextStylePreset}
              onChange={handleRichTextPresetChange}
              options={richTextStylePresetOptions}
              aria-label={t(
                "generalSettings.settings.chatRichTextStyles.preset",
                "Style preset"
              )}
            />
          }
        />

        <div className="grid gap-3 md:grid-cols-2">
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.italicColor",
              "Italic color"
            )}
            {...getResetProps(
              chatRichItalicColor,
              DEFAULT_CHAT_SETTINGS.chatRichItalicColor,
              setChatRichItalicColor
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichItalicColor}
                onChange={(value) =>
                  handleRichTextColorChange(setChatRichItalicColor, value)
                }
                options={richTextColorOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.italicColor",
                  "Italic color"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.italicFont",
              "Italic font"
            )}
            {...getResetProps(
              chatRichItalicFont,
              DEFAULT_CHAT_SETTINGS.chatRichItalicFont,
              setChatRichItalicFont
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichItalicFont}
                onChange={(value) =>
                  handleRichTextFontChange(setChatRichItalicFont, value)
                }
                options={fontOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.italicFont",
                  "Italic font"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.boldColor",
              "Bold color"
            )}
            {...getResetProps(
              chatRichBoldColor,
              DEFAULT_CHAT_SETTINGS.chatRichBoldColor,
              setChatRichBoldColor
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichBoldColor}
                onChange={(value) =>
                  handleRichTextColorChange(setChatRichBoldColor, value)
                }
                options={richTextColorOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.boldColor",
                  "Bold color"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.boldFont",
              "Bold font"
            )}
            {...getResetProps(
              chatRichBoldFont,
              DEFAULT_CHAT_SETTINGS.chatRichBoldFont,
              setChatRichBoldFont
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichBoldFont}
                onChange={(value) =>
                  handleRichTextFontChange(setChatRichBoldFont, value)
                }
                options={fontOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.boldFont",
                  "Bold font"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.quoteTextColor",
              "Quote text color"
            )}
            {...getResetProps(
              chatRichQuoteTextColor,
              DEFAULT_CHAT_SETTINGS.chatRichQuoteTextColor,
              setChatRichQuoteTextColor
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichQuoteTextColor}
                onChange={(value) =>
                  handleRichTextColorChange(setChatRichQuoteTextColor, value)
                }
                options={richTextColorOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.quoteTextColor",
                  "Quote text color"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.quoteFont",
              "Quote font"
            )}
            {...getResetProps(
              chatRichQuoteFont,
              DEFAULT_CHAT_SETTINGS.chatRichQuoteFont,
              setChatRichQuoteFont
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichQuoteFont}
                onChange={(value) =>
                  handleRichTextFontChange(setChatRichQuoteFont, value)
                }
                options={fontOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.quoteFont",
                  "Quote font"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.quoteBorderColor",
              "Quote border color"
            )}
            {...getResetProps(
              chatRichQuoteBorderColor,
              DEFAULT_CHAT_SETTINGS.chatRichQuoteBorderColor,
              setChatRichQuoteBorderColor
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichQuoteBorderColor}
                onChange={(value) =>
                  handleRichTextColorChange(setChatRichQuoteBorderColor, value)
                }
                options={richTextColorOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.quoteBorderColor",
                  "Quote border color"
                )}
              />
            }
          />
          <SettingRow
            label={t(
              "generalSettings.settings.chatRichTextStyles.quoteBackgroundColor",
              "Quote background color"
            )}
            {...getResetProps(
              chatRichQuoteBackgroundColor,
              DEFAULT_CHAT_SETTINGS.chatRichQuoteBackgroundColor,
              setChatRichQuoteBackgroundColor
            )}
            control={
              <Select
                className={SELECT_CLASSNAME}
                value={chatRichQuoteBackgroundColor}
                onChange={(value) =>
                  handleRichTextColorChange(
                    setChatRichQuoteBackgroundColor,
                    value
                  )
                }
                options={richTextColorOptions}
                aria-label={t(
                  "generalSettings.settings.chatRichTextStyles.quoteBackgroundColor",
                  "Quote background color"
                )}
              />
            }
          />
        </div>
      </div>
      <SettingRow
        label={t(
          "generalSettings.settings.allowExternalImages.label",
          "Load external images in messages"
        )}
        {...getResetProps(
          allowExternalImages,
          DEFAULT_CHAT_SETTINGS.allowExternalImages,
          setAllowExternalImages
        )}
        control={
          <Switch
            checked={allowExternalImages}
            onChange={setAllowExternalImages}
            aria-label={t(
              "generalSettings.settings.allowExternalImages.label",
              "Load external images in messages"
            )}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.copyAsFormattedText.label")}
        {...getResetProps(
          copyAsFormattedText,
          DEFAULT_CHAT_SETTINGS.copyAsFormattedText,
          setCopyAsFormattedText
        )}
        control={
          <Switch
            checked={copyAsFormattedText}
            onChange={setCopyAsFormattedText}
            aria-label={t("generalSettings.settings.copyAsFormattedText.label")}
          />
        }
      />
      <SettingRow
        label={
          <span className="inline-flex items-center gap-2">
            {t("generalSettings.settings.tabMentionsEnabled.label")}
            <BetaTag />
          </span>
        }
        {...getResetProps(
          tabMentionsEnabled,
          DEFAULT_CHAT_SETTINGS.tabMentionsEnabled,
          setTabMentionsEnabled
        )}
        control={
          <Switch
            checked={tabMentionsEnabled}
            onChange={setTabMentionsEnabled}
            aria-label={t("generalSettings.settings.tabMentionsEnabled.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.pasteLargeTextAsFile.label")}
        {...getResetProps(
          pasteLargeTextAsFile,
          DEFAULT_CHAT_SETTINGS.pasteLargeTextAsFile,
          setPasteLargeTextAsFile
        )}
        control={
          <Switch
            checked={pasteLargeTextAsFile}
            onChange={setPasteLargeTextAsFile}
            aria-label={t("generalSettings.settings.pasteLargeTextAsFile.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.sidepanelTemporaryChat.label")}
        {...getResetProps(
          sidepanelTemporaryChat,
          DEFAULT_CHAT_SETTINGS.sidepanelTemporaryChat,
          setSidepanelTemporaryChat
        )}
        control={
          <Switch
            checked={sidepanelTemporaryChat}
            onChange={setSidepanelTemporaryChat}
            aria-label={t("generalSettings.settings.sidepanelTemporaryChat.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.removeReasoningTagFromCopy.label")}
        {...getResetProps(
          removeReasoningTagFromCopy,
          DEFAULT_CHAT_SETTINGS.removeReasoningTagFromCopy,
          setRemoveReasoningTagFromCopy
        )}
        control={
          <Switch
            checked={removeReasoningTagFromCopy}
            onChange={setRemoveReasoningTagFromCopy}
            aria-label={t("generalSettings.settings.removeReasoningTagFromCopy.label")}
          />
        }
      />
      <SettingRow
        label={t("generalSettings.settings.promptSearchIncludeServer.label")}
        {...getResetProps(
          promptSearchIncludeServer,
          DEFAULT_CHAT_SETTINGS.promptSearchIncludeServer,
          setPromptSearchIncludeServer
        )}
        control={
          <Switch
            checked={promptSearchIncludeServer}
            onChange={setPromptSearchIncludeServer}
            aria-label={t("generalSettings.settings.promptSearchIncludeServer.label")}
          />
        }
      />

      <div>
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("chatAppearance.title", "Chat Appearance")}
        </h2>
        <p className="mt-1 text-sm text-text-muted">
          {t(
            "chatAppearance.description",
            "Customize colors, fonts, and text sizes for chat messages."
          )}
        </p>
        <div className="border-b border-border mt-3" />
      </div>

      <SettingRow
        label={t(
          "chatAppearance.backgroundImage.label",
          "Chat background image"
        )}
        description={t(
          "chatAppearance.backgroundImage.description",
          "Shown behind the chat screen while you are on /chat."
        )}
        control={
          <div className="flex items-center gap-2">
            {chatBackgroundImage ? (
              <Tooltip
                title={t(
                  "chatAppearance.backgroundImage.clear",
                  "Remove background image"
                )}
              >
                <button
                  type="button"
                  onClick={resetChatBackgroundImage}
                  className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-border text-text-muted transition-colors hover:bg-surface2 hover:text-text"
                  aria-label={t(
                    "chatAppearance.backgroundImage.clear",
                    "Remove background image"
                  )}
                >
                  <RotateCcw className="size-4" aria-hidden="true" />
                </button>
              </Tooltip>
            ) : null}
            <button
              type="button"
              onClick={() => chatBackgroundInputRef.current?.click()}
              className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-primaryStrong"
              aria-label={t(
                "chatAppearance.backgroundImage.upload",
                "Upload image"
              )}
            >
              <Upload className="size-4" aria-hidden="true" />
              <span>
                {t("chatAppearance.backgroundImage.upload", "Upload image")}
              </span>
            </button>
            <input
              ref={chatBackgroundInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleChatBackgroundImageUpload}
            />
          </div>
        }
      />

      <div className="pt-4">
        <h3 className="text-sm font-semibold leading-6 text-text">
          {t("chatAppearance.userHeading", "User messages")}
        </h3>
      </div>

      <SettingRow
        label={t("chatAppearance.userColor", "User text color")}
        id="user-text-color"
        {...getResetProps(
          userTextColor,
          DEFAULT_CHAT_SETTINGS.chatUserTextColor,
          setUserTextColor
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={userTextColor}
            onChange={setUserTextColor}
            options={colorOptions}
            aria-label={t("chatAppearance.userColor", "User text color")}
          />
        }
      />

      <SettingRow
        label={t("chatAppearance.userFont", "User font")}
        id="user-text-font"
        {...getResetProps(
          userTextFont,
          DEFAULT_CHAT_SETTINGS.chatUserTextFont,
          setUserTextFont
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={userTextFont}
            onChange={setUserTextFont}
            options={fontOptions}
            aria-label={t("chatAppearance.userFont", "User font")}
          />
        }
      />

      <SettingRow
        label={t("chatAppearance.userSize", "User text size")}
        id="user-text-size"
        {...getResetProps(
          userTextSize,
          DEFAULT_CHAT_SETTINGS.chatUserTextSize,
          setUserTextSize
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={userTextSize}
            onChange={handleUserTextSizeChange}
            options={sizeOptions}
            aria-label={t("chatAppearance.userSize", "User text size")}
          />
        }
      />

      <div className="pt-4">
        <h3 className="text-sm font-semibold leading-6 text-text">
          {t("chatAppearance.assistantHeading", "Assistant messages")}
        </h3>
      </div>

      <SettingRow
        label={t("chatAppearance.assistantColor", "Assistant text color")}
        id="assistant-text-color"
        {...getResetProps(
          assistantTextColor,
          DEFAULT_CHAT_SETTINGS.chatAssistantTextColor,
          setAssistantTextColor
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={assistantTextColor}
            onChange={setAssistantTextColor}
            options={colorOptions}
            aria-label={t("chatAppearance.assistantColor", "Assistant text color")}
          />
        }
      />

      <SettingRow
        label={t("chatAppearance.assistantFont", "Assistant font")}
        id="assistant-text-font"
        {...getResetProps(
          assistantTextFont,
          DEFAULT_CHAT_SETTINGS.chatAssistantTextFont,
          setAssistantTextFont
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={assistantTextFont}
            onChange={setAssistantTextFont}
            options={fontOptions}
            aria-label={t("chatAppearance.assistantFont", "Assistant font")}
          />
        }
      />

      <SettingRow
        label={t("chatAppearance.assistantSize", "Assistant text size")}
        id="assistant-text-size"
        {...getResetProps(
          assistantTextSize,
          DEFAULT_CHAT_SETTINGS.chatAssistantTextSize,
          setAssistantTextSize
        )}
        control={
          <Select
            className={SELECT_CLASSNAME}
            value={assistantTextSize}
            onChange={handleAssistantTextSizeChange}
            options={sizeOptions}
            aria-label={t("chatAppearance.assistantSize", "Assistant text size")}
          />
        }
      />

      <div className="mt-8">
        <DiscoSkillsSettings />
      </div>
    </div>
  )
}
