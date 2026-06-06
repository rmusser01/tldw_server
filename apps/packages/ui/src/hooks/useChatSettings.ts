import { useStorage } from "@plasmohq/storage/hook"
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings"

export const useChatSettings = () => {
  const [copilotResumeLastChat, setCopilotResumeLastChat] = useStorage(
    "copilotResumeLastChat",
    DEFAULT_CHAT_SETTINGS.copilotResumeLastChat
  )
  const [defaultChatWithWebsite, setDefaultChatWithWebsite] = useStorage(
    "defaultChatWithWebsite",
    DEFAULT_CHAT_SETTINGS.defaultChatWithWebsite
  )
  const [webUIResumeLastChat, setWebUIResumeLastChat] = useStorage(
    "webUIResumeLastChat",
    DEFAULT_CHAT_SETTINGS.webUIResumeLastChat
  )
  const [hideCurrentChatModelSettings, setHideCurrentChatModelSettings] =
    useStorage(
      "hideCurrentChatModelSettings",
      DEFAULT_CHAT_SETTINGS.hideCurrentChatModelSettings
    )
  const [hideQuickChatHelper, setHideQuickChatHelper] = useStorage(
    "hideQuickChatHelper",
    DEFAULT_CHAT_SETTINGS.hideQuickChatHelper
  )
  const [restoreLastChatModel, setRestoreLastChatModel] = useStorage(
    "restoreLastChatModel",
    DEFAULT_CHAT_SETTINGS.restoreLastChatModel
  )
  const [generateTitle, setGenerateTitle] = useStorage(
    "titleGenEnabled",
    DEFAULT_CHAT_SETTINGS.titleGenEnabled
  )
  const [checkWideMode, setCheckWideMode] = useStorage(
    "checkWideMode",
    DEFAULT_CHAT_SETTINGS.checkWideMode
  )
  const [stickyChatInput, setStickyChatInput] = useStorage(
    "stickyChatInput",
    DEFAULT_CHAT_SETTINGS.stickyChatInput
  )
  const [menuDensity, setMenuDensity] = useStorage(
    "menuDensity",
    DEFAULT_CHAT_SETTINGS.menuDensity
  )
  const [openReasoning, setOpenReasoning] = useStorage(
    "openReasoning",
    DEFAULT_CHAT_SETTINGS.openReasoning
  )
  const [userChatBubble, setUserChatBubble] = useStorage(
    "userChatBubble",
    DEFAULT_CHAT_SETTINGS.userChatBubble
  )
  const [autoCopyResponseToClipboard, setAutoCopyResponseToClipboard] =
    useStorage(
      "autoCopyResponseToClipboard",
      DEFAULT_CHAT_SETTINGS.autoCopyResponseToClipboard
    )
  const [useMarkdownForUserMessage, setUseMarkdownForUserMessage] = useStorage(
    "useMarkdownForUserMessage",
    DEFAULT_CHAT_SETTINGS.useMarkdownForUserMessage
  )
  const [renderMermaidDiagrams, setRenderMermaidDiagrams] = useStorage(
    "renderMermaidDiagrams",
    DEFAULT_CHAT_SETTINGS.renderMermaidDiagrams
  )
  const [chatRichTextMode, setChatRichTextMode] = useStorage(
    "chatRichTextMode",
    DEFAULT_CHAT_SETTINGS.chatRichTextMode
  )
  const [chatRichTextStylePreset, setChatRichTextStylePreset] = useStorage(
    "chatRichTextStylePreset",
    DEFAULT_CHAT_SETTINGS.chatRichTextStylePreset
  )
  const [chatRichItalicColor, setChatRichItalicColor] = useStorage(
    "chatRichItalicColor",
    DEFAULT_CHAT_SETTINGS.chatRichItalicColor
  )
  const [chatRichItalicFont, setChatRichItalicFont] = useStorage(
    "chatRichItalicFont",
    DEFAULT_CHAT_SETTINGS.chatRichItalicFont
  )
  const [chatRichBoldColor, setChatRichBoldColor] = useStorage(
    "chatRichBoldColor",
    DEFAULT_CHAT_SETTINGS.chatRichBoldColor
  )
  const [chatRichBoldFont, setChatRichBoldFont] = useStorage(
    "chatRichBoldFont",
    DEFAULT_CHAT_SETTINGS.chatRichBoldFont
  )
  const [chatRichQuoteTextColor, setChatRichQuoteTextColor] = useStorage(
    "chatRichQuoteTextColor",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteTextColor
  )
  const [chatRichQuoteFont, setChatRichQuoteFont] = useStorage(
    "chatRichQuoteFont",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteFont
  )
  const [chatRichQuoteBorderColor, setChatRichQuoteBorderColor] = useStorage(
    "chatRichQuoteBorderColor",
    DEFAULT_CHAT_SETTINGS.chatRichQuoteBorderColor
  )
  const [chatRichQuoteBackgroundColor, setChatRichQuoteBackgroundColor] =
    useStorage(
      "chatRichQuoteBackgroundColor",
      DEFAULT_CHAT_SETTINGS.chatRichQuoteBackgroundColor
    )
  const [copyAsFormattedText, setCopyAsFormattedText] = useStorage(
    "copyAsFormattedText",
    DEFAULT_CHAT_SETTINGS.copyAsFormattedText
  )
  const [allowExternalImages, setAllowExternalImages] = useStorage(
    "allowExternalImages",
    DEFAULT_CHAT_SETTINGS.allowExternalImages
  )
  const [tabMentionsEnabled, setTabMentionsEnabled] = useStorage(
    "tabMentionsEnabled",
    DEFAULT_CHAT_SETTINGS.tabMentionsEnabled
  )
  const [pasteLargeTextAsFile, setPasteLargeTextAsFile] = useStorage(
    "pasteLargeTextAsFile",
    DEFAULT_CHAT_SETTINGS.pasteLargeTextAsFile
  )
  const [sidepanelTemporaryChat, setSidepanelTemporaryChat] = useStorage(
    "sidepanelTemporaryChat",
    DEFAULT_CHAT_SETTINGS.sidepanelTemporaryChat
  )
  const [removeReasoningTagFromCopy, setRemoveReasoningTagFromCopy] =
    useStorage(
      "removeReasoningTagFromCopy",
      DEFAULT_CHAT_SETTINGS.removeReasoningTagFromCopy
    )
  const [promptSearchIncludeServer, setPromptSearchIncludeServer] =
    useStorage(
      "promptSearchIncludeServer",
      DEFAULT_CHAT_SETTINGS.promptSearchIncludeServer
    )
  const [userTextColor, setUserTextColor] = useStorage(
    "chatUserTextColor",
    DEFAULT_CHAT_SETTINGS.chatUserTextColor
  )
  const [assistantTextColor, setAssistantTextColor] = useStorage(
    "chatAssistantTextColor",
    DEFAULT_CHAT_SETTINGS.chatAssistantTextColor
  )
  const [userTextFont, setUserTextFont] = useStorage(
    "chatUserTextFont",
    DEFAULT_CHAT_SETTINGS.chatUserTextFont
  )
  const [assistantTextFont, setAssistantTextFont] = useStorage(
    "chatAssistantTextFont",
    DEFAULT_CHAT_SETTINGS.chatAssistantTextFont
  )
  const [userTextSize, setUserTextSize] = useStorage(
    "chatUserTextSize",
    DEFAULT_CHAT_SETTINGS.chatUserTextSize
  )
  const [assistantTextSize, setAssistantTextSize] = useStorage(
    "chatAssistantTextSize",
    DEFAULT_CHAT_SETTINGS.chatAssistantTextSize
  )

  return {
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
  }
}
