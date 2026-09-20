import React from "react"
import { useNavigate } from "react-router-dom"
import { Modal } from "antd"
import { shallow } from "zustand/shallow"
import { useStorage } from "@plasmohq/storage/hook"
import { useChatBaseState } from "@/hooks/chat/useChatBaseState"
import { focusTextArea } from "@/hooks/utils/messageHelpers"
import { useStoreMessageOption } from "@/store/option"
import { useStoreMessage } from "@/store"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreChatModelSettings } from "@/store/model"
import { cleanupAntOverlays } from "@/utils/cleanup-ant-overlays"
import { requestSettingsNavigation } from "@/utils/settings-return"
import { updatePageTitle } from "@/utils/update-page-title"

type UseClearChatOptions = {
  textareaRef?: React.RefObject<HTMLTextAreaElement | null>
}

type ResettableChatBase = {
  setMessages: (messagesOrUpdater: unknown) => void
  setHistory: (historyOrUpdater: unknown) => void
  setHistoryId: (
    historyId: string | null,
    options?: { preserveServerChatId?: boolean }
  ) => void
  setIsFirstMessage: (isFirstMessage: boolean) => void
  setIsLoading: (isLoading: boolean) => void
  setIsProcessing: (isProcessing: boolean) => void
  setStreaming: (streaming: boolean) => void
}

export const useClearChat = ({ textareaRef }: UseClearChatOptions = {}) => {
  const navigate = useNavigate()
  const currentChatModelSettings = useStoreChatModelSettings()
  const [defaultInternetSearchOn] = useStorage("defaultInternetSearchOn", false)

  const primaryBase = useChatBaseState(useStoreMessageOption)
  const secondaryBase = useChatBaseState(useStoreMessage)
  const resolveClearChatPath = React.useCallback(() => {
    if (typeof window !== "undefined") {
      const hostPath = window.location.pathname.toLowerCase()
      if (hostPath.endsWith("sidepanel.html")) {
        return "/"
      }
    }
    return "/chat"
  }, [])
  const {
    setServerChatId,
    setServerChatVersion,
    setContextFiles,
    setDocumentContext,
    setUploadedFiles,
    setFileRetrievalEnabled,
    setActionInfo,
    setRagMediaIds,
    setRagSearchMode,
    setRagTopK,
    setRagEnableGeneration,
    setRagEnableCitations,
    setRagSources,
    clearQueuedMessages,
    setCompareMode,
    setCompareSelectedModels,
    clearReplyTarget,
    setWebSearch
  } = useStoreMessageOption(
    (state) => ({
      setServerChatId: state.setServerChatId,
      setServerChatVersion: state.setServerChatVersion,
      setContextFiles: state.setContextFiles,
      setDocumentContext: state.setDocumentContext,
      setUploadedFiles: state.setUploadedFiles,
      setFileRetrievalEnabled: state.setFileRetrievalEnabled,
      setActionInfo: state.setActionInfo,
      setRagMediaIds: state.setRagMediaIds,
      setRagSearchMode: state.setRagSearchMode,
      setRagTopK: state.setRagTopK,
      setRagEnableGeneration: state.setRagEnableGeneration,
      setRagEnableCitations: state.setRagEnableCitations,
      setRagSources: state.setRagSources,
      clearQueuedMessages: state.clearQueuedMessages,
      setCompareMode: state.setCompareMode,
      setCompareSelectedModels: state.setCompareSelectedModels,
      clearReplyTarget: state.clearReplyTarget,
      setWebSearch: state.setWebSearch
    }),
    shallow
  )

  return React.useCallback(() => {
    const destination = resolveClearChatPath()
    if (!requestSettingsNavigation(destination)) return
    if (typeof window !== "undefined") {
      Modal.destroyAll()
      cleanupAntOverlays()
    }
    navigate(destination)
    const resetBaseState = (base: ResettableChatBase) => {
      base.setMessages([])
      base.setHistory([])
      base.setHistoryId(null)
      base.setIsFirstMessage(true)
      base.setIsLoading(false)
      base.setIsProcessing(false)
      base.setStreaming(false)
    }
    resetBaseState(primaryBase)
    resetBaseState(secondaryBase)
    setServerChatId(null)
    setServerChatVersion(null)
    setContextFiles([])
    updatePageTitle()
    currentChatModelSettings.reset()
    if (defaultInternetSearchOn) {
      setWebSearch(true)
    }
    focusTextArea(textareaRef)
    setDocumentContext(null)
    setUploadedFiles([])
    setFileRetrievalEnabled(false)
    setActionInfo(null)
    setRagMediaIds(null)
    setRagSearchMode("hybrid")
    setRagTopK(null)
    setRagEnableGeneration(false)
    setRagEnableCitations(false)
    setRagSources([])
    clearQueuedMessages()
    setCompareMode(false)
    setCompareSelectedModels([])
    useStoreMessageOption.setState({
      compareSelectionByCluster: {},
      compareCanonicalByCluster: {},
      compareContinuationModeByCluster: {},
      compareSplitChats: {},
      compareActiveModelsByCluster: {},
      compareParentByHistory: {}
    })
    clearReplyTarget()
    usePlaygroundSessionStore.getState().clearSession()
  }, [
    clearQueuedMessages,
    clearReplyTarget,
    currentChatModelSettings,
    defaultInternetSearchOn,
    navigate,
    resolveClearChatPath,
    setActionInfo,
    setCompareMode,
    setCompareSelectedModels,
    setContextFiles,
    setDocumentContext,
    setFileRetrievalEnabled,
    primaryBase,
    secondaryBase,
    setRagEnableCitations,
    setRagEnableGeneration,
    setRagMediaIds,
    setRagSearchMode,
    setRagSources,
    setRagTopK,
    setServerChatId,
    setServerChatVersion,
    setUploadedFiles,
    setWebSearch,
    textareaRef
  ])
}
