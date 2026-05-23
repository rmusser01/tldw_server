import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useStoreMessageOption } from "~/store/option";
import { useTranslation } from "react-i18next";
import { usePageAssist } from "@/context";
import { useWebUI } from "@/store/webui";
import { useStorage } from "@plasmohq/storage/hook";
import { useStoreChatModelSettings } from "@/store/model";
import { formatFileSize } from "@/utils/format";
import { useAntdNotification } from "./useAntdNotification";
import { useChatBaseState } from "@/hooks/chat/useChatBaseState";
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat";
import { useServerChatHistoryId } from "@/hooks/chat/useServerChatHistoryId";
import { useServerChatLoader } from "@/hooks/chat/useServerChatLoader";
import { useClearChat } from "@/hooks/chat/useClearChat";
import { useCompareMode } from "@/hooks/chat/useCompareMode";
import { useChatActions } from "@/hooks/chat/useChatActions";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { usePromptPersistence } from "@/hooks/chat/usePromptPersistence";
import { useRagSettings } from "@/hooks/chat/useRagSettings";
import { useFileUpload } from "@/hooks/chat/useFileUpload";
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord";
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state";
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant";
import type { AssistantSelection } from "@/types/assistant-selection";
import type { Character } from "@/types/character";
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter";
import { useSetting } from "@/hooks/useSetting";
import { CONTEXT_FILE_SIZE_MB_SETTING } from "@/services/settings/ui-settings";
import {
  DEFAULT_MESSAGE_STEERING_PROMPTS,
  MESSAGE_STEERING_PROMPTS_STORAGE_KEY,
} from "@/utils/message-steering";
import type { MessageSteeringPromptTemplates } from "@/types/message-steering";
import { useChatLoopState } from "@/services/chat-loop/hooks";
import { subscribeChatLoopEvents } from "@/services/chat-loop/bridge";
import type { ChatScope } from "@/types/chat-scope";

export const useMessageOption = (
  opts: { forceCompareEnabled?: boolean; scope?: ChatScope } = {},
) => {
  const logE2EDebug = React.useCallback(
    (
      _key: "syncSystem" | "syncQuick" | "storeSystem" | "storeQuick",
      _payload: Record<string, unknown>,
    ) => {},
    [],
  );
  // Controllers come from Context (for aborting streaming requests)
  const { controller: abortController, setController: setAbortController } =
    usePageAssist();

  const {
    messages,
    setMessages,
    history,
    setHistory,
    streaming,
    setStreaming,
    isFirstMessage,
    setIsFirstMessage,
    historyId,
    setHistoryId,
    isLoading,
    setIsLoading,
    isProcessing,
    setIsProcessing,
    chatMode,
    setChatMode,
    isEmbedding,
    setIsEmbedding,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    useOCR,
    setUseOCR,
  } = useChatBaseState(useStoreMessageOption);

  const {
    webSearch,
    setWebSearch,
    toolChoice,
    setToolChoice,
    isSearchingInternet,
    setIsSearchingInternet,
    queuedMessages: storeQueuedMessages,
    addQueuedMessage: storeAddQueuedMessage,
    setQueuedMessages: storeSetQueuedMessages,
    clearQueuedMessages: storeClearQueuedMessages,
    selectedKnowledge,
    setSelectedKnowledge,
    temporaryChat,
    setTemporaryChat,
    documentContext,
    setDocumentContext,
    uploadedFiles,
    setUploadedFiles,
    contextFiles,
    setContextFiles,
    actionInfo,
    setActionInfo,
    setFileRetrievalEnabled,
    fileRetrievalEnabled,
    ragMediaIds,
    setRagMediaIds,
    ragSearchMode,
    setRagSearchMode,
    ragTopK,
    setRagTopK,
    ragEnableGeneration,
    setRagEnableGeneration,
    ragEnableCitations,
    setRagEnableCitations,
    ragSources,
    setRagSources,
    ragAdvancedOptions,
    setRagAdvancedOptions,
    ragPinnedResults,
    setRagPinnedResults,
    serverChatId,
    setServerChatId,
    serverChatTitle,
    setServerChatTitle,
    serverChatCharacterId,
    setServerChatCharacterId,
    serverChatAssistantKind,
    setServerChatAssistantKind,
    serverChatAssistantId,
    setServerChatAssistantId,
    serverChatPersonaMemoryMode,
    setServerChatPersonaMemoryMode,
    serverChatMetaLoaded,
    setServerChatMetaLoaded,
    serverChatLoadState,
    setServerChatLoadState,
    serverChatLoadError,
    setServerChatLoadError,
    serverChatState,
    setServerChatState,
    serverChatVersion,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
    messageSteeringMode,
    setMessageSteeringMode,
    messageSteeringForceNarrate,
    setMessageSteeringForceNarrate,
    clearMessageSteering,
    replyTarget,
    clearReplyTarget,
  } = useStoreMessageOption();

  const {
    compareMode,
    setCompareMode,
    compareFeatureEnabled,
    setCompareFeatureEnabled,
    compareSelectedModels,
    setCompareSelectedModels,
    compareSelectionByCluster,
    setCompareSelectionForCluster,
    compareActiveModelsByCluster,
    setCompareActiveModelsForCluster,
    compareParentByHistory,
    setCompareParentForHistory,
    compareCanonicalByCluster,
    setCompareCanonicalForCluster,
    compareContinuationModeByCluster,
    setCompareContinuationModeForCluster,
    compareSplitChats,
    setCompareSplitChat,
    compareMaxModels,
    setCompareMaxModels,
    compareModeActive,
    markCompareHistoryCreated,
  } = useCompareMode({ historyId, forceEnabled: opts.forceCompareEnabled });

  const currentChatModelSettings = useStoreChatModelSettings();
  const { selectedModel, setSelectedModel, selectedModelIsLoading } =
    useSelectedModel();
  const [selectedCharacter, setSelectedCharacter] =
    useSelectedCharacter<Character | null>(null);
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null);
  const [defaultInternetSearchOn] = useStorage(
    "defaultInternetSearchOn",
    false,
  );
  const [speechToTextLanguage, setSpeechToTextLanguage] = useStorage(
    "speechToTextLanguage",
    "en-US",
  );
  const [messageSteeringPrompts] = useStorage<MessageSteeringPromptTemplates>(
    MESSAGE_STEERING_PROMPTS_STORAGE_KEY,
    DEFAULT_MESSAGE_STEERING_PROMPTS,
  );
  const {
    state: chatLoopState,
    dispatch: dispatchChatLoopEvent,
    reset: resetChatLoopState,
  } = useChatLoopState();

  const { ttsEnabled } = useWebUI();

  const { t } = useTranslation("option");
  const [contextFileMaxSizeMb] = useSetting(CONTEXT_FILE_SIZE_MB_SETTING);
  const maxContextFileSizeBytes = React.useMemo(
    () => contextFileMaxSizeMb * 1024 * 1024,
    [contextFileMaxSizeMb],
  );
  const maxContextFileSizeLabel = React.useMemo(
    () => formatFileSize(maxContextFileSizeBytes),
    [maxContextFileSizeBytes],
  );
  const queryClient = useQueryClient();
  const invalidateServerChatHistory = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] });
  }, [queryClient]);
  const notification = useAntdNotification();

  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const selectServerChat = useSelectServerChat();
  const { ensureServerChatHistoryId } = useServerChatHistoryId({
    serverChatId,
    historyId,
    setHistoryId,
    temporaryChat,
    t,
  });

  useServerChatLoader({
    ensureServerChatHistoryId,
    notification,
    t,
    scope: opts.scope,
  });
  const { settings: chatSettings } = useChatSettingsRecord({
    historyId,
    serverChatId,
  });
  const effectiveAssistantState = React.useMemo(
    () =>
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: serverChatAssistantKind,
          assistantId: serverChatAssistantId,
          characterId: serverChatCharacterId,
        },
        settings: chatSettings ?? null,
        draftSelection: selectedAssistant,
      }),
    [
      chatSettings,
      selectedAssistant,
      serverChatAssistantId,
      serverChatAssistantKind,
      serverChatCharacterId,
    ],
  );
  const effectiveSelectedAssistant = React.useMemo<AssistantSelection | null>(() => {
    if (effectiveAssistantState.mode === "plain") {
      return selectedAssistant;
    }

    const matchesDraftSelection =
      selectedAssistant?.kind === effectiveAssistantState.kind &&
      selectedAssistant.id === effectiveAssistantState.id;
    const draftMetadata = matchesDraftSelection ? selectedAssistant : null;

    return {
      ...draftMetadata,
      kind: effectiveAssistantState.kind!,
      id: effectiveAssistantState.id!,
      name:
        effectiveAssistantState.displayName ??
        draftMetadata?.name ??
        (effectiveAssistantState.kind === "persona" ? "Persona" : "Assistant"),
      avatar_url: effectiveAssistantState.avatarUrl ?? draftMetadata?.avatar_url ?? null,
      system_prompt:
        effectiveAssistantState.systemPromptSnapshot ??
        draftMetadata?.system_prompt ??
        null,
    };
  }, [effectiveAssistantState, selectedAssistant]);

  React.useEffect(() => {
    if (!serverChatId || temporaryChat) return;
    void ensureServerChatHistoryId(serverChatId, serverChatTitle || undefined);
  }, [ensureServerChatHistoryId, serverChatId, serverChatTitle, temporaryChat]);

  usePromptPersistence({
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    logE2EDebug,
  });

  useRagSettings({
    historyId,
    serverChatId,
    messagesLength: messages.length,
    setRagSearchMode,
    setRagTopK,
    setRagEnableGeneration,
    setRagEnableCitations,
    setRagSources,
    setRagAdvancedOptions,
  });

  const { handleFileUpload, removeUploadedFile, clearUploadedFiles } =
    useFileUpload({
      maxContextFileSizeBytes,
      maxContextFileSizeLabel,
      notification,
      t,
      uploadedFiles,
      setUploadedFiles,
      contextFiles,
      setContextFiles,
    });

  const handleSetFileRetrievalEnabled = async (enabled: boolean) => {
    setFileRetrievalEnabled(enabled);
  };

  const clearChat = useClearChat({ textareaRef });
  React.useEffect(
    () => subscribeChatLoopEvents(dispatchChatLoopEvent),
    [dispatchChatLoopEvent],
  );
  const {
    onSubmit: submitChat,
    sendPerModelReply,
    regenerateLastMessage,
    stopStreamingRequest,
    editMessage,
    deleteMessage,
    toggleMessagePinned,
    createChatBranch,
    createCompareBranch,
  } = useChatActions({
    t,
    notification,
    abortController,
    setAbortController,
    messages,
    setMessages,
    history,
    setHistory,
    historyId,
    setHistoryId,
    temporaryChat,
    selectedModel,
    useOCR,
    selectedSystemPrompt,
    selectedKnowledge,
    toolChoice,
    webSearch,
    currentChatModelSettings,
    setIsSearchingInternet,
    setIsProcessing,
    setStreaming,
    setActionInfo,
    fileRetrievalEnabled,
    ragMediaIds,
    ragSearchMode,
    ragTopK,
    ragEnableGeneration,
    ragEnableCitations,
    ragSources,
    ragAdvancedOptions,
    serverChatId,
    serverChatTitle,
    serverChatCharacterId,
    serverChatAssistantKind,
    serverChatAssistantId,
    serverChatPersonaMemoryMode,
    serverChatMetaLoaded,
    serverChatState,
    serverChatTopic,
    serverChatClusterId,
    serverChatSource,
    serverChatExternalRef,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatAssistantKind,
    setServerChatAssistantId,
    setServerChatPersonaMemoryMode,
    setServerChatMetaLoaded,
    setServerChatState,
    setServerChatVersion,
    setServerChatTopic,
    setServerChatClusterId,
    setServerChatSource,
    setServerChatExternalRef,
    ensureServerChatHistoryId,
    contextFiles,
    setContextFiles,
    documentContext,
    setDocumentContext,
    uploadedFiles,
    compareModeActive,
    compareSelectedModels,
    compareMaxModels,
    compareFeatureEnabled,
    markCompareHistoryCreated,
    messageSteeringPrompts,
    messageSteeringMode,
    messageSteeringForceNarrate,
    clearMessageSteering,
    replyTarget,
    clearReplyTarget,
    setSelectedQuickPrompt,
    setSelectedSystemPrompt,
    invalidateServerChatHistory,
    selectedCharacter,
    selectedAssistant: effectiveSelectedAssistant,
    scope: opts.scope,
  });
  const onSubmit = React.useCallback(
    async (...args: Parameters<typeof submitChat>) => {
      resetChatLoopState();
      return submitChat(...args);
    },
    [resetChatLoopState, submitChat],
  );

  return {
    editMessage,
    deleteMessage,
    toggleMessagePinned,
    messages,
    setMessages,
    onSubmit,
    setStreaming,
    streaming,
    setHistory,
    historyId,
    setHistoryId,
    selectServerChat,
    setIsFirstMessage,
    isLoading,
    setIsLoading,
    isProcessing,
    setIsProcessing,
    stopStreamingRequest,
    clearChat,
    selectedModel,
    selectedModelIsLoading,
    setSelectedModel,
    chatMode,
    setChatMode,
    isEmbedding,
    setIsEmbedding,
    speechToTextLanguage,
    setSpeechToTextLanguage,
    regenerateLastMessage,
    webSearch,
    setWebSearch,
    toolChoice,
    setToolChoice,
    isSearchingInternet,
    setIsSearchingInternet,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    messageSteeringMode,
    setMessageSteeringMode,
    messageSteeringForceNarrate,
    setMessageSteeringForceNarrate,
    clearMessageSteering,
    textareaRef,
    selectedKnowledge,
    setSelectedKnowledge,
    ttsEnabled,
    temporaryChat,
    setTemporaryChat,
    useOCR,
    setUseOCR,
    defaultInternetSearchOn,
    history,
    uploadedFiles,
    contextFiles,
    fileRetrievalEnabled,
    setFileRetrievalEnabled: handleSetFileRetrievalEnabled,
    handleFileUpload,
    removeUploadedFile,
    clearUploadedFiles,
    actionInfo,
    setActionInfo,
    setContextFiles,
    createChatBranch,
    queuedMessages: storeQueuedMessages,
    addQueuedMessage: storeAddQueuedMessage,
    setQueuedMessages: storeSetQueuedMessages,
    clearQueuedMessages: storeClearQueuedMessages,
    serverChatId,
    setServerChatId,
    serverChatTitle,
    setServerChatTitle,
    serverChatCharacterId,
    setServerChatCharacterId,
    serverChatAssistantKind,
    setServerChatAssistantKind,
    serverChatAssistantId,
    setServerChatAssistantId,
    serverChatPersonaMemoryMode,
    setServerChatPersonaMemoryMode,
    serverChatMetaLoaded,
    setServerChatMetaLoaded,
    serverChatLoadState,
    setServerChatLoadState,
    serverChatLoadError,
    setServerChatLoadError,
    serverChatState,
    setServerChatState,
    serverChatVersion,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
    chatLoopState,
    ragMediaIds,
    setRagMediaIds,
    ragSearchMode,
    setRagSearchMode,
    ragTopK,
    setRagTopK,
    ragEnableGeneration,
    setRagEnableGeneration,
    ragEnableCitations,
    setRagEnableCitations,
    ragSources,
    setRagSources,
    ragPinnedResults,
    setRagPinnedResults,
    documentContext,
    compareMode,
    setCompareMode,
    compareFeatureEnabled,
    setCompareFeatureEnabled,
    compareSelectedModels,
    setCompareSelectedModels,
    compareSelectionByCluster,
    setCompareSelectionForCluster,
    compareActiveModelsByCluster,
    setCompareActiveModelsForCluster,
    sendPerModelReply,
    createCompareBranch,
    compareParentByHistory,
    setCompareParentForHistory,
    compareCanonicalByCluster,
    setCompareCanonicalForCluster,
    compareContinuationModeByCluster,
    setCompareContinuationModeForCluster,
    compareSplitChats,
    setCompareSplitChat,
    compareMaxModels,
    setCompareMaxModels,
    selectedCharacter,
    setSelectedCharacter,
    selectedAssistant: effectiveSelectedAssistant,
    setSelectedAssistant,
    replyTarget,
    clearReplyTarget,
  };
};
