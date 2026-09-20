import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { systemPromptForNonRag } from "~/services/tldw-server";
import { useStoreMessageOption, type Message } from "~/store/option";
import { useStoreMessage } from "~/store";
import { getContentFromCurrentTab } from "~/libs/get-html";
// RAG now uses tldw_server endpoints instead of local embeddings
import { ChatHistory, type MessageMetadataExtra } from "@/store/option";
import {
  deleteChatForEdit,
  deleteChatAfterMessageId,
  generateID,
  getPromptById,
  removeMessageByIndex,
  removeMessageById,
  updateMessageByIndex,
  updateMessageById,
} from "@/db/dexie/helpers";
import { useTranslation } from "react-i18next";
import { usePageAssist } from "@/context";
import { buildAssistantErrorContent } from "@/utils/chat-error-message";
import { buildCharacterChatAssistantErrorContent } from "@/hooks/chat/useCharacterChatMode";
import { createCharacterEmoteStream } from "@/hooks/chat/character-emote-stream";
import { detectCharacterMood } from "@/utils/character-mood";
import { useStorage } from "@plasmohq/storage/hook";
import { useStoreChatModelSettings } from "@/store/model";
import { getAllDefaultModelSettings } from "@/services/model-settings";
import { pageAssistModel } from "@/models";
import { getPrompt } from "@/services/application";
import { humanMessageFormatter } from "@/utils/human-message";
import { generateHistory } from "@/utils/generate-history";
import {
  tldwClient,
  type ConversationState,
} from "@/services/tldw/TldwApiClient";
import { resolveWebsiteChatContext } from "@/hooks/useMessage.website-context";
import { getScreenshotFromCurrentTab } from "@/libs/get-screenshot";
import {
  isReasoningEnded,
  isReasoningStarted,
  mergeReasoningContent,
  removeReasoning,
} from "@/libs/reasoning";
import { getModelNicknameByID } from "@/db/dexie/nickname";
import { systemPromptFormatter } from "@/utils/system-message";
import type { Character } from "@/types/character";
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter";
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant";
import {
  createBranchMessage,
  createRegenerateLastMessage,
} from "./handlers/messageHandlers";
import { consumeStreamingChunk } from "@/utils/streaming-chunks";
import type { CharacterEmoteEvent } from "@/utils/character-emotes";
import type { ToolCall } from "@/types/tool-calls";
import {
  createSaveMessageOnError,
  createSaveMessageOnSuccess,
  validateBeforeSubmit,
} from "./utils/messageHelpers";
import {
  buildMessageVariant,
  getLastUserMessageId,
  normalizeMessageVariants,
  updateActiveVariant,
} from "@/utils/message-variants";
import { resolveImageBackendCandidates } from "@/utils/image-backends";
import { normalChatMode } from "./chat-modes/normalChatMode";
import { tabChatMode } from "./chat-modes/tabChatMode";
import { documentChatMode } from "./chat-modes/documentChatMode";
import { updatePageTitle } from "@/utils/update-page-title";
import { useAntdNotification } from "./useAntdNotification";
import { useChatBaseState } from "@/hooks/chat/useChatBaseState";
import { useSelectedModel } from "@/hooks/chat/useSelectedModel";
import { normalizeConversationState } from "@/utils/conversation-state";
import {
  resolveApiProviderForModel,
  resolveExplicitProviderForSelectedModel,
} from "@/utils/resolve-api-provider";
import type { ChatDocuments } from "@/models/ChatTypes";
import type { UploadedFile } from "@/db/dexie/types";
import { applyMcpModuleDisclosureFromToolCalls } from "@/utils/mcp-disclosure";
import { normalizeChatModelId } from "@/utils/chat-model-availability";
import { validateSelectedChatModelAvailability } from "@/utils/chat-model-validation";
import { discardAbortedTurnIfRequested } from "@/hooks/chat/abort-turn-cleanup";
import { resolveSavedDegradedCharacterPersist } from "@/hooks/chat/characterPersistOutcome";
import { hydrateTrackedCharacterForSend } from "@/hooks/chat/tracked-character-hydration";
import {
  collectGreetings,
  isGreetingMessageType,
} from "@/utils/character-greetings";
import { resolveServerChatAssistantIdentity } from "@/hooks/chat/useServerChatLoader";
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state";
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord";
import {
  assistantSelectionToCharacter,
  characterToAssistantSelection,
  getAssistantSelectionMode,
  isPersonaAssistantSelection,
  personaToAssistantSelection,
  type AssistantSelection,
} from "@/types/assistant-selection";
import { ensurePersonaServerChat } from "@/hooks/chat/personaServerChat";
import { useChatLoopState } from "@/services/chat-loop/hooks";
import { subscribeChatLoopEvents } from "@/services/chat-loop/bridge";
import { extractChatLoopEvent } from "@/services/chat-loop/stream";
import { resolveUseMessageSendMode } from "@/hooks/useMessage.routing";
import { syncChatSettingsForServerChat } from "@/services/chat-settings";
import {
  loadServicePromptSnapshot,
  renderServicePromptPart,
  type ServicePromptSnapshot,
} from "@/services/service-prompts";
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error";

const extractToolCalls = (generationInfo: unknown): ToolCall[] | undefined => {
  if (!generationInfo || typeof generationInfo !== "object") return undefined;
  const candidate =
    (generationInfo as any).tool_calls ?? (generationInfo as any).toolCalls;
  return Array.isArray(candidate) ? (candidate as ToolCall[]) : undefined;
};

type ServerBackedMessage = Message & {
  serverMessageId?: string;
  serverMessageVersion?: number;
};

export const useMessage = () => {
  // Controllers come from Context (for aborting streaming requests)
  const {
    controller: abortController,
    setController: setAbortController,
    embeddingController,
    setEmbeddingController,
  } = usePageAssist();
  const discardCurrentTurnOnAbortRef = React.useRef(false);

  // Messages now come from Zustand store (single source of truth)
  const messages = useStoreMessageOption((state) => state.messages);
  const setMessages = useStoreMessageOption((state) => state.setMessages);

  const { t } = useTranslation("option");
  const queryClient = useQueryClient();
  const invalidateServerChatHistory = React.useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] });
  }, [queryClient]);
  const { selectedModel, setSelectedModel } = useSelectedModel();
  const currentChatModelSettings = useStoreChatModelSettings();
  const {
    setIsSearchingInternet,
    webSearch,
    setWebSearch,
    toolChoice,
    setToolChoice,
    isSearchingInternet,
    temporaryChat,
    setTemporaryChat,
    queuedMessages,
    addQueuedMessage,
    setQueuedMessages,
    clearQueuedMessages,
    fileRetrievalEnabled,
    setActionInfo,
    replyTarget,
    clearReplyTarget,
  } = useStoreMessageOption();
  const [defaultInternetSearchOn] = useStorage(
    "defaultInternetSearchOn",
    false,
  );

  const [defaultChatWithWebsite] = useStorage("defaultChatWithWebsite", false);

  const [chatWithWebsiteEmbedding] = useStorage(
    "chatWithWebsiteEmbedding",
    false,
  );
  const [maxWebsiteContext] = useStorage("maxWebsiteContext", 4028);
  const [selectedCharacter] = useSelectedCharacter<Character | null>(null);
  const [selectedAssistant, setSelectedAssistant] = useSelectedAssistant(null);

  const {
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
  const { currentURL, setCurrentURL } = useStoreMessage();
  const {
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
    serverChatState,
    setServerChatState,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
  } = useStoreMessageOption();
  const notification = useAntdNotification();
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
      avatar_url:
        effectiveAssistantState.avatarUrl ?? draftMetadata?.avatar_url ?? null,
      system_prompt:
        effectiveAssistantState.systemPromptSnapshot ??
        draftMetadata?.system_prompt ??
        null,
    };
  }, [effectiveAssistantState, selectedAssistant]);
  const [sidepanelTemporaryChat] = useStorage("sidepanelTemporaryChat", false);
  const [speechToTextLanguage, setSpeechToTextLanguage] = useStorage(
    "speechToTextLanguage",
    "en-US",
  );
  const {
    state: chatLoopState,
    dispatch: dispatchChatLoopEvent,
    reset: resetChatLoopState,
  } = useChatLoopState();

  React.useEffect(
    () => subscribeChatLoopEvents(dispatchChatLoopEvent),
    [dispatchChatLoopEvent],
  );

  const ensureSelectedChatModelIsAvailable = React.useCallback(
    async (selectedModelId: string): Promise<boolean> => {
      const normalizedSelectedModel = normalizeChatModelId(selectedModelId);
      if (!normalizedSelectedModel) {
        notification.error({
          message: t("error"),
          description: t("validationSelectModel"),
        });
        return false;
      }

      try {
        const validation = await validateSelectedChatModelAvailability(
          normalizedSelectedModel,
        );

        if (validation.status === "valid") {
          return true;
        }

        notification.warning({
          message: t("warning", "Warning"),
          description:
            validation.reason === "catalog-empty"
              ? t(
                  "playground:composer.validationModelCatalogUnavailableInline",
                  "Unable to verify model availability from the cached catalog. Continuing without refreshing; refresh models if this request fails.",
                )
              : t(
                  "playground:composer.validationModelUnavailableInline",
                  "Selected model may be stale or unavailable. Continuing without refreshing; refresh models or choose a different model if this request fails.",
                ),
        });
        return true;
      } catch (error) {
        console.error("Failed to validate selected model availability:", error);
        notification.warning({
          message: t("warning", "Warning"),
          description: t(
            "playground:composer.validationModelCatalogUnavailableInline",
            "Unable to verify model availability from the cached catalog. Continuing without refreshing; refresh models if this request fails.",
          ),
        });
        return true;
      }
    },
    [notification, t],
  );

  React.useEffect(() => {
    if (!serverChatId || serverChatMetaLoaded) return;
    const loadChatMeta = async () => {
      try {
        await tldwClient.initialize().catch(() => null);
        const chat = await tldwClient.getChat(serverChatId);
        const { assistantKind, assistantId, characterId, personaMemoryMode } =
          resolveServerChatAssistantIdentity(
            (chat as unknown as Record<string, unknown> | null) ?? null,
          );
        setServerChatTitle(String((chat as any)?.title || ""));
        setServerChatCharacterId(characterId);
        setServerChatAssistantKind(assistantKind);
        setServerChatAssistantId(assistantId);
        setServerChatPersonaMemoryMode(personaMemoryMode);
        setServerChatState(
          (chat as any)?.state ??
            (chat as any)?.conversation_state ??
            "in-progress",
        );
        setServerChatVersion((chat as any)?.version ?? null);
        setServerChatTopic((chat as any)?.topic_label ?? null);
        setServerChatClusterId((chat as any)?.cluster_id ?? null);
        setServerChatSource((chat as any)?.source ?? null);
        setServerChatExternalRef((chat as any)?.external_ref ?? null);
        if (assistantKind === "persona" && assistantId) {
          const profile = await tldwClient
            .getPersonaProfile(assistantId)
            .catch(() => null);
          await setSelectedAssistant(
            personaToAssistantSelection(
              (profile as Record<string, unknown> | null) ?? {
                id: assistantId,
                name:
                  (chat as any)?.assistant_name ??
                  (chat as any)?.title ??
                  "Persona",
              },
            ),
          );
        } else if (assistantKind === "character" && assistantId) {
          const character = await tldwClient
            .getCharacter(assistantId)
            .catch(() => null);
          await setSelectedAssistant(
            characterToAssistantSelection(
              (character as Record<string, unknown> | null) ?? {
                id: assistantId,
                name:
                  (chat as any)?.assistant_name ??
                  (chat as any)?.title ??
                  "Assistant",
              },
            ),
          );
        }
        setServerChatMetaLoaded(true);
      } catch {
        // ignore metadata hydration failures
      }
    };
    void loadChatMeta();
  }, [
    serverChatId,
    serverChatMetaLoaded,
    setSelectedAssistant,
    setServerChatAssistantId,
    setServerChatAssistantKind,
    setServerChatCharacterId,
    setServerChatClusterId,
    setServerChatExternalRef,
    setServerChatPersonaMemoryMode,
    setServerChatMetaLoaded,
    setServerChatSource,
    setServerChatState,
    setServerChatTitle,
    setServerChatTopic,
    setServerChatVersion,
  ]);

  // Local embedding store removed; rely on tldw_server RAG

  const clearChat = () => {
    stopStreamingRequest();
    setMessages([]);
    setHistory([]);
    setHistoryId(null);
    setIsFirstMessage(true);
    setIsLoading(false);
    setIsProcessing(false);
    setStreaming(false);
    updatePageTitle();
    currentChatModelSettings.reset();
    setServerChatId(null);
    if (defaultInternetSearchOn) {
      setWebSearch(true);
    }
    if (defaultChatWithWebsite) {
      setChatMode("rag");
    }
    if (sidepanelTemporaryChat) {
      setTemporaryChat(true);
    }
    clearReplyTarget();
  };

  const saveMessageOnSuccess = createSaveMessageOnSuccess(
    temporaryChat,
    setHistoryId as (
      id: string,
      options?: { preserveServerChatId?: boolean },
    ) => void,
    {
      onServerConversationLinked: (conversationId) => {
        setServerChatId(conversationId);
        setServerChatMetaLoaded(false);
        invalidateServerChatHistory();
        if (!serverChatId) {
          void syncChatSettingsForServerChat({
            historyId,
            serverChatId: conversationId,
            allowScratchFallback: true,
          }).catch(() => {
            // Best-effort scratch-to-server settings reconciliation.
          });
        }
      },
    }
  );
  const saveMessageOnError = createSaveMessageOnError(
    temporaryChat,
    history,
    setHistory,
    setHistoryId as (
      id: string,
      options?: { preserveServerChatId?: boolean },
    ) => void,
  );

  const chatWithWebsiteMode = async (
    message: string,
    image: string,
    isRegenerate: boolean,
    messages: Message[],
    history: ChatHistory,
    signal: AbortSignal,
    embeddingSignal: AbortSignal,
    servicePromptSnapshot: ServicePromptSnapshot,
    regenerateFromMessage?: Message,
  ) => {
    const executionSignal = servicePromptSnapshot.scopeSignal;
    const scopeInvalidatedSignal = servicePromptSnapshot.scopeInvalidatedSignal;
    const shouldAbortForScopeChange = () =>
      scopeInvalidatedSignal.aborted;
    const answerPrompt =
      servicePromptSnapshot.definitions["chat.rag.answer"];
    const rewritePrompt =
      servicePromptSnapshot.definitions["chat.rag.question_rewrite"];
    if (!answerPrompt || !rewritePrompt) {
      throw new Error("The Sidepanel RAG Service Prompt snapshot is incomplete.");
    }

    if (!selectedModel || selectedModel.trim().length === 0) {
      notification.error({
        message: t("error"),
        description: t("validationSelectModel"),
      });
      return;
    }

    const model = selectedModel.trim();
    setStreaming(true);
    const userDefaultModelSettings = await getAllDefaultModelSettings();

    const ollama = await pageAssistModel({
      model,
      requestScope: servicePromptSnapshot.requestScope,
    });

    let newMessage: Message[] = [];
    const resolvedAssistantMessageId = generateID();
    const resolvedUserMessageId = !isRegenerate ? generateID() : undefined;
    let generateMessageId = resolvedAssistantMessageId;
    const createdAt = Date.now();
    const modelInfo = await getModelNicknameByID(model);
    const fallbackParentMessageId = getLastUserMessageId(messages);
    const resolvedAssistantParentMessageId = isRegenerate
      ? (regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId)
      : (resolvedUserMessageId ?? null);
    const regenerateVariants =
      isRegenerate && regenerateFromMessage
        ? normalizeMessageVariants(regenerateFromMessage)
        : [];

    if (!isRegenerate) {
      newMessage = [
        ...messages,
        {
          isBot: false,
          name: "You",
          message,
          sources: [],
          images: [],
          createdAt,
          id: resolvedUserMessageId,
          parentMessageId: null,
        },
        {
          isBot: true,
          name: model,
          message: "▋",
          sources: [],
          createdAt,
          id: generateMessageId,
          modelImage: modelInfo?.model_avatar,
          modelName: modelInfo?.model_name || model,
          parentMessageId: resolvedAssistantParentMessageId ?? null,
        },
      ];
    } else {
      newMessage = [
        ...messages,
        {
          isBot: true,
          name: model,
          message: "▋",
          sources: [],
          createdAt,
          id: generateMessageId,
          modelImage: modelInfo?.model_avatar,
          modelName: modelInfo?.model_name || model,
          parentMessageId: resolvedAssistantParentMessageId ?? null,
        },
      ];
    }

    setMessages(newMessage);
    if (regenerateVariants.length > 0) {
      setMessages((prev) => {
        const next = [...prev];
        const lastIndex = next.findLastIndex(
          (msg) => msg.id === resolvedAssistantMessageId,
        );
        if (lastIndex >= 0) {
          const stub = next[lastIndex];
          const variants = [...regenerateVariants, buildMessageVariant(stub)];
          next[lastIndex] = {
            ...stub,
            variants,
            activeVariantIndex: variants.length - 1,
          };
        }
        return next;
      });
    }
    let fullText = "";
    let contentToSave = "";
    let embedURL = "";
    let embedHTML = "";
    let embedType = "html";
    let embedPDF: { content: string; page: number }[] = [];
    if (chatWithWebsiteEmbedding) {
      const {
        content: html,
        url: websiteUrl,
        type,
        pdf,
      } = await getContentFromCurrentTab(true);

      embedHTML = html;
      embedURL = websiteUrl;
      embedType = type;
      embedPDF = pdf;
      if (messages.length === 0) {
        setCurrentURL(websiteUrl);
      } else if (currentURL !== websiteUrl) {
        setCurrentURL(websiteUrl);
      } else {
        embedURL = currentURL;
      }
    }
    setMessages(newMessage);
    try {
      let query = message;
      if (newMessage.length > 2) {
        const lastTenMessages = newMessage.slice(-10);
        lastTenMessages.pop();
        const chat_history = lastTenMessages
          .map((message) => {
            return `${message.isBot ? "Assistant: " : "Human: "}${message.message}`;
          })
          .join("\n");
        const promptForQuestion = renderServicePromptPart(
          rewritePrompt.definition,
          "template",
          rewritePrompt.parts.template,
          { chat_history, question: message },
        );
        const questionOllama = await pageAssistModel({
          model,
          toolChoice: "none",
          tools: [],
          saveToDb: false,
          requestScope: servicePromptSnapshot.requestScope,
        });
        const questionMessage = await humanMessageFormatter({
          content: [
            {
              text: promptForQuestion,
              type: "text",
            },
          ],
          model,
          useOCR,
        });
        const response = await questionOllama.invoke([questionMessage], {
          signal: executionSignal,
        });
        query = response.content.toString();
        query = removeReasoning(query);
      }

      let context: string = "";
      let source: {
        name: any;
        type: any;
        mode: string;
        url: string;
        pageContent: string;
        metadata: Record<string, any>;
      }[] = [];

      if (chatWithWebsiteEmbedding) {
        const websiteContext = await resolveWebsiteChatContext({
          client: tldwClient,
          embedURL,
          embedType,
          embedHTML,
          embedPDF,
          maxWebsiteContext,
          query,
          signal: executionSignal,
          requestScope: servicePromptSnapshot.requestScope,
        });
        context = websiteContext.context;
        source = websiteContext.source;
      }

      let humanMessage = await humanMessageFormatter({
        content: [
          {
            text: renderServicePromptPart(
              answerPrompt.definition,
              "template",
              answerPrompt.parts.template,
              { context, question: query },
            ),
            type: "text",
          },
        ],
        model,
        useOCR,
      });

      const applicationChatHistory = generateHistory(history, model);

      let generationInfo: any | undefined = undefined;

      const chunks = await ollama.stream(
        [...applicationChatHistory, humanMessage],
        {
          signal: executionSignal,
          callbacks: [
            {
              handleLLMEnd(output: any): any {
                try {
                  generationInfo = output?.generations?.[0][0]?.generationInfo;
                } catch (e) {
                  console.error("handleLLMEnd error", e);
                }
              },
            },
          ],
        },
      );
      let count = 0;
      let reasoningStartTime: Date | null = null;
      let reasoningEndTime: Date | null = null;
      let timetaken = 0;
      let apiReasoning = false;
      for await (const chunk of chunks) {
        const chunkState = consumeStreamingChunk(
          { fullText, contentToSave, apiReasoning },
          chunk,
        );
        fullText = chunkState.fullText;
        contentToSave = chunkState.contentToSave;
        apiReasoning = chunkState.apiReasoning;
        if (count === 0) {
          setIsProcessing(true);
        }
        if (isReasoningStarted(fullText) && !reasoningStartTime) {
          reasoningStartTime = new Date();
        }

        if (
          reasoningStartTime &&
          !reasoningEndTime &&
          isReasoningEnded(fullText)
        ) {
          reasoningEndTime = new Date();
          const reasoningTime =
            reasoningEndTime.getTime() - reasoningStartTime.getTime();
          timetaken = reasoningTime;
        }
        setMessages((prev) => {
          return prev.map((message) => {
            if (message.id === generateMessageId) {
              return updateActiveVariant(message, {
                message: fullText + "▋",
                reasoning_time_taken: timetaken,
              });
            }
            return message;
          });
        });
        count++;
      }

      if (executionSignal.aborted) {
        const abortError = new Error("Request cancelled");
        abortError.name = "AbortError";
        throw abortError;
      }

      const toolCalls = extractToolCalls(generationInfo);
      applyMcpModuleDisclosureFromToolCalls(toolCalls);
      setMessages((prev) => {
        return prev.map((message) => {
          if (message.id === generateMessageId) {
            return updateActiveVariant(message, {
              message: fullText,
              sources: source,
              generationInfo,
              toolCalls,
              reasoning_time_taken: timetaken,
            });
          }
          return message;
        });
      });

      setHistory([
        ...history,
        {
          role: "user",
          content: message,
          image,
        },
        {
          role: "assistant",
          content: fullText,
        },
      ]);

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate,
        selectedModel: model,
        message,
        image,
        fullText,
        source,
        message_source: "copilot",
        generationInfo,
        reasoning_time_taken: timetaken,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
        scopeSignal: servicePromptSnapshot.scopeSignal,
        scopeInvalidatedSignal:
          servicePromptSnapshot.scopeInvalidatedSignal,
        requestScope: servicePromptSnapshot.requestScope,
      });

      if (scopeInvalidatedSignal.aborted) {
        const abortError = new Error("Request scope changed");
        abortError.name = "AbortError";
        throw abortError;
      }

      setIsProcessing(false);
      setStreaming(false);
    } catch (e) {
      const scopeAborted = shouldAbortForScopeChange();
      if (scopeAborted || isRequestConfigScopeChangedError(e)) {
        setMessages(messages);
        setHistory(history);
        setIsProcessing(false);
        setStreaming(false);
        setIsEmbedding(false);
        return;
      }
      if (
        discardAbortedTurnIfRequested({
          discardRequested: discardCurrentTurnOnAbortRef.current,
          error: e,
          previousMessages: messages,
          previousHistory: history,
          setMessages,
          setHistory,
        })
      ) {
        setIsProcessing(false);
        setStreaming(false);
        setIsEmbedding(false);
        return;
      }

      console.error(e);
      const assistantContent = buildAssistantErrorContent(fullText, e);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === generateMessageId
            ? updateActiveVariant(msg, { message: assistantContent })
            : msg,
        ),
      );
      let errorSave: string | null;
      try {
        errorSave = await saveMessageOnError({
          e,
          botMessage: assistantContent,
          history,
          historyId,
          image,
          selectedModel: model,
          setHistory,
          setHistoryId,
          userMessage: message,
          isRegenerating: isRegenerate,
          message_source: "copilot",
          userMessageId: resolvedUserMessageId,
          assistantMessageId: resolvedAssistantMessageId,
          assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
          scopeSignal: servicePromptSnapshot.scopeSignal,
          scopeInvalidatedSignal:
            servicePromptSnapshot.scopeInvalidatedSignal,
          requestScope: servicePromptSnapshot.requestScope,
          shouldAbortForScopeChange,
        });
      } catch (persistenceError) {
        if (
          shouldAbortForScopeChange() ||
          isRequestConfigScopeChangedError(persistenceError)
        ) {
          setMessages(messages);
          setHistory(history);
          setIsProcessing(false);
          setStreaming(false);
          setIsEmbedding(false);
          return;
        }
        throw persistenceError;
      }

      if (shouldAbortForScopeChange()) {
        setMessages(messages);
        setHistory(history);
        setIsProcessing(false);
        setStreaming(false);
        setIsEmbedding(false);
        return;
      }

      if (!errorSave) {
        notification.error({
          message: t("error"),
          description: e?.message || t("somethingWentWrong"),
        });
      }
      setIsProcessing(false);
      setStreaming(false);
      setIsEmbedding(false);
    } finally {
      discardCurrentTurnOnAbortRef.current = false;
      setEmbeddingController(null);
    }
  };

  const visionChatMode = async (
    message: string,
    image: string,
    isRegenerate: boolean,
    messages: Message[],
    history: ChatHistory,
    signal: AbortSignal,
    regenerateFromMessage?: Message,
  ) => {
    if (!selectedModel || selectedModel.trim().length === 0) {
      notification.error({
        message: t("error"),
        description: t("validationSelectModel"),
      });
      return;
    }

    const model = selectedModel.trim();
    setStreaming(true);
    const ollama = await pageAssistModel({ model });

    let newMessage: Message[] = [];
    const resolvedAssistantMessageId = generateID();
    const resolvedUserMessageId = !isRegenerate ? generateID() : undefined;
    let generateMessageId = resolvedAssistantMessageId;
    const createdAt = Date.now();
    const modelInfo = await getModelNicknameByID(model);
    const fallbackParentMessageId = getLastUserMessageId(messages);
    const resolvedAssistantParentMessageId = isRegenerate
      ? (regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId)
      : (resolvedUserMessageId ?? null);
    const regenerateVariants =
      isRegenerate && regenerateFromMessage
        ? normalizeMessageVariants(regenerateFromMessage)
        : [];

    if (!isRegenerate) {
      newMessage = [
        ...messages,
        {
          isBot: false,
          name: "You",
          message,
          sources: [],
          images: [],
          createdAt,
          id: resolvedUserMessageId,
          parentMessageId: null,
        },
        {
          isBot: true,
          name: model,
          message: "▋",
          sources: [],
          createdAt,
          id: generateMessageId,
          modelImage: modelInfo?.model_avatar,
          modelName: modelInfo?.model_name || model,
          parentMessageId: resolvedAssistantParentMessageId ?? null,
        },
      ];
    } else {
      newMessage = [
        ...messages,
        {
          isBot: true,
          name: model,
          message: "▋",
          sources: [],
          createdAt,
          id: generateMessageId,
          modelImage: modelInfo?.model_avatar,
          modelName: modelInfo?.model_name || model,
          parentMessageId: resolvedAssistantParentMessageId ?? null,
        },
      ];
    }
    setMessages(newMessage);
    if (regenerateVariants.length > 0) {
      setMessages((prev) => {
        const next = [...prev];
        const lastIndex = next.findLastIndex(
          (msg) => msg.id === resolvedAssistantMessageId,
        );
        if (lastIndex >= 0) {
          const stub = next[lastIndex];
          const variants = [...regenerateVariants, buildMessageVariant(stub)];
          next[lastIndex] = {
            ...stub,
            variants,
            activeVariantIndex: variants.length - 1,
          };
        }
        return next;
      });
    }
    let fullText = "";
    let contentToSave = "";

    try {
      const prompt = await systemPromptForNonRag();
      const selectedPrompt = await getPromptById(selectedSystemPrompt);

      const applicationChatHistory = [];
      // Inject selected character's system prompt at highest priority
      if (selectedCharacter?.system_prompt) {
        applicationChatHistory.unshift(
          await systemPromptFormatter({
            content: selectedCharacter.system_prompt,
          }),
        );
      }

      const data = await getScreenshotFromCurrentTab();

      const visionImage = data?.screenshot || "";

      if (visionImage === "") {
        throw new Error(
          data?.error ||
            "Please close and reopen the side panel. This is a bug that will be fixed soon.",
        );
      }

      if (!selectedCharacter?.system_prompt && prompt && !selectedPrompt) {
        applicationChatHistory.unshift(
          await systemPromptFormatter({
            content: prompt,
          }),
        );
      }
      if (!selectedCharacter?.system_prompt && selectedPrompt) {
        const selectedPromptContent =
          selectedPrompt.system_prompt ?? selectedPrompt.content;
        applicationChatHistory.unshift(
          await systemPromptFormatter({
            content: selectedPromptContent,
          }),
        );
      }

      let humanMessage = await humanMessageFormatter({
        content: [
          {
            text: message,
            type: "text",
          },
          {
            image_url: visionImage,
            type: "image_url",
          },
        ],
        model,
        useOCR,
      });

      let generationInfo: any | undefined = undefined;

      const chunks = await ollama.stream(
        [...applicationChatHistory, humanMessage],
        {
          signal: signal,
          callbacks: [
            {
              handleLLMEnd(output: any): any {
                try {
                  generationInfo = output?.generations?.[0][0]?.generationInfo;
                } catch (e) {
                  console.error("handleLLMEnd error", e);
                }
              },
            },
          ],
        },
      );
      let count = 0;
      let reasoningStartTime: Date | undefined = undefined;
      let reasoningEndTime: Date | undefined = undefined;
      let timetaken = 0;
      let apiReasoning = false;
      for await (const chunk of chunks) {
        const chunkState = consumeStreamingChunk(
          { fullText, contentToSave, apiReasoning },
          chunk,
        );
        fullText = chunkState.fullText;
        contentToSave = chunkState.contentToSave;
        apiReasoning = chunkState.apiReasoning;
        if (count === 0) {
          setIsProcessing(true);
        }
        if (isReasoningStarted(fullText) && !reasoningStartTime) {
          reasoningStartTime = new Date();
        }

        if (
          reasoningStartTime &&
          !reasoningEndTime &&
          isReasoningEnded(fullText)
        ) {
          reasoningEndTime = new Date();
          const reasoningTime =
            reasoningEndTime.getTime() - reasoningStartTime.getTime();
          timetaken = reasoningTime;
        }
        setMessages((prev) => {
          return prev.map((message) => {
            if (message.id === generateMessageId) {
              return updateActiveVariant(message, {
                message: fullText + "▋",
                reasoning_time_taken: timetaken,
              });
            }
            return message;
          });
        });
        count++;
      }
      const toolCalls = extractToolCalls(generationInfo);
      applyMcpModuleDisclosureFromToolCalls(toolCalls);
      setMessages((prev) => {
        return prev.map((message) => {
          if (message.id === generateMessageId) {
            return updateActiveVariant(message, {
              message: fullText,
              generationInfo,
              toolCalls,
              reasoning_time_taken: timetaken,
            });
          }
          return message;
        });
      });

      setHistory([
        ...history,
        {
          role: "user",
          content: message,
        },
        {
          role: "assistant",
          content: fullText,
        },
      ]);

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate,
        selectedModel: model,
        message,
        image,
        fullText,
        source: [],
        message_source: "copilot",
        generationInfo,
        reasoning_time_taken: timetaken,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
      });

      setIsProcessing(false);
      setStreaming(false);
    } catch (e) {
      if (
        discardAbortedTurnIfRequested({
          discardRequested: discardCurrentTurnOnAbortRef.current,
          error: e,
          previousMessages: messages,
          previousHistory: history,
          setMessages,
          setHistory,
        })
      ) {
        setIsProcessing(false);
        setStreaming(false);
        setIsEmbedding(false);
        return;
      }

      const assistantContent = buildAssistantErrorContent(fullText, e);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === generateMessageId
            ? updateActiveVariant(msg, { message: assistantContent })
            : msg,
        ),
      );
      const errorSave = await saveMessageOnError({
        e,
        botMessage: assistantContent,
        history,
        historyId,
        image,
        selectedModel: model,
        setHistory,
        setHistoryId,
        userMessage: message,
        isRegenerating: isRegenerate,
        message_source: "copilot",
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
      });

      if (!errorSave) {
        notification.error({
          message: t("error"),
          description: e?.message || t("somethingWentWrong"),
        });
      }
      setIsProcessing(false);
      setStreaming(false);
      setIsEmbedding(false);
    } finally {
      discardCurrentTurnOnAbortRef.current = false;
      setAbortController(null);
      setEmbeddingController(null);
    }
  };

  const characterChatMode = async (
    message: string,
    image: string,
    isRegenerate: boolean,
    messages: Message[],
    history: ChatHistory,
    signal: AbortSignal,
    model: string,
    characterOverride?: Character | null,
    regenerateFromMessage?: Message,
    serverChatIdOverride?: string | null,
    controller?: AbortController,
  ) => {
    setStreaming(true);
    const activeCharacter = characterOverride ?? selectedCharacter;
    const resolveGreetingText = (): string => {
      const fromMessages = messages.find(
        (entry) =>
          entry.isBot &&
          isGreetingMessageType(entry.messageType) &&
          typeof entry.message === "string" &&
          entry.message.trim().length > 0,
      );
      if (fromMessages?.message) {
        return fromMessages.message.trim();
      }

      const fromHistory = history.find(
        (entry) =>
          entry.role === "assistant" &&
          isGreetingMessageType(entry.messageType) &&
          typeof entry.content === "string" &&
          entry.content.trim().length > 0,
      );
      if (fromHistory?.content) {
        return fromHistory.content.trim();
      }

      const fromCharacter = collectGreetings(activeCharacter as any).find(
        (candidate) =>
          typeof candidate === "string" && candidate.trim().length > 0,
      );
      if (fromCharacter) {
        return fromCharacter.trim();
      }

      return "";
    };
    const greetingText = resolveGreetingText();
    const hasGreetingInHistory =
      greetingText.length > 0 &&
      history.some(
        (entry) =>
          entry.role === "assistant" &&
          typeof entry.content === "string" &&
          entry.content.trim() === greetingText,
      );
    const historyBase: ChatHistory =
      greetingText.length > 0 && !hasGreetingInHistory
        ? [
            {
              role: "assistant",
              content: greetingText,
              messageType: "character:greeting",
            },
            ...history,
          ]
        : history;
    let fullText = "";
    let contentToSave = "";
    let inactivityTimer: ReturnType<typeof setTimeout> | null = null;
    const resolvedAssistantMessageId = generateID();
    const resolvedUserMessageId = !isRegenerate ? generateID() : undefined;
    let persistedUserServerMessageId: string | undefined;
    let generateMessageId = resolvedAssistantMessageId;
    const fallbackParentMessageId = getLastUserMessageId(messages);
    const resolvedAssistantParentMessageId = isRegenerate
      ? (regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId)
      : (resolvedUserMessageId ?? null);
    const regenerateVariants =
      isRegenerate && regenerateFromMessage
        ? normalizeMessageVariants(regenerateFromMessage)
        : [];

    if (!activeCharacter?.id) {
      throw new Error("No character selected");
    }
    const emoteStream = createCharacterEmoteStream();
    const getExplicitEmoteMoodLabel = () =>
      emoteStream.events.length > 0
        ? emoteStream.events[emoteStream.events.length - 1]?.state
        : undefined;
    const getExplicitEmoteMetadataExtra = (): MessageMetadataExtra | undefined => {
      const moodLabel = getExplicitEmoteMoodLabel();
      return moodLabel
        ? {
            mood_label: moodLabel,
            mood_confidence: null,
            mood_topic: null,
            emote_events: [...emoteStream.events],
          }
        : undefined;
    };
    const applyEmoteEventsToMessage = (
      newEvents: CharacterEmoteEvent[],
    ) => {
      if (newEvents.length === 0) return;
      const moodLabel = getExplicitEmoteMoodLabel();
      setMessages((prev) =>
        prev.map((m) =>
          m.id === generateMessageId
            ? updateActiveVariant(m, {
                moodLabel,
                metadataExtra: {
                  ...(m.metadataExtra ?? {}),
                  emote_events: [...emoteStream.events],
                },
              })
            : m,
        ),
      );
    };
    const flushCharacterEmoteStream = () => {
      const flushedEmotes = emoteStream.flush({ fullText, contentToSave });
      fullText = flushedEmotes.fullText;
      contentToSave = flushedEmotes.contentToSave;
      applyEmoteEventsToMessage(flushedEmotes.emoteEvents);
    };

    try {
      const hasImageInput =
        typeof image === "string" && image.trim().length > 0;
      if (!isRegenerate && message.trim().length === 0 && !hasImageInput) {
        notification.error({
          message: t("error"),
          description: t(
            "playground:composer.validationMessageRequired",
            "Type a message before sending.",
          ),
        });
        setIsProcessing(false);
        setStreaming(false);
        return;
      }

      await tldwClient.initialize();

      // Visual placeholder
      const modelInfo = await getModelNicknameByID(model);
      const characterName =
        activeCharacter?.name || modelInfo?.model_name || model;
      const characterAvatar =
        activeCharacter?.avatar_url || modelInfo?.model_avatar;
      const createdAt = Date.now();
      const hasGreetingInMessages = messages.some((entry) => {
        if (!entry?.isBot) return false;
        if (isGreetingMessageType(entry?.messageType)) return true;
        if (!greetingText) return false;
        return (
          typeof entry.message === "string" &&
          entry.message.trim() === greetingText
        );
      });
      const greetingSeedMessage: Message | null =
        greetingText.length > 0 && !hasGreetingInMessages
          ? {
              isBot: true,
              role: "assistant",
              name: characterName,
              message: greetingText,
              messageType: "character:greeting",
              sources: [],
              createdAt,
              id: generateID(),
              modelImage: characterAvatar,
              modelName: characterName,
            }
          : null;
      const chatMessagesBase = greetingSeedMessage
        ? [greetingSeedMessage, ...messages]
        : messages;
      const assistantStub: Message = {
        isBot: true,
        name: characterName,
        message: "▋",
        sources: [],
        createdAt,
        id: generateMessageId,
        modelImage: characterAvatar,
        modelName: characterName,
        parentMessageId: resolvedAssistantParentMessageId ?? null,
      };
      if (regenerateVariants.length > 0) {
        const variants = [
          ...regenerateVariants,
          buildMessageVariant(assistantStub),
        ];
        assistantStub.variants = variants;
        assistantStub.activeVariantIndex = variants.length - 1;
      }

      const newMessageList: Message[] = !isRegenerate
        ? [
            ...chatMessagesBase,
            {
              isBot: false,
              name: "You",
              message,
              sources: [],
              images: [],
              createdAt,
              id: resolvedUserMessageId,
              parentMessageId: null,
            },
            assistantStub,
          ]
        : [...chatMessagesBase, assistantStub];
      setMessages(newMessageList);

      const overrideChatId =
        typeof serverChatIdOverride === "string" &&
        serverChatIdOverride.trim().length > 0
          ? serverChatIdOverride.trim()
          : null;

      // Ensure server chat session exists
      let chatId = overrideChatId || serverChatId;
      let createdNewChat = false;
      if (!chatId) {
        type TldwChatMeta =
          | {
              id?: string | number;
              chat_id?: string | number;
              state?: string;
              conversation_state?: string;
              topic_label?: string | null;
              cluster_id?: string | null;
              source?: string | null;
              external_ref?: string | null;
            }
          | string
          | number
          | null
          | undefined;

        const created = (await tldwClient.createChat({
          character_id: activeCharacter.id,
          state: serverChatState || "in-progress",
          topic_label: serverChatTopic || undefined,
          cluster_id: serverChatClusterId || undefined,
          source: serverChatSource || undefined,
          external_ref: serverChatExternalRef || undefined,
        })) as TldwChatMeta;

        let rawId: string | number | undefined;
        if (created && typeof created === "object") {
          const {
            id,
            chat_id,
            version,
            state,
            conversation_state,
            topic_label,
            cluster_id,
            source,
            external_ref,
          } = created as {
            id?: string | number;
            chat_id?: string | number;
            version?: number;
            state?: string | null;
            conversation_state?: string | null;
            topic_label?: string | null;
            cluster_id?: string | null;
            source?: string | null;
            external_ref?: string | null;
          };
          rawId = id ?? chat_id;
          const normalizedState = normalizeConversationState(
            state ?? conversation_state ?? null,
          );
          setServerChatState(normalizedState);
          setServerChatVersion(typeof version === "number" ? version : null);
          setServerChatTopic(topic_label ?? null);
          setServerChatClusterId(cluster_id ?? null);
          setServerChatSource(source ?? null);
          setServerChatExternalRef(external_ref ?? null);
        } else if (typeof created === "string" || typeof created === "number") {
          rawId = created;
        }

        const normalizedId = rawId != null ? String(rawId) : "";
        if (!normalizedId) {
          throw new Error("Failed to create character chat session");
        }
        chatId = normalizedId;
        createdNewChat = true;
        setServerChatId(normalizedId);
        setServerChatTitle(String((created as any)?.title || ""));
        setServerChatCharacterId(
          (created as any)?.character_id ?? activeCharacter?.id ?? null,
        );
        setServerChatMetaLoaded(true);
        invalidateServerChatHistory();
      }

      if (createdNewChat && !isRegenerate && greetingText.length > 0) {
        try {
          const createdGreeting = (await tldwClient.addChatMessage(chatId, {
            role: "assistant",
            content: greetingText,
          })) as { id?: string | number; version?: number } | null;
          if (createdGreeting?.id != null) {
            setMessages((prev) => {
              const updated = [...prev] as ServerBackedMessage[];
              const serverMessageId = String(createdGreeting.id);
              const serverMessageVersion = createdGreeting.version;
              for (let i = 0; i < updated.length; i += 1) {
                if (
                  updated[i]?.isBot &&
                  isGreetingMessageType(updated[i]?.messageType) &&
                  !updated[i]?.serverMessageId
                ) {
                  updated[i] = {
                    ...updated[i],
                    serverMessageId,
                    serverMessageVersion,
                  };
                  break;
                }
              }
              return updated as Message[];
            });
          }
        } catch (greetingPersistError) {
          console.warn(
            "Failed to persist character greeting for new chat:",
            greetingPersistError,
          );
        }
      }

      // Add user message to server (only if not regenerate)
      if (!isRegenerate) {
        type TldwChatMessage = {
          id?: string | number;
          version?: number;
          role?: string;
          content?: string;
          image_base64?: string;
        };

        const payload: TldwChatMessage = { role: "user" };
        const trimmedUserMessage = message.trim();
        if (trimmedUserMessage.length > 0) {
          payload.content = message;
        }
        let normalizedImage = image;
        if (
          normalizedImage.length > 0 &&
          !normalizedImage.startsWith("data:")
        ) {
          const payloadValue = normalizedImage.includes(",")
            ? normalizedImage.split(",")[1]
            : normalizedImage;
          if (payloadValue !== undefined && payloadValue.length > 0) {
            normalizedImage = `data:image/jpeg;base64,${payloadValue}`;
          }
        }
        if (normalizedImage && normalizedImage.startsWith("data:")) {
          const b64 = normalizedImage.includes(",")
            ? normalizedImage.split(",")[1]
            : normalizedImage;
          if (b64) {
            payload.image_base64 = b64;
          }
        }
        if (payload.content || payload.image_base64) {
          const createdUser = (await tldwClient.addChatMessage(
            chatId,
            payload,
          )) as TldwChatMessage | null;
          persistedUserServerMessageId =
            createdUser?.id != null ? String(createdUser.id) : undefined;
          setMessages((prev) => {
            const updated = [...prev] as ServerBackedMessage[];
            const serverMessageId =
              createdUser?.id != null ? String(createdUser.id) : undefined;
            const serverMessageVersion = createdUser?.version;
            for (let i = updated.length - 1; i >= 0; i--) {
              if (!updated[i].isBot) {
                updated[i] = {
                  ...updated[i],
                  serverMessageId,
                  serverMessageVersion,
                };
                break;
              }
            }
            return updated as Message[];
          });
        }
      }

      // Stream completion from server /chats/{id}/complete-v2
      let count = 0;
      let reasoningStartTime: Date | null = null;
      let reasoningEndTime: Date | null = null;
      let timetaken = 0;
      let apiReasoning = false;

      const explicitProvider = resolveExplicitProviderForSelectedModel({
        currentSelectedModel: selectedModel,
        requestedSelectedModel: model,
        explicitProvider: currentChatModelSettings.apiProvider,
      });
      const resolvedApiProvider = await resolveApiProviderForModel({
        modelId: model,
        explicitProvider,
      });

      const normalizedModel = model.replace(/^tldw:/, "").trim();
      const resolvedModel =
        normalizedModel.length > 0 ? normalizedModel : model;

      const shouldPersistToServer = !temporaryChat;
      // Stream-inactivity watchdog: if no chunk arrives within the timeout, abort
      // the underlying stream so a stalled character response cannot hang forever.
      const STREAM_INACTIVITY_TIMEOUT_MS = 60_000;
      let inactivityAborted = false;
      const resetInactivityTimer = () => {
        if (inactivityTimer) clearTimeout(inactivityTimer);
        inactivityTimer = setTimeout(() => {
          inactivityAborted = true;
          controller?.abort();
        }, STREAM_INACTIVITY_TIMEOUT_MS);
      };
      resetInactivityTimer();
      for await (const chunk of tldwClient.streamCharacterChatCompletion(
        chatId,
        {
          include_character_context: true,
          model: resolvedModel,
          provider: resolvedApiProvider,
          save_to_db: shouldPersistToServer,
        },
        { signal },
      )) {
        resetInactivityTimer();
        const loopEvent = extractChatLoopEvent(chunk);
        if (loopEvent) {
          dispatchChatLoopEvent(loopEvent);
        }
        const previousFullText = fullText;
        const chunkState = consumeStreamingChunk(
          { fullText, contentToSave, apiReasoning },
          chunk,
        );
        const sanitizedChunkState = emoteStream.sanitizeChunk(chunkState);
        fullText = sanitizedChunkState.fullText;
        contentToSave = sanitizedChunkState.contentToSave;
        apiReasoning = sanitizedChunkState.apiReasoning;
        applyEmoteEventsToMessage(sanitizedChunkState.emoteEvents);

        if (chunkState.token && fullText !== previousFullText) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === generateMessageId
                ? updateActiveVariant(m, {
                    message: fullText + "▋",
                    reasoning_time_taken: timetaken,
                  })
                : m,
            ),
          );
        }
        if (count === 0) setIsProcessing(true);

        if (isReasoningStarted(fullText) && !reasoningStartTime) {
          reasoningStartTime = new Date();
        }

        if (
          reasoningStartTime &&
          !reasoningEndTime &&
          isReasoningEnded(fullText)
        ) {
          reasoningEndTime = new Date();
          const reasoningTime =
            reasoningEndTime.getTime() - reasoningStartTime.getTime();
          timetaken = reasoningTime;
        }

        count++;
        if (signal?.aborted) break;
      }
      if (inactivityTimer) clearTimeout(inactivityTimer);
      flushCharacterEmoteStream();

      if (inactivityAborted) {
        const timeoutError = new Error(
          "Stream timed out: no data received for 60 seconds",
        );
        (timeoutError as any).name = "StreamInactivityTimeout";
        throw timeoutError;
      }

      if (signal?.aborted) {
        const abortError = new Error("AbortError");
        (abortError as any).name = "AbortError";
        throw abortError;
      }
      setMessages((prev) =>
        prev.map((m) =>
          m.id === generateMessageId
            ? updateActiveVariant(m, {
                message: fullText,
                reasoning_time_taken: timetaken,
              })
            : m,
        ),
      );

      // Persist assistant reply on server
      const finalContent = contentToSave || fullText;
      const finalPersistedContent = finalContent.trim();
      if (finalPersistedContent.length > 0) {
        let fallbackSpeakerId: number | undefined;
        let speakerCharacterId: number | undefined;
        let detectedMood: ReturnType<typeof detectCharacterMood> | undefined;
        let resolvedMoodLabel: string | undefined;
        let resolvedMoodConfidence: number | undefined;
        let resolvedMoodTopic: string | undefined;
        let metadataExtra: MessageMetadataExtra | undefined;
        try {
          fallbackSpeakerId = Number.parseInt(
            String(activeCharacter.id),
            10,
          );
          speakerCharacterId =
            Number.isFinite(fallbackSpeakerId) && fallbackSpeakerId > 0
              ? fallbackSpeakerId
              : undefined;
          if (emoteStream.events.length > 0) {
            resolvedMoodLabel =
              emoteStream.events[emoteStream.events.length - 1]?.state;
          } else {
            detectedMood = detectCharacterMood({
              assistantText: finalPersistedContent,
              userText: message,
            });
            resolvedMoodLabel = detectedMood.label;
            resolvedMoodConfidence =
              typeof detectedMood.confidence === "number" &&
              Number.isFinite(detectedMood.confidence)
                ? detectedMood.confidence
                : undefined;
            resolvedMoodTopic =
              typeof detectedMood.topic === "string" &&
              detectedMood.topic.trim()
                ? detectedMood.topic.trim()
                : undefined;
          }
          const persistPayload: Record<string, unknown> = {
            assistant_content: finalPersistedContent,
            assistant_message_id: generateMessageId,
            speaker_character_id: speakerCharacterId,
            speaker_character_name: characterName,
          };
          if (resolvedMoodLabel) {
            persistPayload.mood_label = resolvedMoodLabel;
          }
          if (typeof resolvedMoodConfidence === "number") {
            persistPayload.mood_confidence = resolvedMoodConfidence;
          }
          if (resolvedMoodTopic) {
            persistPayload.mood_topic = resolvedMoodTopic;
          }
          if (emoteStream.events.length > 0) {
            persistPayload.emote_events = [...emoteStream.events];
          }
          if (persistedUserServerMessageId) {
            persistPayload.user_message_id = persistedUserServerMessageId;
          }
          metadataExtra = {
            speaker_character_id: speakerCharacterId ?? null,
            speaker_character_name: characterName,
            mood_label: resolvedMoodLabel,
            mood_confidence: resolvedMoodConfidence ?? null,
            mood_topic: resolvedMoodTopic ?? null,
            ...(emoteStream.events.length > 0
              ? { emote_events: [...emoteStream.events] }
              : {}),
          };
          const persisted = (await tldwClient.persistCharacterCompletion(
            chatId,
            persistPayload,
          )) as {
            assistant_message_id?: string | number;
            message_id?: string | number;
            id?: string | number;
            version?: number;
          } | null;
          const createdAsstServerId =
            persisted?.assistant_message_id ??
            persisted?.message_id ??
            persisted?.id;
          const createdAsstVersion = persisted?.version;
          setMessages(
            (prev) =>
              (prev as ServerBackedMessage[]).map((m) => {
                if (m.id !== generateMessageId) return m;
                const serverMessageId =
                  createdAsstServerId != null
                    ? String(createdAsstServerId)
                    : undefined;
                return updateActiveVariant(m, {
                  serverMessageId,
                  serverMessageVersion: createdAsstVersion,
                  metadataExtra,
                  speakerCharacterId: speakerCharacterId ?? null,
                  speakerCharacterName: characterName,
                  moodLabel: resolvedMoodLabel,
                  moodConfidence: resolvedMoodConfidence ?? null,
                  moodTopic: resolvedMoodTopic ?? null,
                });
              }) as Message[],
          );
        } catch (e) {
          console.error(
            "Failed to persist assistant message via completions/persist:",
            e,
          );
          const savedOutcome = resolveSavedDegradedCharacterPersist(e);
          if (savedOutcome?.saved) {
            setMessages(
              (prev) =>
                (prev as ServerBackedMessage[]).map((m) => {
                  if (m.id !== generateMessageId) return m;
                  const serverMessageId =
                    savedOutcome.assistantMessageId != null
                      ? String(savedOutcome.assistantMessageId)
                      : undefined;
                  return updateActiveVariant(m, {
                    serverMessageId,
                    serverMessageVersion: savedOutcome.version,
                    metadataExtra,
                    speakerCharacterId: speakerCharacterId ?? null,
                    speakerCharacterName: characterName,
                    moodLabel: resolvedMoodLabel,
                    moodConfidence: resolvedMoodConfidence ?? null,
                    moodTopic: resolvedMoodTopic ?? null,
                  });
                }) as Message[],
            );
          } else {
            try {
              const createdAsst = (await tldwClient.addChatMessage(chatId, {
                role: "assistant",
                content: finalPersistedContent,
                ...(metadataExtra ? { metadata_extra: metadataExtra } : {}),
              })) as { id?: string | number; version?: number } | null;
              setMessages(
                (prev) =>
                  (prev as ServerBackedMessage[]).map((m) => {
                    if (m.id !== generateMessageId) return m;
                    const serverMessageId =
                      createdAsst?.id != null
                        ? String(createdAsst.id)
                        : undefined;
                    return updateActiveVariant(m, {
                      serverMessageId,
                      serverMessageVersion: createdAsst?.version,
                      metadataExtra,
                      speakerCharacterId: speakerCharacterId ?? null,
                      speakerCharacterName: characterName,
                      moodLabel: resolvedMoodLabel,
                      moodConfidence: resolvedMoodConfidence ?? null,
                      moodTopic: resolvedMoodTopic ?? null,
                    });
                  }) as Message[],
              );
            } catch (fallbackError) {
              console.error(
                "Failed fallback assistant persistence with addChatMessage:",
                fallbackError,
              );
            }
          }
        }
      } else {
        console.warn(
          "Skipping assistant persistence because completion content is empty.",
        );
      }

      const lastEntry = historyBase[historyBase.length - 1];
      const prevEntry = historyBase[historyBase.length - 2];
      const endsWithUser =
        lastEntry?.role === "user" && lastEntry.content === message;
      const endsWithUserAssistant =
        lastEntry?.role === "assistant" &&
        prevEntry?.role === "user" &&
        prevEntry.content === message;

      if (isRegenerate) {
        if (endsWithUser) {
          setHistory([
            ...historyBase,
            { role: "assistant", content: finalContent },
          ]);
        } else if (endsWithUserAssistant) {
          setHistory(
            historyBase.map((entry, index) =>
              index === historyBase.length - 1 && entry.role === "assistant"
                ? { ...entry, content: finalContent }
                : entry,
            ),
          );
        } else {
          setHistory([
            ...historyBase,
            { role: "user", content: message, image },
            { role: "assistant", content: finalContent },
          ]);
        }
      } else {
        setHistory([
          ...historyBase,
          { role: "user", content: message, image },
          { role: "assistant", content: finalContent },
        ]);
      }

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate,
        selectedModel: model,
        modelId: model,
        message,
        image,
        fullText,
        source: [],
        message_source: "copilot",
        reasoning_time_taken: timetaken,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
        conversationId: chatId,
      });

      setIsProcessing(false);
      setStreaming(false);
    } catch (e) {
      if (
        discardAbortedTurnIfRequested({
          discardRequested: discardCurrentTurnOnAbortRef.current,
          error: e,
          previousMessages: messages,
          previousHistory: history,
          setMessages,
          setHistory,
        })
      ) {
        setIsProcessing(false);
        setStreaming(false);
        return;
      }

      flushCharacterEmoteStream();
      const recoveryMoodLabel = getExplicitEmoteMoodLabel();
      const recoveryMetadataExtra = getExplicitEmoteMetadataExtra();
      const assistantContent = buildCharacterChatAssistantErrorContent(
        fullText,
        e,
        t,
      );
      if (generateMessageId) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === generateMessageId
              ? updateActiveVariant(msg, {
                  message: assistantContent,
                  ...(recoveryMetadataExtra
                    ? {
                        metadataExtra: {
                          ...(msg.metadataExtra ?? {}),
                          ...recoveryMetadataExtra,
                        },
                        moodLabel: recoveryMoodLabel,
                        moodConfidence: null,
                        moodTopic: null,
                      }
                    : {}),
                })
              : msg,
          ),
        );
      }
      const errorSave = await saveMessageOnError({
        e,
        botMessage: assistantContent,
        history: historyBase,
        historyId,
        image,
        selectedModel: model,
        modelId: model,
        setHistory,
        setHistoryId,
        userMessage: message,
        isRegenerating: isRegenerate,
        message_source: "copilot",
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
      });

      if (!errorSave) {
        const isInactivityTimeout =
          e instanceof Error && (e as any).name === "StreamInactivityTimeout";
        notification.error({
          message: isInactivityTimeout
            ? t("playground:streamTimeout", {
                defaultValue: "Stream timed out",
              })
            : t("error"),
          description: e?.message || t("somethingWentWrong"),
        });
      }
      setIsProcessing(false);
      setStreaming(false);
    } finally {
      discardCurrentTurnOnAbortRef.current = false;
      if (inactivityTimer) clearTimeout(inactivityTimer);
      setAbortController((current: AbortController | null) =>
        current === controller ? null : current,
      );
    }
  };

  // Web search mode removed - use tldw_server for search functionality

  const presetChatMode = async (
    message: string,
    image: string,
    isRegenerate: boolean,
    messages: Message[],
    history: ChatHistory,
    signal: AbortSignal,
    messageType: string,
    regenerateFromMessage?: Message,
    options?: {
      selectedModel?: string | null;
      useOCR?: boolean;
    },
  ) => {
    const resolvedPresetModel = (
      typeof options?.selectedModel === "string"
        ? options.selectedModel
        : selectedModel || ""
    ).trim();
    const resolvedPresetUseOCR =
      typeof options?.useOCR === "boolean" ? options.useOCR : useOCR;

    if (!resolvedPresetModel) {
      notification.error({
        message: t("error"),
        description: t("validationSelectModel"),
      });
      return;
    }

    const model = resolvedPresetModel;
    setStreaming(true);

    if (image.length > 0) {
      if (!image.startsWith("data:")) {
        const payload = image.includes(",") ? image.split(",")[1] : image;
        if (payload && payload.length > 0) {
          image = `data:image/jpeg;base64,${payload}`;
        }
      }
    }

    const ollama = await pageAssistModel({ model });

    let newMessage: Message[] = [];
    const resolvedAssistantMessageId = generateID();
    const resolvedUserMessageId = !isRegenerate ? generateID() : undefined;
    let generateMessageId = resolvedAssistantMessageId;
    const createdAt = Date.now();
    const modelInfo = await getModelNicknameByID(model);
    const fallbackParentMessageId = getLastUserMessageId(messages);
    const resolvedAssistantParentMessageId = isRegenerate
      ? (regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId)
      : (resolvedUserMessageId ?? null);
    const regenerateVariants =
      isRegenerate && regenerateFromMessage
        ? normalizeMessageVariants(regenerateFromMessage)
        : [];

    if (!isRegenerate) {
      newMessage = [
        ...messages,
        {
          isBot: false,
          name: "You",
          message,
          sources: [],
          images: [image],
          createdAt,
          id: resolvedUserMessageId,
          messageType: messageType,
          parentMessageId: null,
        },
        {
          isBot: true,
          name: model,
          message: "▋",
          sources: [],
          createdAt,
          id: generateMessageId,
          modelImage: modelInfo?.model_avatar,
          modelName: modelInfo?.model_name || model,
          parentMessageId: resolvedAssistantParentMessageId ?? null,
        },
      ];
    } else {
      newMessage = [
        ...messages,
        {
          isBot: true,
          name: model,
          message: "▋",
          sources: [],
          createdAt,
          id: generateMessageId,
          modelImage: modelInfo?.model_avatar,
          modelName: modelInfo?.model_name || model,
          parentMessageId: resolvedAssistantParentMessageId ?? null,
        },
      ];
    }
    setMessages(newMessage);
    if (regenerateVariants.length > 0) {
      setMessages((prev) => {
        const next = [...prev];
        const lastIndex = next.findLastIndex(
          (msg) => msg.id === resolvedAssistantMessageId,
        );
        if (lastIndex >= 0) {
          const stub = next[lastIndex];
          const variants = [...regenerateVariants, buildMessageVariant(stub)];
          next[lastIndex] = {
            ...stub,
            variants,
            activeVariantIndex: variants.length - 1,
          };
        }
        return next;
      });
    }
    let fullText = "";
    let contentToSave = "";

    try {
      const prompt = await getPrompt(messageType);
      let humanMessage = await humanMessageFormatter({
        content: [
          {
            text: prompt.replace("{text}", message),
            type: "text",
          },
        ],
        model,
        useOCR: resolvedPresetUseOCR,
      });
      if (image.length > 0) {
        humanMessage = await humanMessageFormatter({
          content: [
            {
              text: prompt.replace("{text}", message),
              type: "text",
            },
            {
              image_url: image,
              type: "image_url",
            },
          ],
          model,
          useOCR: resolvedPresetUseOCR,
        });
      }

      let generationInfo: any | undefined = undefined;

      const chunks = await ollama.stream([humanMessage], {
        signal: signal,
        callbacks: [
          {
            handleLLMEnd(output: any): any {
              try {
                generationInfo = output?.generations?.[0][0]?.generationInfo;
              } catch (e) {
                console.error("handleLLMEnd error", e);
              }
            },
          },
        ],
      });
      let count = 0;
      let reasoningStartTime: Date | null = null;
      let reasoningEndTime: Date | null = null;
      let timetaken = 0;
      let apiReasoning = false;
      for await (const chunk of chunks) {
        const chunkState = consumeStreamingChunk(
          { fullText, contentToSave, apiReasoning },
          chunk,
        );
        fullText = chunkState.fullText;
        contentToSave = chunkState.contentToSave;
        apiReasoning = chunkState.apiReasoning;
        if (count === 0) {
          setIsProcessing(true);
        }
        if (isReasoningStarted(fullText) && !reasoningStartTime) {
          reasoningStartTime = new Date();
        }

        if (
          reasoningStartTime &&
          !reasoningEndTime &&
          isReasoningEnded(fullText)
        ) {
          reasoningEndTime = new Date();
          const reasoningTime =
            reasoningEndTime.getTime() - reasoningStartTime.getTime();
          timetaken = reasoningTime;
        }
        setMessages((prev) => {
          return prev.map((message) => {
            if (message.id === generateMessageId) {
              return updateActiveVariant(message, {
                message: fullText + "▋",
                reasoning_time_taken: timetaken,
              });
            }
            return message;
          });
        });
        count++;
      }

      const toolCalls = extractToolCalls(generationInfo);
      applyMcpModuleDisclosureFromToolCalls(toolCalls);
      setMessages((prev) => {
        return prev.map((message) => {
          if (message.id === generateMessageId) {
            return updateActiveVariant(message, {
              message: fullText,
              generationInfo,
              toolCalls,
              reasoning_time_taken: timetaken,
            });
          }
          return message;
        });
      });

      setHistory([
        ...history,
        {
          role: "user",
          content: message,
          image,
          messageType,
        },
        {
          role: "assistant",
          content: fullText,
        },
      ]);

      await saveMessageOnSuccess({
        historyId,
        setHistoryId,
        isRegenerate,
        selectedModel: model,
        message,
        image,
        fullText,
        source: [],
        message_source: "copilot",
        message_type: messageType,
        generationInfo,
        reasoning_time_taken: timetaken,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
      });

      setIsProcessing(false);
      setStreaming(false);
    } catch (e) {
      if (
        discardAbortedTurnIfRequested({
          discardRequested: discardCurrentTurnOnAbortRef.current,
          error: e,
          previousMessages: messages,
          previousHistory: history,
          setMessages,
          setHistory,
        })
      ) {
        setIsProcessing(false);
        setStreaming(false);
        return;
      }

      const assistantContent = buildAssistantErrorContent(fullText, e);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === generateMessageId
            ? updateActiveVariant(msg, { message: assistantContent })
            : msg,
        ),
      );
      const errorSave = await saveMessageOnError({
        e,
        botMessage: assistantContent,
        history,
        historyId,
        image,
        selectedModel: model,
        setHistory,
        setHistoryId,
        userMessage: message,
        isRegenerating: isRegenerate,
        message_source: "copilot",
        message_type: messageType,
        userMessageId: resolvedUserMessageId,
        assistantMessageId: resolvedAssistantMessageId,
        assistantParentMessageId: resolvedAssistantParentMessageId ?? null,
      });

      if (!errorSave) {
        notification.error({
          message: t("error"),
          description: e?.message || t("somethingWentWrong"),
        });
      }
      setIsProcessing(false);
      setStreaming(false);
    } finally {
      discardCurrentTurnOnAbortRef.current = false;
      setAbortController(null);
    }
  };

  const onSubmit = async ({
    message,
    image,
    isRegenerate,
    controller,
    memory,
    messages: chatHistory,
    messageType,
    regenerateFromMessage,
    docs,
    uploadedFiles,
    imageBackendOverride,
    requestOverrides,
    serverChatIdOverride,
  }: {
    message: string;
    image: string;
    isRegenerate?: boolean;
    messages?: Message[];
    memory?: ChatHistory;
    controller?: AbortController;
    messageType?: string;
    regenerateFromMessage?: Message;
    docs?: ChatDocuments;
    uploadedFiles?: UploadedFile[];
    imageBackendOverride?: string;
    requestOverrides?: {
      selectedModel?: string | null;
      selectedSystemPrompt?: string | null;
      toolChoice?: "auto" | "none" | "required";
      useOCR?: boolean;
      webSearch?: boolean;
      chatMode?: "normal" | "rag" | "vision";
      historyForModel?: ChatHistory;
      messageForModel?: string;
    };
    serverChatIdOverride?: string | null;
  }) => {
    const trimmedImageBackendOverride =
      typeof imageBackendOverride === "string"
        ? imageBackendOverride.trim()
        : "";
    const hasExplicitImageBackend = trimmedImageBackendOverride.length > 0;
    const resolvedSelectedModel =
      typeof requestOverrides?.selectedModel === "string" &&
      requestOverrides.selectedModel.trim().length > 0
        ? requestOverrides.selectedModel.trim()
        : selectedModel || "";
    const resolvedChatMode =
      requestOverrides?.chatMode === "normal" ||
      requestOverrides?.chatMode === "rag" ||
      requestOverrides?.chatMode === "vision"
        ? requestOverrides.chatMode
        : chatMode;
    const model =
      (hasExplicitImageBackend
        ? trimmedImageBackendOverride || resolvedSelectedModel
        : resolvedSelectedModel
      ).trim() || "image-generation";
    const imageBackendCandidates = hasExplicitImageBackend
      ? [trimmedImageBackendOverride]
      : resolveImageBackendCandidates(
          currentChatModelSettings?.apiProvider,
          model,
        );
    const usesLegacySidepanelRag =
      !hasExplicitImageBackend &&
      imageBackendCandidates.length === 0 &&
      (!uploadedFiles || uploadedFiles.length === 0) &&
      (!docs || docs.length === 0) &&
      !messageType &&
      resolvedChatMode === "rag";
    const activeController = controller ?? new AbortController();
    const signal = activeController.signal;
    const releaseActiveController = () =>
      setAbortController((current: AbortController | null) =>
        current === activeController ? null : current,
      );
    let servicePromptSnapshot: ServicePromptSnapshot | undefined;
    if (usesLegacySidepanelRag) {
      setAbortController(activeController);
      try {
        servicePromptSnapshot = await loadServicePromptSnapshot(
          ["chat.rag.answer", "chat.rag.question_rewrite"],
          { signal },
        );
      } catch (error) {
        releaseActiveController();
        const errorName =
          typeof error === "object" && error !== null && "name" in error
            ? error.name
            : undefined;
        if (signal.aborted || errorName === "AbortError") {
          return;
        }
        if (
          typeof error === "object" &&
          error !== null &&
          "code" in error &&
          error.code === "service_prompt_migration_required"
        ) {
          notification.warning({
            message: t(
              "workflowPromptsReviewRequired",
              "Workflow prompts need review",
            ),
            description: (
              <>
                Browser-local prompt values must be imported or discarded.{" "}
                <a
                  href="/options.html#/settings/prompt"
                  target="_blank"
                  rel="noreferrer"
                >
                  Review workflow prompts
                </a>
              </>
            ),
          });
          return;
        }
        notification.error({
          message: t(
            "workflowPromptsUnavailable",
            "Workflow prompts unavailable",
          ),
          description: t(
            "workflowPromptsUnavailableDescription",
            "Unable to load workflow prompts from your tldw server. Check the connection and try again.",
          ),
        });
        return;
      }
    }

    let replyActive = false;
    try {
      resetChatLoopState();
    const resolvedSelectedSystemPrompt =
      requestOverrides &&
      Object.prototype.hasOwnProperty.call(
        requestOverrides,
        "selectedSystemPrompt",
      )
        ? (requestOverrides.selectedSystemPrompt ?? "")
        : (selectedSystemPrompt ?? "");
    const resolvedToolChoice =
      requestOverrides?.toolChoice === "auto" ||
      requestOverrides?.toolChoice === "required" ||
      requestOverrides?.toolChoice === "none"
        ? requestOverrides.toolChoice
        : toolChoice;
    const resolvedUseOCR =
      typeof requestOverrides?.useOCR === "boolean"
        ? requestOverrides.useOCR
        : useOCR;
    const resolvedWebSearch =
      typeof requestOverrides?.webSearch === "boolean"
        ? requestOverrides.webSearch
        : webSearch;
    const conversationContextOverrides = {
      historyForModel: Array.isArray(requestOverrides?.historyForModel)
        ? requestOverrides.historyForModel
        : undefined,
      messageForModel:
        typeof requestOverrides?.messageForModel === "string"
          ? requestOverrides.messageForModel
          : undefined,
    };
    if (!hasExplicitImageBackend) {
      if (!validateBeforeSubmit(resolvedSelectedModel, t, notification)) {
        return;
      }
      let modelAvailable: boolean;
      modelAvailable = await ensureSelectedChatModelIsAvailable(
        resolvedSelectedModel,
      );
      if (!modelAvailable) {
        return;
      }
    }

    if (!usesLegacySidepanelRag) setAbortController(activeController);
    replyActive =
      Boolean(replyTarget) &&
      !isRegenerate &&
      !messageType &&
      resolvedChatMode === "normal" &&
      !selectedCharacter?.id;
    const replyOverrides = replyActive
      ? (() => {
          const userMessageId = generateID();
          const assistantMessageId = generateID();
          return {
            userMessageId,
            assistantMessageId,
            userParentMessageId: replyTarget?.id ?? null,
            assistantParentMessageId: userMessageId,
          };
        })()
      : {};

      if (hasExplicitImageBackend || imageBackendCandidates.length > 0) {
        await normalChatMode(
          message,
          image,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          {
            selectedModel: model,
            useOCR: resolvedUseOCR,
            selectedSystemPrompt: resolvedSelectedSystemPrompt,
            currentChatModelSettings,
            setMessages,
            saveMessageOnSuccess,
            saveMessageOnError,
            setHistory,
            setIsProcessing,
            setStreaming,
            setAbortController,
            historyId,
            setHistoryId: setHistoryId as (
              id: string,
              options?: { preserveServerChatId?: boolean },
            ) => void,
            webSearch: resolvedWebSearch,
            setIsSearchingInternet,
            uploadedFiles: hasExplicitImageBackend ? [] : uploadedFiles,
            imageBackendOverride: hasExplicitImageBackend
              ? trimmedImageBackendOverride
              : undefined,
            regenerateFromMessage,
            ...conversationContextOverrides,
            ...replyOverrides,
          },
        );
        return;
      }
      if (uploadedFiles && uploadedFiles.length > 0) {
        await documentChatMode(
          message,
          image,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          uploadedFiles,
          {
            selectedModel: model,
            useOCR: resolvedUseOCR,
            currentChatModelSettings,
            toolChoice: resolvedToolChoice,
            setMessages,
            saveMessageOnSuccess,
            saveMessageOnError,
            setHistory,
            setIsProcessing,
            setStreaming,
            setAbortController,
            historyId: historyId ?? null,
            setHistoryId,
            fileRetrievalEnabled,
            setActionInfo,
            regenerateFromMessage,
            ...replyOverrides,
          },
        );
        return;
      }
      if (docs && docs.length > 0) {
        await tabChatMode(
          message,
          image,
          docs,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          {
            selectedModel: model,
            useOCR: resolvedUseOCR,
            selectedSystemPrompt: resolvedSelectedSystemPrompt,
            toolChoice: resolvedToolChoice,
            setMessages,
            saveMessageOnSuccess,
            saveMessageOnError,
            setHistory,
            setIsProcessing,
            setStreaming,
            setAbortController,
            historyId: historyId ?? null,
            setHistoryId,
            regenerateFromMessage,
            ...replyOverrides,
          },
        );
        return;
      }
      // this means that the user is trying to send something from a selected text on the web
      if (messageType) {
        await presetChatMode(
          message,
          image,
          isRegenerate,
          chatHistory || messages,
          memory || history,
          signal,
          messageType,
          regenerateFromMessage,
          {
            selectedModel: resolvedSelectedModel,
            useOCR: resolvedUseOCR,
          },
        );
      } else {
        if (resolvedChatMode === "normal") {
          const sendMode = resolveUseMessageSendMode({
            effectiveMode: effectiveAssistantState.mode,
            hasEffectiveAssistant: Boolean(effectiveSelectedAssistant),
            draftAssistantKind: selectedAssistant?.kind ?? null,
            draftAssistantSelectionMode: getAssistantSelectionMode(selectedAssistant),
          });
          const trackedCharacterForSend =
            sendMode === "tracked_character"
              ? (() => {
                  if (
                    selectedCharacter?.id != null &&
                    String(selectedCharacter.id) === effectiveAssistantState.id
                  ) {
                    return selectedCharacter;
                  }
                  return assistantSelectionToCharacter<Character & Record<string, unknown>>(
                    effectiveSelectedAssistant,
                  );
                })()
              : null;

          if (sendMode === "tracked_character" && trackedCharacterForSend?.id) {
            const hydratedTrackedCharacter =
              await hydrateTrackedCharacterForSend(
                trackedCharacterForSend,
                (characterId) => tldwClient.getCharacter(characterId),
              );
            await characterChatMode(
              message,
              image,
              isRegenerate,
              chatHistory || messages,
              memory || history,
              signal,
              model,
              hydratedTrackedCharacter,
              regenerateFromMessage,
              serverChatIdOverride,
              activeController,
            );
          } else if (sendMode === "tracked_persona" && isPersonaAssistantSelection(effectiveSelectedAssistant)) {
            const personaServerChat = await ensurePersonaServerChat({
              assistant: effectiveSelectedAssistant,
              serverChatIdOverride,
              serverChatId,
              serverChatTitle,
              serverChatAssistantKind,
              serverChatAssistantId,
              serverChatPersonaMemoryMode,
              serverChatMetaLoaded,
              serverChatState,
              serverChatTopic,
              serverChatClusterId,
              serverChatSource,
              serverChatExternalRef,
              historyId,
              temporaryChat,
              createChat: (payload) => tldwClient.createChat(payload),
              ensureServerChatHistoryId: async () => historyId,
              invalidateServerChatHistory,
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
            });
            await normalChatMode(
              message,
              image,
              isRegenerate,
              chatHistory || messages,
              memory || history,
              signal,
              {
                selectedModel: model,
                useOCR: resolvedUseOCR,
                selectedSystemPrompt: resolvedSelectedSystemPrompt,
                currentChatModelSettings,
                setMessages,
                saveMessageOnError,
                setHistory,
                setIsProcessing,
                setStreaming,
                setAbortController,
                historyId: personaServerChat.historyId,
                setHistoryId: setHistoryId as (
                  id: string,
                  options?: { preserveServerChatId?: boolean },
                ) => void,
                assistantIdentity: {
                  name: effectiveSelectedAssistant.name,
                  avatarUrl:
                    typeof effectiveSelectedAssistant.avatar_url === "string"
                      ? effectiveSelectedAssistant.avatar_url
                      : undefined,
                },
                saveMessageOnSuccess: (data) =>
                  saveMessageOnSuccess({
                    ...data,
                    conversationId: personaServerChat.chatId,
                  }),
                webSearch: resolvedWebSearch,
                setIsSearchingInternet,
                uploadedFiles,
                regenerateFromMessage,
                ...conversationContextOverrides,
                ...replyOverrides,
              },
            );
          } else if (sendMode === "overlay" && effectiveSelectedAssistant) {
            await normalChatMode(
              message,
              image,
              isRegenerate,
              chatHistory || messages,
              memory || history,
              signal,
              {
                selectedModel: model,
                useOCR: resolvedUseOCR,
                selectedSystemPrompt: resolvedSelectedSystemPrompt,
                currentChatModelSettings,
                setMessages,
                saveMessageOnSuccess,
                saveMessageOnError,
                setHistory,
                setIsProcessing,
                setStreaming,
                setAbortController,
                historyId,
                setHistoryId: setHistoryId as (
                  id: string,
                  options?: { preserveServerChatId?: boolean },
                ) => void,
                assistantIdentity: {
                  name: effectiveSelectedAssistant.name,
                  avatarUrl:
                    typeof effectiveSelectedAssistant.avatar_url === "string"
                      ? effectiveSelectedAssistant.avatar_url
                      : undefined,
                },
                overlaySystemPrompt:
                  effectiveAssistantState.systemPromptSnapshot ?? undefined,
                webSearch: resolvedWebSearch,
                setIsSearchingInternet,
                regenerateFromMessage,
                ...conversationContextOverrides,
                ...replyOverrides,
              },
            );
          } else {
            await normalChatMode(
              message,
              image,
              isRegenerate,
              chatHistory || messages,
              memory || history,
              signal,
              {
                selectedModel: model,
                useOCR: resolvedUseOCR,
                selectedSystemPrompt: resolvedSelectedSystemPrompt,
                currentChatModelSettings,
                setMessages,
                saveMessageOnSuccess,
                saveMessageOnError,
                setHistory,
                setIsProcessing,
                setStreaming,
                setAbortController,
                historyId,
                setHistoryId: setHistoryId as (
                  id: string,
                  options?: { preserveServerChatId?: boolean },
                ) => void,
                webSearch: resolvedWebSearch,
                setIsSearchingInternet,
                regenerateFromMessage,
                ...conversationContextOverrides,
                ...replyOverrides,
              },
            );
          }
        } else if (resolvedChatMode === "vision") {
          await visionChatMode(
            message,
            image,
            isRegenerate,
            chatHistory || messages,
            memory || history,
            signal,
            regenerateFromMessage,
          );
        } else {
          const newEmbeddingController = new AbortController();
          let embeddingSignal = newEmbeddingController.signal;
          setEmbeddingController(newEmbeddingController);
          await chatWithWebsiteMode(
            message,
            image,
            isRegenerate,
            chatHistory || messages,
            memory || history,
            signal,
            embeddingSignal,
            servicePromptSnapshot!,
            regenerateFromMessage,
          );
        }
      }
    } finally {
      if (usesLegacySidepanelRag) {
        servicePromptSnapshot!.release();
        releaseActiveController();
      }
      if (replyActive) {
        clearReplyTarget();
      }
    }
  };

  const stopStreamingRequest = (options?: unknown) => {
    const discardTurn =
      typeof options === "object" &&
      options !== null &&
      "discardTurn" in options &&
      options.discardTurn === true;
    if (
      discardTurn &&
      (isEmbedding || abortController || embeddingController)
    ) {
      discardCurrentTurnOnAbortRef.current = true;
    }
    if (isEmbedding) {
      if (embeddingController) {
        embeddingController.abort();
        setEmbeddingController(null);
      }
    }
    if (abortController) {
      abortController.abort();
      setAbortController(null);
    }
  };

  const editMessage = async (
    index: number,
    message: string,
    isHuman: boolean,
    isSend = true,
  ) => {
    const newHistory = history;

    if (isHuman) {
      const currentHumanMessage = (messages as ServerBackedMessage[])[index];
      const updatedMessages = messages.map((msg, idx) =>
        idx === index ? { ...msg, message } : msg,
      );
      if (!isSend) {
        setMessages(updatedMessages);
        const updatedHistory = newHistory.map((item, idx) =>
          idx === index ? { ...item, content: message } : item,
        );
        setHistory(updatedHistory);
        // TASK-12104: address the Dexie row by stable id (not UI array index),
        // which is offset by any non-persisted greeting seed at UI index 0.
        if (currentHumanMessage?.id) {
          await updateMessageById(historyId, currentHumanMessage.id, message);
        } else {
          await updateMessageByIndex(historyId, index, message);
        }
        if (
          serverChatId &&
          (selectedCharacter?.id || serverChatAssistantKind === "persona") &&
          currentHumanMessage?.serverMessageId
        ) {
          try {
            const srv = await tldwClient.getMessage(
              currentHumanMessage.serverMessageId,
            );
            const ver = srv?.version;
            if (ver != null) {
              await tldwClient.editMessage(
                currentHumanMessage.serverMessageId,
                message,
                Number(ver),
                serverChatId ?? undefined,
              );
            }
          } catch {}
        }
        return;
      }
      const previousMessages = updatedMessages.slice(0, index + 1);
      setMessages(previousMessages);
      const previousHistory = newHistory.slice(0, index);
      setHistory(previousHistory);
      // TASK-12104: id-address the edited row (and delete what follows it) so a
      // greeting-offset UI index cannot overwrite/keep the wrong Dexie rows.
      if (currentHumanMessage?.id) {
        await updateMessageById(historyId, currentHumanMessage.id, message);
        await deleteChatAfterMessageId(historyId, currentHumanMessage.id);
      } else {
        await updateMessageByIndex(historyId, index, message);
        await deleteChatForEdit(historyId, index);
      }
      // Server-backed edit and cleanup
      if (
        serverChatId &&
        (selectedCharacter?.id || serverChatAssistantKind === "persona")
      ) {
        if (currentHumanMessage?.serverMessageId) {
          try {
            const srv = await tldwClient.getMessage(
              currentHumanMessage.serverMessageId,
            );
            const ver = srv?.version;
            if (ver != null) {
              await tldwClient.editMessage(
                currentHumanMessage.serverMessageId,
                message,
                Number(ver),
                serverChatId ?? undefined,
              );
            }
          } catch {}
        }
        try {
          const res: any = await tldwClient.listChatMessages(serverChatId, {
            include_deleted: "false",
          });
          const list: any[] = Array.isArray(res) ? res : res?.messages || [];
          const serverIds = list.map((m: any) => m.id);
          const targetSrvId = currentHumanMessage?.serverMessageId;
          const startIdx = targetSrvId ? serverIds.indexOf(targetSrvId) : -1;
          if (startIdx >= 0) {
            for (let i = startIdx + 1; i < list.length; i++) {
              const m = list[i];
              try {
                await tldwClient.deleteMessage(
                  m.id,
                  Number(m.version),
                  serverChatId ?? undefined,
                );
              } catch {}
            }
          }
        } catch {}
      }
      const abortController = new AbortController();
      await onSubmit({
        message: message,
        image: currentHumanMessage.images[0] || "",
        isRegenerate: true,
        messages: previousMessages,
        memory: previousHistory,
        controller: abortController,
      });
    } else {
      // Assistant message edited
      const currentAssistant = (messages as ServerBackedMessage[])[index];
      const updatedMessages = messages.map((msg, idx) =>
        idx === index ? { ...msg, message } : msg,
      );
      setMessages(updatedMessages);
      const updatedHistory = newHistory.map((item, idx) =>
        idx === index ? { ...item, content: message } : item,
      );
      setHistory(updatedHistory);
      // TASK-12104: address the assistant Dexie row by stable id.
      if (currentAssistant?.id) {
        await updateMessageById(historyId, currentAssistant.id, message);
      } else {
        await updateMessageByIndex(historyId, index, message);
      }
      // Server-backed: update assistant server message too
      if (
        serverChatId &&
        (selectedCharacter?.id || serverChatAssistantKind === "persona") &&
        currentAssistant?.serverMessageId
      ) {
        try {
          const srv = await tldwClient.getMessage(
            currentAssistant.serverMessageId,
          );
          const ver = srv?.version;
          if (ver != null) {
            await tldwClient.editMessage(
              currentAssistant.serverMessageId,
              message,
              Number(ver),
              serverChatId ?? undefined,
            );
          }
        } catch {}
      }
    }
  };

  const deleteMessage = React.useCallback(
    async (index: number) => {
      const target = messages[index];
      if (!target) return;

      const targetId = target.serverMessageId ?? target.id;
      if (replyTarget?.id && targetId && replyTarget.id === targetId) {
        clearReplyTarget();
      }

      if (target.serverMessageId) {
        await tldwClient.initialize().catch(() => null);
        let expectedVersion = target.serverMessageVersion;
        if (expectedVersion == null) {
          const serverMessage = await tldwClient.getMessage(
            target.serverMessageId,
          );
          expectedVersion = serverMessage?.version;
        }
        if (expectedVersion == null) {
          throw new Error("Missing server message version");
        }
        await tldwClient.deleteMessage(
          target.serverMessageId,
          Number(expectedVersion),
          serverChatId ?? undefined,
        );
        invalidateServerChatHistory();
      }

      if (historyId) {
        // TASK-12104: remove the Dexie row by stable id so a non-persisted
        // greeting at UI index 0 does not shift us onto the wrong row. An
        // unsaved greeting's id is absent from Dexie, so this is a safe no-op.
        if (target.id) {
          await removeMessageById(historyId, target.id);
        } else {
          await removeMessageByIndex(historyId, index);
        }
      }

      setMessages(messages.filter((_, idx) => idx !== index));
      setHistory(history.filter((_, idx) => idx !== index));
    },
    [
      clearReplyTarget,
      history,
      historyId,
      invalidateServerChatHistory,
      messages,
      replyTarget?.id,
      serverChatId,
      setHistory,
      setMessages,
    ],
  );

  const createChatBranch = createBranchMessage({
    notification,
    historyId,
    setHistory,
    setHistoryId,
    setMessages,
    setSelectedSystemPrompt,
    setSystemPrompt: currentChatModelSettings.setSystemPrompt,
    serverChatId,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatMetaLoaded,
    serverChatState,
    setServerChatState,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
    onServerChatMutated: invalidateServerChatHistory,
    characterId: selectedCharacter?.id ?? null,
    chatTitle: serverChatTitle ?? null,
    messages,
    history,
  });

  const createServerOnlyChatBranch = createBranchMessage({
    notification,
    historyId,
    setHistory,
    setHistoryId,
    setMessages,
    setSelectedSystemPrompt,
    setSystemPrompt: currentChatModelSettings.setSystemPrompt,
    serverChatId,
    setServerChatId,
    setServerChatTitle,
    setServerChatCharacterId,
    setServerChatMetaLoaded,
    serverChatState,
    setServerChatState,
    setServerChatVersion,
    serverChatTopic,
    setServerChatTopic,
    serverChatClusterId,
    setServerChatClusterId,
    serverChatSource,
    setServerChatSource,
    serverChatExternalRef,
    setServerChatExternalRef,
    onServerChatMutated: invalidateServerChatHistory,
    characterId: selectedCharacter?.id ?? null,
    chatTitle: serverChatTitle ?? null,
    messages,
    history,
    serverOnly: true,
  });

  const regenerateLastMessage = createRegenerateLastMessage({
    validateBeforeSubmitFn: () => true,
    history,
    messages,
    setHistory,
    setMessages,
    onSubmit,
    beforeSubmit: async ({ nextMessages }) => {
      if (!serverChatId) return;
      if (selectedCharacter?.id == null && serverChatCharacterId == null)
        return;

      const branchIndex = nextMessages.length - 1;
      if (branchIndex < 0) return;

      const branchedChatId = await createServerOnlyChatBranch(branchIndex);
      if (!branchedChatId) {
        throw new Error("Failed to create branch for regeneration");
      }

      return {
        submitExtras: {
          serverChatIdOverride: branchedChatId,
        },
      };
    },
  });

  return {
    messages,
    setMessages,
    editMessage,
    deleteMessage,
    onSubmit,
    setStreaming,
    streaming,
    setHistory,
    historyId,
    setHistoryId,
    setIsFirstMessage,
    isLoading,
    setIsLoading,
    isProcessing,
    setIsProcessing,
    stopStreamingRequest,
    clearChat,
    selectedModel,
    setSelectedModel,
    chatMode,
    setChatMode,
    isEmbedding,
    setIsEmbedding,
    regenerateLastMessage,
    webSearch,
    setWebSearch,
    isSearchingInternet,
    setIsSearchingInternet,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    speechToTextLanguage,
    setSpeechToTextLanguage,
    useOCR,
    setUseOCR,
    defaultInternetSearchOn,
    defaultChatWithWebsite,
    serverChatId,
    setServerChatId,
    serverChatState,
    setServerChatState,
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
    history,
    createChatBranch,
    temporaryChat,
    setTemporaryChat,
    toolChoice,
    setToolChoice,
    sidepanelTemporaryChat,
    queuedMessages,
    addQueuedMessage,
    setQueuedMessages,
    clearQueuedMessages,
  };
};
