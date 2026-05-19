import React from "react";
import type { TFunction } from "i18next";
import { generateID } from "@/db/dexie/helpers";
import { getModelNicknameByID } from "@/db/dexie/nickname";
import { isReasoningEnded, isReasoningStarted } from "@/libs/reasoning";
import { buildAssistantErrorContent } from "@/utils/chat-error-message";
import { detectCharacterMood } from "@/utils/character-mood";
import {
  buildMessageVariant,
  getLastUserMessageId,
  normalizeMessageVariants,
  updateActiveVariant,
} from "@/utils/message-variants";
import {
  resolveExplicitProviderForSelectedModel,
  resolveApiProviderForModel,
} from "@/utils/resolve-api-provider";
import { consumeStreamingChunk } from "@/utils/streaming-chunks";
import { normalizeConversationState } from "@/utils/conversation-state";
import {
  buildGreetingOptionsFromEntries,
  collectGreetingEntries,
  collectGreetings,
  isGreetingMessageType,
  resolveGreetingSelection,
} from "@/utils/character-greetings";
import {
  toMessageSteeringPromptPayload,
} from "@/utils/message-steering";
import type { MessageSteeringPromptTemplates } from "@/types/message-steering";
import {
  tldwClient,
  type ConversationState,
} from "@/services/tldw/TldwApiClient";
import {
  discardAbortedTurnIfRequested,
} from "@/hooks/chat/abort-turn-cleanup";
import { resolveSavedDegradedCharacterPersist } from "@/hooks/chat/characterPersistOutcome";
import type { Character } from "@/types/character";
import type { ChatScope } from "@/types/chat-scope";
import type { ChatHistory, Message } from "@/store/option";
import type { SaveMessageData } from "@/types/chat-modes";
import {
  attemptCharacterStreamRecoveryPersist,
  type TldwChatMeta,
  type SaveMessagePayload,
  type ChatModelSettingsStore,
} from "./chat-action-utils";

const STREAMING_UPDATE_INTERVAL_MS = 80;

export type CharacterChatModeParams = {
  message: string;
  image: string;
  isRegenerate: boolean;
  messages: Message[];
  history: ChatHistory;
  signal: AbortSignal;
  model: string;
  regenerateFromMessage?: Message;
  character?: Character | null;
  controller: AbortController;
  serverChatIdOverride?: string | null;
  messageSteering: {
    continueAsUser: boolean;
    impersonateUser: boolean;
    forceNarrate: boolean;
  };
};

export type CharacterChatModeDeps = {
  t: TFunction;
  notification: any;
  selectedCharacter: Character | null;
  temporaryChat: boolean;
  historyId: string | null;
  serverChatId: string | null;
  serverChatCharacterId: string | number | null;
  serverChatState: ConversationState | null;
  serverChatTopic: string | null;
  serverChatClusterId: string | null;
  serverChatSource: string | null;
  serverChatExternalRef: string | null;
  currentChatModelSettings: ChatModelSettingsStore;
  setMessages: (
    messagesOrUpdater: Message[] | ((prev: Message[]) => Message[]),
  ) => void;
  setHistory: (
    historyOrUpdater: ChatHistory | ((prev: ChatHistory) => ChatHistory),
  ) => void;
  setHistoryId: (
    historyId: string | null,
    options?: { preserveServerChatId?: boolean },
  ) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  setStreaming: (streaming: boolean) => void;
  setAbortController: (controller: AbortController | null) => void;
  setServerChatId: (id: string | null) => void;
  setServerChatTitle: (title: string | null) => void;
  setServerChatCharacterId: (id: string | number | null) => void;
  setServerChatMetaLoaded: (loaded: boolean) => void;
  setServerChatState: (state: ConversationState | null) => void;
  setServerChatVersion: (version: number | null) => void;
  setServerChatTopic: (topic: string | null) => void;
  setServerChatClusterId: (clusterId: string | null) => void;
  setServerChatSource: (source: string | null) => void;
  setServerChatExternalRef: (ref: string | null) => void;
  invalidateServerChatHistory: () => void;
  greetingEnabled: boolean;
  greetingSelectionId: string | null;
  greetingsChecksum: string | null;
  useCharacterDefault: boolean;
  directedCharacterId: number | null;
  resolvedMessageSteeringPrompts: MessageSteeringPromptTemplates;
  getEffectiveSelectedModel: (preferred?: string | null) => string | null;
  saveMessageOnSuccess: (payload?: SaveMessagePayload) => Promise<string | null>;
  saveMessageOnError: (payload: any) => Promise<any>;
  discardCurrentTurnOnAbortRef: React.MutableRefObject<boolean>;
  scope?: ChatScope;
};

export const createCharacterChatMode = (deps: CharacterChatModeDeps) => {
  return async (params: CharacterChatModeParams) => {
    const {
      message,
      image,
      isRegenerate,
      messages: chatHistory,
      history: chatMemory,
      signal,
      model,
      regenerateFromMessage,
      character,
      controller,
      serverChatIdOverride,
      messageSteering,
    } = params;

    const {
      t,
      notification,
      selectedCharacter,
      temporaryChat,
      historyId,
      serverChatId,
      serverChatCharacterId,
      serverChatState,
      serverChatTopic,
      serverChatClusterId,
      serverChatSource,
      serverChatExternalRef,
      currentChatModelSettings,
      setMessages,
      setHistory,
      setIsProcessing,
      setStreaming,
      setAbortController,
      setServerChatId,
      setServerChatTitle,
      setServerChatCharacterId,
      setServerChatMetaLoaded,
      setServerChatState,
      setServerChatVersion,
      setServerChatTopic,
      setServerChatClusterId,
      setServerChatSource,
      setServerChatExternalRef,
      invalidateServerChatHistory,
      greetingEnabled,
      greetingSelectionId,
      greetingsChecksum,
      useCharacterDefault,
      directedCharacterId,
      resolvedMessageSteeringPrompts,
      getEffectiveSelectedModel,
      saveMessageOnSuccess,
      saveMessageOnError,
      discardCurrentTurnOnAbortRef,
      scope,
    } = deps;

    const activeCharacter = character ?? selectedCharacter;
    if (!activeCharacter?.id) {
      throw new Error("No character selected");
    }

    const resolveGreetingText = (): string => {
      if (!greetingEnabled) return "";

      const hasUserTurns =
        chatHistory.some((entry) => !entry.isBot) ||
        chatMemory.some((entry) => entry.role === "user");
      const greetingEntries = collectGreetingEntries(activeCharacter as any);
      const greetingOptions = buildGreetingOptionsFromEntries(greetingEntries);
      const selectedGreeting =
        resolveGreetingSelection({
          options: greetingOptions,
          greetingSelectionId,
          greetingsChecksum,
          useCharacterDefault,
          fallback: "first",
        }).option?.text?.trim() ?? "";
      if (!hasUserTurns && selectedGreeting.length > 0) {
        return selectedGreeting;
      }

      const fromMessages = chatHistory.find(
        (entry) =>
          entry.isBot &&
          isGreetingMessageType(entry.messageType) &&
          typeof entry.message === "string" &&
          entry.message.trim().length > 0,
      );
      if (fromMessages?.message) {
        return fromMessages.message.trim();
      }

      const fromHistory = chatMemory.find(
        (entry) =>
          entry.role === "assistant" &&
          isGreetingMessageType(entry.messageType) &&
          typeof entry.content === "string" &&
          entry.content.trim().length > 0,
      );
      if (fromHistory?.content) {
        return fromHistory.content.trim();
      }

      if (selectedGreeting.length > 0) {
        return selectedGreeting;
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
      chatMemory.some(
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
            ...chatMemory,
          ]
        : chatMemory;

    let fullText = "";
    let contentToSave = "";
    const resolvedAssistantMessageId = generateID();
    const resolvedUserMessageId = !isRegenerate ? generateID() : undefined;
    let persistedUserServerMessageId: string | undefined;
    let activeChatId: string | null = null;
    let assistantPersistedToServer = false;
    let generateMessageId = resolvedAssistantMessageId;
    const fallbackParentMessageId = getLastUserMessageId(chatHistory);
    const resolvedAssistantParentMessageId = isRegenerate
      ? (regenerateFromMessage?.parentMessageId ?? fallbackParentMessageId)
      : (resolvedUserMessageId ?? null);
    const regenerateVariants =
      isRegenerate && regenerateFromMessage
        ? normalizeMessageVariants(regenerateFromMessage)
        : [];
    const resolvedModel = model?.trim();
    let streamingTimer: ReturnType<typeof setTimeout> | null = null;
    let pendingStreamingText = "";
    let pendingReasoningTime = 0;
    let lastStreamingUpdateAt = 0;
    let inactivityTimer: ReturnType<typeof setTimeout> | null = null;

    const flushStreamingUpdate = () => {
      if (pendingStreamingText.length === 0) return;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === generateMessageId
            ? updateActiveVariant(m, {
                message: pendingStreamingText,
                reasoning_time_taken: pendingReasoningTime,
              })
            : m,
        ),
      );
      pendingStreamingText = "";
      pendingReasoningTime = 0;
      lastStreamingUpdateAt = Date.now();
    };

    const scheduleStreamingUpdate = (text: string, reasoningTime: number) => {
      pendingStreamingText = text;
      pendingReasoningTime = reasoningTime;
      if (streamingTimer !== null) return;
      const elapsed = Date.now() - lastStreamingUpdateAt;
      const delay = Math.max(0, STREAMING_UPDATE_INTERVAL_MS - elapsed);
      streamingTimer = setTimeout(() => {
        streamingTimer = null;
        flushStreamingUpdate();
      }, delay);
    };

    const cancelStreamingUpdate = () => {
      if (streamingTimer === null) return;
      clearTimeout(streamingTimer);
      streamingTimer = null;
    };

    const abortCancelStreamingUpdate = () => cancelStreamingUpdate();
    signal.addEventListener("abort", abortCancelStreamingUpdate);

    try {
      if (!resolvedModel) {
        notification.error({
          message: t("error"),
          description: t("validationSelectModel"),
        });
        setIsProcessing(false);
        setStreaming(false);
        return;
      }

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

      await tldwClient.initialize().catch(() => null);

      const modelInfo = await getModelNicknameByID(resolvedModel);
      const characterName =
        activeCharacter?.name || modelInfo?.model_name || resolvedModel;
      const characterAvatar =
        activeCharacter?.avatar_url || modelInfo?.model_avatar;
      const createdAt = Date.now();
      const hasGreetingInMessages = chatHistory.some((entry) => {
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
        ? [greetingSeedMessage, ...chatHistory]
        : chatHistory;
      const assistantStub: Message = {
        isBot: true,
        role: "assistant",
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
              role: "user",
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

      const activeCharacterId = String(activeCharacter.id);
      const serverCharacterId =
        serverChatCharacterId != null ? String(serverChatCharacterId) : null;
      const overrideChatId =
        typeof serverChatIdOverride === "string" &&
        serverChatIdOverride.trim().length > 0
          ? serverChatIdOverride.trim()
          : null;
      const resolvedServerChatId = overrideChatId || serverChatId;
      const shouldResetServerChat =
        Boolean(resolvedServerChatId) &&
        (!serverCharacterId || serverCharacterId !== activeCharacterId);

      if (shouldResetServerChat) {
        setServerChatId(null);
        setServerChatCharacterId(null);
        setServerChatMetaLoaded(false);
        setServerChatTitle(null);
        setServerChatState("in-progress");
        setServerChatVersion(null);
        setServerChatTopic(null);
        setServerChatClusterId(null);
        setServerChatSource(null);
        setServerChatExternalRef(null);
      }

      let chatId = shouldResetServerChat ? null : resolvedServerChatId;
      let createdNewChat = false;
      if (!chatId) {
        const created = (await tldwClient.createChat(
          {
            character_id: activeCharacter.id,
            state: serverChatState || "in-progress",
            topic_label: serverChatTopic || undefined,
            cluster_id: serverChatClusterId || undefined,
            source: serverChatSource || undefined,
            external_ref: serverChatExternalRef || undefined,
          },
          scope ? { scope } : undefined,
        )) as TldwChatMeta;

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
          } = created;
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
        const createdTitle =
          created && typeof created === "object"
            ? String(created.title ?? "")
            : "";
        const createdCharacterId =
          created && typeof created === "object"
            ? (created.character_id ?? activeCharacter?.id ?? null)
            : (activeCharacter?.id ?? null);
        setServerChatTitle(createdTitle);
        setServerChatCharacterId(createdCharacterId);
        setServerChatMetaLoaded(true);
        invalidateServerChatHistory();
      }
      activeChatId = chatId;

      if (createdNewChat && !isRegenerate && greetingText.length > 0) {
        try {
          const createdGreeting = (await tldwClient.addChatMessage(
            chatId,
            {
              role: "assistant",
              content: greetingText,
            },
            scope ? { scope } : undefined,
          )) as { id?: string | number; version?: number } | null;
          if (createdGreeting?.id != null) {
            setMessages((prev) => {
              const updated = [...prev] as (Message & {
                serverMessageId?: string;
                serverMessageVersion?: number;
              })[];
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
          if (b64 !== undefined && b64.length > 0) {
            payload.image_base64 = b64;
          }
        }
        if (payload.content || payload.image_base64) {
          const createdUser = (await tldwClient.addChatMessage(
            chatId,
            payload,
            scope ? { scope } : undefined,
          )) as TldwChatMessage | null;
          persistedUserServerMessageId =
            createdUser?.id != null ? String(createdUser.id) : undefined;
          setMessages((prev) => {
            const updated = [...prev] as (Message & {
              serverMessageId?: string;
              serverMessageVersion?: number;
            })[];
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

      let count = 0;
      let reasoningStartTime: Date | null = null;
      let reasoningEndTime: Date | null = null;
      let timetaken = 0;
      let apiReasoning = false;
      let streamTransportInterrupted = false;
      let streamTransportInterruptionReason: string | null = null;

      const explicitProvider = resolveExplicitProviderForSelectedModel({
        currentSelectedModel: getEffectiveSelectedModel(),
        requestedSelectedModel: resolvedModel,
        explicitProvider: currentChatModelSettings.apiProvider,
      });
      const resolvedApiProvider = await resolveApiProviderForModel({
        modelId: resolvedModel,
        explicitProvider,
      });
      const normalizedModel = resolvedModel.replace(/^tldw:/, "").trim();
      const streamModel =
        normalizedModel.length > 0 ? normalizedModel : resolvedModel;

      const shouldPersistToServer = !temporaryChat;
      const STREAM_INACTIVITY_TIMEOUT_MS = 60_000;
      let inactivityAborted = false;
      const resetInactivityTimer = () => {
        if (inactivityTimer) clearTimeout(inactivityTimer);
        inactivityTimer = setTimeout(() => {
          inactivityAborted = true;
          controller.abort();
        }, STREAM_INACTIVITY_TIMEOUT_MS);
      };
      resetInactivityTimer();
      for await (const chunk of tldwClient.streamCharacterChatCompletion(
        chatId,
        {
          include_character_context: true,
          model: streamModel,
          provider: resolvedApiProvider,
          save_to_db: shouldPersistToServer,
          directed_character_id: directedCharacterId ?? undefined,
          continue_as_user: messageSteering.continueAsUser,
          impersonate_user: messageSteering.impersonateUser,
          force_narrate: messageSteering.forceNarrate,
          message_steering_prompts: toMessageSteeringPromptPayload(
            resolvedMessageSteeringPrompts,
          ),
        },
        scope ? { signal, scope } : { signal },
      )) {
        resetInactivityTimer();
        const interruptionEvent =
          chunk && typeof chunk === "object" && !Array.isArray(chunk)
            ? (chunk as Record<string, unknown>)
            : null;
        if (
          typeof interruptionEvent?.event === "string" &&
          interruptionEvent.event.toLowerCase() ===
            "stream_transport_interrupted"
        ) {
          streamTransportInterrupted = true;
          const detail =
            typeof interruptionEvent.detail === "string"
              ? interruptionEvent.detail.trim()
              : "";
          streamTransportInterruptionReason =
            detail.length > 0 ? detail : streamTransportInterruptionReason;
          continue;
        }
        const chunkState = consumeStreamingChunk(
          { fullText, contentToSave, apiReasoning },
          chunk,
        );
        fullText = chunkState.fullText;
        contentToSave = chunkState.contentToSave;
        apiReasoning = chunkState.apiReasoning;

        if (chunkState.token) {
          scheduleStreamingUpdate(`${fullText}▋`, timetaken);
        }
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

        count++;
        if (signal?.aborted) break;
      }
      cancelStreamingUpdate();
      flushStreamingUpdate();
      if (inactivityTimer) clearTimeout(inactivityTimer);

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
            ? (() => {
                const nextVariantPayload: Record<string, unknown> = {
                  message: fullText,
                  reasoning_time_taken: timetaken,
                };
                if (streamTransportInterrupted) {
                  const existingGenerationInfo =
                    m.generationInfo &&
                    typeof m.generationInfo === "object" &&
                    !Array.isArray(m.generationInfo)
                      ? (m.generationInfo as Record<string, unknown>)
                      : {};
                  nextVariantPayload.generationInfo = {
                    ...existingGenerationInfo,
                    streamTransportInterrupted: true,
                    partialResponseSaved: true,
                    streamTransportInterruptionReason:
                      streamTransportInterruptionReason ||
                      "Stream transport interrupted; partial response saved.",
                  };
                }
                return updateActiveVariant(m, nextVariantPayload);
              })()
            : m,
        ),
      );

      const finalContent = contentToSave || fullText;
      const finalPersistedContent = finalContent.trim();

      if (finalPersistedContent.length > 0) {
        let fallbackSpeakerId: number | undefined;
        let speakerCharacterId: number | undefined;
        let detectedMood: ReturnType<typeof detectCharacterMood> | undefined;
        let resolvedMoodLabel: string | undefined;
        let resolvedMoodConfidence: number | undefined;
        let resolvedMoodTopic: string | undefined;
        let metadataExtra: Record<string, unknown> | undefined;
        try {
          fallbackSpeakerId = Number.parseInt(
            String(activeCharacter.id),
            10,
          );
          speakerCharacterId =
            Number.isFinite(directedCharacterId ?? NaN) &&
            (directedCharacterId ?? 0) > 0
              ? directedCharacterId
              : Number.isFinite(fallbackSpeakerId) && fallbackSpeakerId > 0
                ? fallbackSpeakerId
                : undefined;
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
            typeof detectedMood.topic === "string" && detectedMood.topic.trim()
              ? detectedMood.topic.trim()
              : undefined;

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
          if (persistedUserServerMessageId) {
            persistPayload.user_message_id = persistedUserServerMessageId;
          }
          metadataExtra = {
            speaker_character_id: speakerCharacterId ?? null,
            speaker_character_name: characterName,
            mood_label: resolvedMoodLabel,
            mood_confidence: resolvedMoodConfidence ?? null,
            mood_topic: resolvedMoodTopic ?? null,
          };

          const persisted = (await tldwClient.persistCharacterCompletion(
            chatId,
            persistPayload,
            scope ? { scope } : undefined,
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
          assistantPersistedToServer = createdAsstServerId != null;
          setMessages(
            (prev) =>
              (prev as any[]).map((m) => {
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
            assistantPersistedToServer = true;
            setMessages(
              (prev) =>
                (prev as any[]).map((m) => {
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
              const createdAsst = (await tldwClient.addChatMessage(
                chatId,
                {
                  role: "assistant",
                  content: finalPersistedContent,
                },
                scope ? { scope } : undefined,
              )) as { id?: string | number; version?: number } | null;
              assistantPersistedToServer = createdAsst?.id != null;
              setMessages(
                (prev) =>
                  (prev as any[]).map((m) => {
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
        isRegenerate,
        selectedModel: resolvedModel,
        modelId: resolvedModel,
        message,
        image,
        fullText: finalContent,
        source: [],
        message_source: "web-ui",
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
          previousMessages: chatHistory,
          previousHistory: chatMemory,
          setMessages,
          setHistory,
        })
      ) {
        setIsProcessing(false);
        setStreaming(false);
        return;
      }

      cancelStreamingUpdate();
      await attemptCharacterStreamRecoveryPersist({
        chatId: activeChatId,
        temporaryChat,
        assistantContent: contentToSave || fullText,
        alreadyPersisted: assistantPersistedToServer,
        error: e,
        persist: async (assistantContent) => {
          if (!activeChatId) return false;
          const createdAsst = (await tldwClient.addChatMessage(
            activeChatId,
            {
              role: "assistant",
              content: assistantContent,
            },
            scope ? { scope } : undefined,
          )) as { id?: string | number; version?: number } | null;
          if (createdAsst?.id == null) return false;
          assistantPersistedToServer = true;
          setMessages(
            (prev) =>
              (prev as any[]).map((m) => {
                if (m.id !== generateMessageId) return m;
                return updateActiveVariant(m, {
                  serverMessageId: String(createdAsst.id),
                  serverMessageVersion: createdAsst?.version,
                });
              }) as Message[],
          );
          return true;
        },
      });
      const assistantContent = buildAssistantErrorContent(fullText, e);
      const interruptionReason =
        e instanceof Error ? e.message : t("somethingWentWrong");
      if (generateMessageId) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === generateMessageId
              ? updateActiveVariant(msg, {
                  message: assistantContent,
                  generationInfo: {
                    ...(msg.generationInfo || {}),
                    interrupted: true,
                    interruptionReason,
                    interruptedAt: Date.now(),
                  },
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
        selectedModel: resolvedModel,
        modelId: resolvedModel,
        userMessage: message,
        isRegenerating: isRegenerate,
        message_source: "web-ui",
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
          description: e instanceof Error ? e.message : t("somethingWentWrong"),
        });
      }
      setIsProcessing(false);
      setStreaming(false);
    } finally {
      discardCurrentTurnOnAbortRef.current = false;
      cancelStreamingUpdate();
      signal.removeEventListener("abort", abortCancelStreamingUpdate);
      if (inactivityTimer) clearTimeout(inactivityTimer);
      setAbortController(null);
    }
  };
};
