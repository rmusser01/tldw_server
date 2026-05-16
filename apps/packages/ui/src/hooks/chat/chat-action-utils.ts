import { isAbortLikeError } from "@/hooks/chat/abort-turn-cleanup";
import {
  isImageGenerationMessageType,
} from "@/utils/image-generation-chat";
import type { ImageGenerationEventSyncPolicy } from "@/utils/image-generation-chat";
import type { Message } from "@/store/option";
import type { ToolChoice } from "@/store/option";
import type { ImageGenerationEventSyncMode } from "@/utils/image-generation-chat";
import type { SaveMessageData } from "@/types/chat-modes";
import type { ChatModelSettings } from "@/store/model";
import { isGreetingMessageType } from "@/utils/character-greetings";
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ChatModelSettingsStore = ChatModelSettings & {
  setSystemPrompt?: (prompt: string) => void;
};

export type ChatModeOverrides = {
  historyId?: string | null;
  serverChatId?: string | null;
  selectedModel?: string | null;
  selectedSystemPrompt?: string | null;
  toolChoice?: ToolChoice | null;
  useOCR?: boolean;
  webSearch?: boolean;
  imageEventSyncPolicy?: ImageGenerationEventSyncPolicy;
  ragMediaIds?: number[] | null;
  fileRetrievalEnabled?: boolean;
  selectedKnowledge?: unknown;
} & Record<string, unknown>;

export type SaveMessagePayload = Omit<SaveMessageData, "setHistoryId"> & {
  setHistoryId?: SaveMessageData["setHistoryId"];
  conversationId?: string | number | null;
  message_source?: "copilot" | "web-ui" | "server" | "branch";
  message_type?: string;
};

export type TldwChatMeta =
  | {
      id?: string | number;
      chat_id?: string | number;
      version?: number;
      state?: string | null;
      conversation_state?: string | null;
      topic_label?: string | null;
      cluster_id?: string | null;
      source?: string | null;
      external_ref?: string | null;
      title?: string | null;
      character_id?: string | number | null;
      assistant_kind?: "character" | "persona" | null;
      assistant_id?: string | number | null;
      persona_memory_mode?: "read_only" | "read_write" | null;
    }
  | string
  | number
  | null
  | undefined;

export type ChatSubmitResult =
  | { status: "submitted" }
  | { status: "failed"; errorMessage: string }
  | { status: "skipped"; reason: string };

export const chatSubmitSubmitted = (): ChatSubmitResult => ({
  status: "submitted",
});

export const chatSubmitFailed = (errorMessage: string): ChatSubmitResult => ({
  status: "failed",
  errorMessage,
});

export const chatSubmitSkipped = (reason: string): ChatSubmitResult => ({
  status: "skipped",
  reason,
});

export const isChatSubmitSuccess = (result: ChatSubmitResult) =>
  result.status === "submitted";

export const normalizeChatSubmitResult = (
  result: ChatSubmitResult | void | undefined,
): ChatSubmitResult => {
  if (typeof result === "undefined") {
    return chatSubmitSubmitted();
  }
  return result;
};

export const getChatSubmitIssueMessage = (result: ChatSubmitResult): string => {
  if (result.status === "failed") return result.errorMessage;
  if (result.status === "skipped") return result.reason;
  return "";
};

export const throwIfChatSubmitUnsuccessful = (
  result: ChatSubmitResult | void | undefined,
) => {
  const normalized = normalizeChatSubmitResult(result);
  if (isChatSubmitSuccess(normalized)) return;
  throw new Error(getChatSubmitIssueMessage(normalized));
};

export const aggregateChatSubmitResults = (
  results: ChatSubmitResult[],
): ChatSubmitResult => {
  if (results.some(isChatSubmitSuccess)) {
    return chatSubmitSubmitted();
  }

  const failedResult = results.find(
    (result): result is Extract<ChatSubmitResult, { status: "failed" }> =>
      result.status === "failed",
  );
  if (failedResult) {
    return failedResult;
  }

  const skippedResult = results.find(
    (result): result is Extract<ChatSubmitResult, { status: "skipped" }> =>
      result.status === "skipped",
  );
  if (skippedResult) {
    return skippedResult;
  }

  return chatSubmitSkipped("No chat submissions completed");
};

// ---------------------------------------------------------------------------
// Pure utility functions
// ---------------------------------------------------------------------------

const normalizeRagMediaIds = (value: unknown): number[] | null => {
  if (!Array.isArray(value)) return null;
  return value.filter(
    (mediaId): mediaId is number =>
      typeof mediaId === "number" && Number.isFinite(mediaId),
  );
};

export const resolveTurnRagMediaIds = ({
  requestOverrides,
  ragMediaIds,
}: {
  requestOverrides?: Pick<ChatModeOverrides, "ragMediaIds"> | null;
  ragMediaIds: number[] | null;
}): number[] | null => {
  const hasExplicitOverride =
    requestOverrides != null &&
    Object.prototype.hasOwnProperty.call(requestOverrides, "ragMediaIds") &&
    requestOverrides.ragMediaIds !== undefined;

  if (hasExplicitOverride) {
    return normalizeRagMediaIds(requestOverrides?.ragMediaIds);
  }

  return normalizeRagMediaIds(ragMediaIds);
};

export const resolveTurnFileRetrievalEnabled = ({
  requestOverrides,
  fileRetrievalEnabled,
}: {
  requestOverrides?: Pick<ChatModeOverrides, "fileRetrievalEnabled"> | null;
  fileRetrievalEnabled: boolean;
}): boolean =>
  typeof requestOverrides?.fileRetrievalEnabled === "boolean"
    ? requestOverrides.fileRetrievalEnabled
    : fileRetrievalEnabled;

export const shouldUseRagForTurn = ({
  selectedKnowledge,
  fileRetrievalEnabled,
  ragMediaIds,
}: {
  selectedKnowledge: unknown;
  fileRetrievalEnabled: boolean;
  ragMediaIds: number[] | null;
}) =>
  Boolean(selectedKnowledge) ||
  (fileRetrievalEnabled && Array.isArray(ragMediaIds) && ragMediaIds.length > 0);

export const attemptCharacterStreamRecoveryPersist = async ({
  chatId,
  temporaryChat,
  assistantContent,
  alreadyPersisted,
  error,
  persist,
}: {
  chatId: string | null;
  temporaryChat: boolean;
  assistantContent: string;
  alreadyPersisted: boolean;
  error: unknown;
  persist: (content: string) => Promise<boolean>;
}): Promise<boolean> => {
  if (alreadyPersisted || temporaryChat) return false;
  if (!chatId || isAbortLikeError(error)) return false;
  const trimmedContent = assistantContent.trim();
  if (!trimmedContent) return false;
  try {
    return await persist(trimmedContent);
  } catch {
    return false;
  }
};

// ---------------------------------------------------------------------------
// Compare helpers
// ---------------------------------------------------------------------------

export const getMessageModelKey = (message: Message) =>
  message.modelId || message.modelName || message.name;

export const shouldIncludeMessageForModel = (
  message: Message,
  modelId: string,
) => {
  if (!message.isBot) {
    if (message.messageType === "compare:perModelUser") {
      return message.modelId === modelId;
    }
    return true;
  }
  const messageModel = getMessageModelKey(message);
  if (!messageModel) {
    return false;
  }
  return messageModel === modelId;
};

export const getCompareUserMessageId = (
  items: Message[],
  clusterId: string,
) =>
  items.find(
    (message) =>
      message.messageType === "compare:user" &&
      message.clusterId === clusterId,
  )?.id || null;

export const getLastThreadMessageId = (
  items: Message[],
  clusterId: string,
  modelId: string,
) => {
  const threadMessages = items.filter(
    (message) =>
      message.clusterId === clusterId &&
      getMessageModelKey(message) === modelId,
  );
  const lastThreadMessage = threadMessages[threadMessages.length - 1];
  return lastThreadMessage?.id || getCompareUserMessageId(items, clusterId);
};

export const getCompareBranchMessageIds = (
  items: Message[],
  clusterId: string,
  modelId: string,
) => {
  const userIndex = items.findIndex(
    (message) =>
      message.messageType === "compare:user" &&
      message.clusterId === clusterId,
  );
  if (userIndex === -1) {
    return [];
  }

  const messageIds = new Set<string>();
  items.forEach((message, index) => {
    if (!message.id) {
      return;
    }
    if (index < userIndex) {
      if (shouldIncludeMessageForModel(message, modelId)) {
        messageIds.add(message.id);
      }
      return;
    }
    if (message.clusterId !== clusterId) {
      return;
    }
    if (message.messageType === "compare:user") {
      messageIds.add(message.id);
      return;
    }
    if (shouldIncludeMessageForModel(message, modelId)) {
      messageIds.add(message.id);
    }
  });

  return Array.from(messageIds);
};

export const buildHistoryFromMessagesFactory = (greetingEnabled: boolean) => {
  return (items: Message[]) =>
    items
      .filter(
        (message) =>
          !isImageGenerationMessageType(message.messageType) &&
          (greetingEnabled
            ? true
            : !isGreetingMessageType(message.messageType)),
      )
      .map((message) => ({
        role: (message.isBot ? "assistant" : "user") as "assistant" | "user",
        content: message.message,
        image: message.images?.[0],
        messageType: message.messageType,
      }));
};

export const buildHistoryForModel = (
  items: Message[],
  modelId: string,
  buildHistoryFromMessages: (items: Message[]) => any[],
) =>
  buildHistoryFromMessages(
    items.filter((message) => shouldIncludeMessageForModel(message, modelId)),
  );

export const resolveCompareModelSelection = (modelKey: string) => {
  const rawModelKey = String(modelKey || "").trim();
  const modelSelection = parseProviderQualifiedModelSelection(rawModelKey);
  const selectedModel = modelSelection.modelId || rawModelKey;
  const historyModelKey =
    modelSelection.provider && selectedModel
      ? `${modelSelection.provider}:${selectedModel}`
      : selectedModel;

  return {
    selectedModel,
    historyModelKey,
    provider: modelSelection.provider,
  };
};
