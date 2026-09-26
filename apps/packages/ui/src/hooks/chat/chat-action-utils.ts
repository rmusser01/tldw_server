import { isAbortLikeError } from "@/hooks/chat/abort-turn-cleanup";
import {
  isImageGenerationMessageType,
} from "@/utils/image-generation-chat";
import type { ImageGenerationEventSyncPolicy } from "@/utils/image-generation-chat";
import type { Knowledge, Message, MessageMetadataExtra, ToolChoice } from "@/store/option";
import type { UploadedFile } from "@/db/dexie/types";
import type { ImageGenerationEventSyncMode } from "@/utils/image-generation-chat";
import type { SaveMessageData } from "@/types/chat-modes";
import type { ChatModelSettings } from "@/store/model";
import type { ChatResearchContext } from "@/services/tldw/TldwApiClient";
import type { DynamicUIRequest } from "@/types/dynamic-ui";
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
  researchContext?: ChatResearchContext;
  dynamicUIRequest?: DynamicUIRequest;
  userMetadataExtra?: MessageMetadataExtra;
  ragMediaIds?: number[] | null;
  fileRetrievalEnabled?: boolean;
  contextFiles?: UploadedFile[];
  uploadedFiles?: UploadedFile[];
  selectedKnowledge?: Knowledge | null;
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

export const resolveTurnContextFiles = ({
  requestOverrides,
  contextFiles,
}: {
  requestOverrides?: Pick<ChatModeOverrides, "contextFiles"> | null;
  contextFiles: UploadedFile[];
}): UploadedFile[] =>
  Array.isArray(requestOverrides?.contextFiles)
    ? requestOverrides.contextFiles
    : contextFiles;

export const resolveTurnUploadedFiles = ({
  requestOverrides,
  uploadedFiles,
}: {
  requestOverrides?: Pick<ChatModeOverrides, "uploadedFiles"> | null;
  uploadedFiles: UploadedFile[];
}): UploadedFile[] =>
  Array.isArray(requestOverrides?.uploadedFiles)
    ? requestOverrides.uploadedFiles
    : uploadedFiles;

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

  const messageIds: string[] = [];
  items.forEach((message, index) => {
    if (!message.id) {
      return;
    }
    if (index < userIndex) {
      if (shouldIncludeMessageForModel(message, modelId)) {
        messageIds.push(message.id);
      }
      return;
    }
    if (message.clusterId !== clusterId) {
      return;
    }
    if (message.messageType === "compare:user") {
      messageIds.push(message.id);
      return;
    }
    if (shouldIncludeMessageForModel(message, modelId)) {
      messageIds.push(message.id);
    }
  });

  return messageIds;
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

/** Capture a clicked stable boundary before asynchronous preparation; UI rows never certify ownership. */
export const createSelectedForkAction =
  (
    controller:
      | import("./useHistorySelection").HistorySelectionController
      | null,
    handler: (
      request: import("@/types/history-selection").ForkRequestV1,
    ) => Promise<import("@/types/history-selection").ForkResultV1>,
    historyId: string | null,
    notification?: {
      error: (options: { message: string; description: string }) => void;
    },
  ) =>
  async (
    messageId: string,
    comparison?: { model_id: string; cluster_id: string | null },
  ): Promise<import("@/types/history-selection").ForkResultV1> => {
    const origin = controller?.getCurrent();
    const current = controller?.fence() ?? (() => false);
    const report = (
      result: import("@/types/history-selection").ForkResultV1,
    ) => {
      if (
        current() &&
        result.state !== "committed" &&
        result.state !== "legacy_completed"
      )
        notification?.error({
          message: "Branch unavailable",
          description: result.code,
        });
      return result;
    };
    const binding = {
      operation_id: crypto.randomUUID(),
      owner_key: origin?.view?.owner_key ?? "unavailable",
    };
    if (
      !origin?.owner ||
      !origin.view ||
      !messageId ||
      (origin.owner.kind === "local" && (!historyId || historyId === "temp"))
    )
      return report({
        ...binding,
        state: "blocked",
        code: "history_selection_unavailable",
      });
    if (origin.owner.kind === "unavailable")
      return report({...binding, state: "blocked", code: origin.owner.code});
    if (origin.view.conversation_id !== origin.owner.conversation_id || (origin.owner.kind === "local" && origin.view.conversation_id !== historyId))
      return report({
        ...binding,
        state: "rejected",
        code: "owner_conversation_mismatch",
      });
    const view = structuredClone(origin.view);
    try {
      const { captureLocalForkSelection, forkRequestDigest } =
        await import("@/db/dexie/branch");
      if (origin.owner.kind === "native" && comparison)
        return report({...binding, state: "blocked", code: "native_comparison_fork_unsupported"});
      const input = origin.owner.kind === "native"
        ? await (await import("@/services/chat-history-selection")).captureNativeForkSelection(origin.owner,
          {...view, cursor: {kind: "after_message", message_id: messageId}}, {validate_lease: current})
        : await captureLocalForkSelection(
        origin.owner,
        comparison
          ? {
              ...comparison,
              cursor: { kind: "after_message", message_id: messageId },
            }
          : {
              ...view,
              cursor: { kind: "after_message", message_id: messageId },
            },
        { validate_lease: current },
      );
      if (!current())
        return report({
          ...binding,
          state: "rejected",
          code: "stale_selection",
        });
      const request = {
        ...binding,
        destination_owner_key: binding.owner_key,
        input,
        request_digest: "",
      };
      return report(
        await handler({
          ...request,
          request_digest: forkRequestDigest(request),
        }),
      );
    } catch (error) {
      return report({
        ...binding,
        state: "rejected",
        code: error instanceof Error ? error.message : "fork_capture_failed",
      });
    }
  };

/** UI target selection only; the owner revalidates every member and order. */
export const getCompareBranchBoundaryId = (
  items: Message[],
  clusterId: string,
  modelId: string,
): string => {
  const common = items.filter(
    (row) => row.clusterId === clusterId && row.messageType === "compare:user",
  );
  if (common.length !== 1 || !common[0].id) return "";
  const thread = items.filter(
    (row) =>
      row.clusterId === clusterId &&
      row.messageType !== "compare:user" &&
      getMessageModelKey(row) === modelId,
  );
  if (!thread.length) return common[0].id;
  const parents = new Set(
    thread.map((row) => row.parentMessageId).filter(Boolean),
  );
  const leaves = thread.filter((row) => row.id && !parents.has(row.id));
  return leaves.length === 1 ? leaves[0].id! : "";
};

/** Capture persistence authority synchronously with the stable UI target. */
export const captureLocalMutationOwner = (
  controller:
    | import("./useHistorySelection").HistorySelectionController
    | null
    | undefined,
  historyId: string | null,
  serverChatId?: string | null
): import("@/db/dexie/history-selection").LocalHistoryOwnerV1 => {
  const origin = controller?.getCurrent()
  if (serverChatId || origin?.owner?.kind === "native")
    throw new Error("native_history_mutation_unavailable")
  if (!historyId || historyId === "temp")
    throw new Error("temporary_history_unavailable")
  if (
    !origin ||
    origin.status !== "ready" ||
    origin.owner?.kind !== "local" ||
    !origin.view
  )
    throw new Error("history_selection_unavailable")
  if (
    origin.owner.conversation_id !== historyId ||
    origin.view.conversation_id !== historyId ||
    origin.view.owner_key !== origin.owner.owner_key
  )
    throw new Error("owner_conversation_mismatch")
  return { ...origin.owner }
}
