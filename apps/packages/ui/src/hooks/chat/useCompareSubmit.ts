import React from "react";
import type { TFunction } from "i18next";
import {
  generateID,
  saveHistory,
  saveMessage,
  formatToChatHistory,
  formatToMessage,
  getSessionFiles,
  getPromptById,
  updateHistory,
} from "@/db/dexie/helpers";
import { generateBranchFromMessageIds } from "@/db/dexie/branch";
import type { UploadedFile } from "@/db/dexie/types";
import { normalChatMode } from "@/hooks/chat-modes/normalChatMode";
import { ragMode } from "@/hooks/chat-modes/ragMode";
import { tabChatMode } from "@/hooks/chat-modes/tabChatMode";
import { documentChatMode } from "@/hooks/chat-modes/documentChatMode";
import { trackCompareMetric } from "@/utils/compare-metrics";
import { updatePageTitle } from "@/utils/update-page-title";
import type { ChatDocuments } from "@/models/ChatTypes";
import type { ChatHistory, Message, Knowledge } from "@/store/option";
import {
  getLastThreadMessageId,
  getCompareBranchMessageIds,
  buildHistoryForModel,
  resolveCompareModelSelection,
  type SaveMessagePayload,
  type ChatModelSettingsStore,
} from "./chat-action-utils";
import {
  hasActiveMessageSteering,
} from "@/utils/message-steering";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { generateTitle } from "@/services/title";

export type UseCompareSubmitOptions = {
  t: TFunction;
  notification: any;
  messages: Message[];
  history: ChatHistory;
  historyId: string | null;
  temporaryChat: boolean;
  selectedKnowledge: Knowledge | null;
  fileRetrievalEnabled: boolean;
  ragMediaIds: number[] | null;
  uploadedFiles: UploadedFile[];
  contextFiles: UploadedFile[];
  documentContext: ChatDocuments | null;
  compareFeatureEnabled: boolean;
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
  setStreaming: (streaming: boolean) => void;
  setIsProcessing: (isProcessing: boolean) => void;
  setAbortController: (controller: AbortController | null) => void;
  setContextFiles: (files: UploadedFile[]) => void;
  setSelectedSystemPrompt: (prompt: string) => void;
  markCompareHistoryCreated: (historyId: string) => void;
  clearMessageSteering: () => void;
  invalidateServerChatHistory: () => void;
  serverChatId: string | null;
  setServerChatId: (id: string | null) => void;
  setServerChatTitle: (title: string | null) => void;
  buildChatModeParams: (overrides?: any) => Promise<any>;
  buildHistoryFromMessages: (items: Message[]) => ChatHistory;
  getEffectiveSelectedModel: (preferred?: string | null) => string | null;
  resolvedMessageSteering: any;
};

export const useCompareSubmit = (opts: UseCompareSubmitOptions) => {
  const {
    t,
    notification,
    messages,
    history,
    historyId,
    temporaryChat,
    selectedKnowledge,
    fileRetrievalEnabled,
    ragMediaIds,
    uploadedFiles,
    contextFiles,
    documentContext,
    compareFeatureEnabled,
    currentChatModelSettings,
    setMessages,
    setHistory,
    setHistoryId,
    setStreaming,
    setIsProcessing,
    setAbortController,
    setContextFiles,
    setSelectedSystemPrompt,
    markCompareHistoryCreated,
    clearMessageSteering,
    serverChatId,
    buildChatModeParams,
    buildHistoryFromMessages,
    getEffectiveSelectedModel,
    resolvedMessageSteering,
  } = opts;

  const applySelectionProviderToSettings = (
    settings: ChatModelSettingsStore,
    provider?: string,
  ): ChatModelSettingsStore =>
    provider ? { ...settings, apiProvider: provider } : settings;

  const buildCompareHistoryTitle = React.useCallback(
    (title: string) => {
      const trimmed =
        title?.trim() || t("common:untitled", { defaultValue: "Untitled" });
      return t(
        "playground:composer.compareHistoryPrefix",
        "Compare: {{title}}",
        { title: trimmed },
      );
    },
    [t],
  );

  const buildCompareSplitTitle = React.useCallback(
    (title: string) => {
      const trimmed =
        title?.trim() || t("common:untitled", { defaultValue: "Untitled" });
      const suffix = t(
        "playground:composer.compareHistorySuffix",
        "(from compare)",
      );
      if (trimmed.includes(suffix)) {
        return trimmed;
      }
      return `${trimmed} ${suffix}`.trim();
    },
    [t],
  );

  const sendPerModelReply = async ({
    clusterId,
    modelId,
    message,
  }: {
    clusterId: string;
    modelId: string;
    message: string;
  }) => {
    const trimmed = message.trim();
    if (!trimmed) {
      return;
    }

    if (!compareFeatureEnabled) {
      notification.error({
        message: t("error"),
        description: t(
          "playground:composer.compareDisabled",
          "Compare mode is disabled in settings.",
        ),
      });
      return;
    }

    const messageSteeringForTurn = resolvedMessageSteering;
    const shouldConsumeSteering = hasActiveMessageSteering(
      messageSteeringForTurn,
    );

    setStreaming(true);
    const newController = new AbortController();
    setAbortController(newController);
    const signal = newController.signal;

    const baseMessages = messages;
    const baseHistory = history;
    const userMessageId = generateID();
    const assistantMessageId = generateID();
    const modelSelection = resolveCompareModelSelection(modelId);
    const userParentMessageId = getLastThreadMessageId(
      baseMessages,
      clusterId,
      modelSelection.historyModelKey,
    );

    try {
      const chatModeParams = await buildChatModeParams({
        messageSteering: messageSteeringForTurn,
      });
      const enhancedChatModeParams = {
        ...chatModeParams,
        uploadedFiles: uploadedFiles,
      };
      const historyForModel = buildHistoryForModel(
        baseMessages,
        modelSelection.historyModelKey,
        buildHistoryFromMessages,
      );
      const perModelOverrides = {
        selectedModel: modelSelection.selectedModel,
        currentChatModelSettings: applySelectionProviderToSettings(
          enhancedChatModeParams.currentChatModelSettings,
          modelSelection.provider,
        ),
        clusterId,
        userMessageType: "compare:perModelUser",
        assistantMessageType: "compare:reply",
        modelIdOverride: modelSelection.historyModelKey,
        userMessageId,
        assistantMessageId,
        userParentMessageId,
        assistantParentMessageId: userMessageId,
        historyForModel,
      };

      if (contextFiles.length > 0) {
        await documentChatMode(
          trimmed,
          "",
          false,
          baseMessages,
          baseHistory,
          signal,
          contextFiles,
          {
            ...chatModeParams,
            ...perModelOverrides,
          },
        );
        return;
      }

      if (documentContext && documentContext.length > 0) {
        await tabChatMode(
          trimmed,
          "",
          documentContext,
          false,
          baseMessages,
          baseHistory,
          signal,
          {
            ...chatModeParams,
            ...perModelOverrides,
          },
        );
        return;
      }

      const hasScopedRagMediaIds =
        Array.isArray(ragMediaIds) && ragMediaIds.length > 0;
      const shouldUseRag =
        Boolean(selectedKnowledge) ||
        (fileRetrievalEnabled && hasScopedRagMediaIds);
      if (shouldUseRag) {
        await ragMode(trimmed, "", false, baseMessages, baseHistory, signal, {
          ...chatModeParams,
          ...perModelOverrides,
        });
        return;
      }

      await normalChatMode(
        trimmed,
        "",
        false,
        baseMessages,
        baseHistory,
        signal,
        {
          ...enhancedChatModeParams,
          ...perModelOverrides,
        },
      );
    } catch (e) {
      const errorMessage =
        e instanceof Error ? e.message : t("somethingWentWrong");
      notification.error({
        message: t("error"),
        description: errorMessage,
      });
    } finally {
      setStreaming(false);
      setIsProcessing(false);
      setAbortController(null);
      if (shouldConsumeSteering) {
        clearMessageSteering();
      }
    }
  };

  const createCompareBranch = async ({
    clusterId,
    modelId,
    open = true,
  }: {
    clusterId: string;
    modelId: string;
    open?: boolean;
  }): Promise<string | null> => {
    if (!historyId || historyId === "temp") {
      return null;
    }

    const modelSelection = resolveCompareModelSelection(modelId);
    const messageIds = getCompareBranchMessageIds(
      messages,
      clusterId,
      modelSelection.historyModelKey,
    );
    if (messageIds.length === 0) {
      return null;
    }

    try {
      const newBranch = await generateBranchFromMessageIds(
        historyId,
        messageIds,
      );
      if (!newBranch) {
        return null;
      }

      const splitTitle = buildCompareSplitTitle(newBranch.history.title || "");
      await updateHistory(newBranch.history.id, splitTitle);

      void trackCompareMetric({ type: "split_single" });

      if (open) {
        setHistory(formatToChatHistory(newBranch.messages));
        setMessages(formatToMessage(newBranch.messages));
        setHistoryId(newBranch.history.id);
        const systemFiles = await getSessionFiles(newBranch.history.id);
        setContextFiles(systemFiles);

        const lastUsedPrompt = newBranch?.history?.last_used_prompt;
        if (lastUsedPrompt) {
          if (lastUsedPrompt.prompt_id) {
            const prompt = await getPromptById(lastUsedPrompt.prompt_id);
            if (prompt) {
              setSelectedSystemPrompt(lastUsedPrompt.prompt_id);
            }
          }
          if (currentChatModelSettings?.setSystemPrompt) {
            currentChatModelSettings.setSystemPrompt(
              lastUsedPrompt.prompt_content,
            );
          }
        }
      }

      return newBranch.history.id;
    } catch (e) {
      console.log("[compare-branch] failed", e);
      return null;
    }
  };

  return {
    sendPerModelReply,
    createCompareBranch,
    buildCompareHistoryTitle,
    buildCompareSplitTitle,
  };
};
