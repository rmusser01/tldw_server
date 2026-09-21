import React from "react";
import { removeMessageByIndex } from "@/db/dexie/helpers";
import {
  createRegenerateLastMessage,
  createEditMessage,
  createBranchMessage,
} from "@/hooks/handlers/messageHandlers";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import type { ConversationState } from "@/services/tldw/TldwApiClient";
import type { UploadedFile } from "@/db/dexie/types";
import type { ChatHistory, Message, ReplyTarget } from "@/store/option";
import type { ChatScope } from "@/types/chat-scope";
import type { ChatModelSettingsStore } from "./chat-action-utils";

export type UseMessageOperationsOptions = {
  messages: Message[];
  history: ChatHistory;
  historyId: string | null;
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
  abortController: AbortController | null;
  setAbortController: (controller: AbortController | null) => void;
  serverChatId: string | null;
  serverChatTitle: string | null;
  serverChatCharacterId: string | number | null;
  serverChatState: ConversationState | null;
  serverChatTopic: string | null;
  serverChatClusterId: string | null;
  serverChatSource: string | null;
  serverChatExternalRef: string | null;
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
  replyTarget: ReplyTarget | null;
  clearReplyTarget: () => void;
  setSelectedSystemPrompt: (prompt: string) => void;
  currentChatModelSettings: ChatModelSettingsStore;
  setContextFiles: (files: UploadedFile[]) => void;
  notification: any;
  discardCurrentTurnOnAbortRef: React.MutableRefObject<boolean>;
  selectedCharacter: { id?: string | number } | null;
  scope?: ChatScope;
};

export const useMessageOperations = (opts: UseMessageOperationsOptions) => {
  const {
    messages,
    history,
    historyId,
    setMessages,
    setHistory,
    setHistoryId,
    abortController,
    setAbortController,
    serverChatId,
    serverChatTitle,
    serverChatCharacterId,
    serverChatState,
    serverChatTopic,
    serverChatClusterId,
    serverChatSource,
    serverChatExternalRef,
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
    replyTarget,
    clearReplyTarget,
    setSelectedSystemPrompt,
    currentChatModelSettings,
    setContextFiles,
    notification,
    discardCurrentTurnOnAbortRef,
    selectedCharacter,
    scope,
  } = opts;

  const stopStreamingRequest = React.useCallback(
    (options?: unknown) => {
      if (!abortController) {
        return;
      }

      const discardTurn =
        typeof options === "object" &&
        options !== null &&
        "discardTurn" in options &&
        options.discardTurn === true;

      if (discardTurn) {
        discardCurrentTurnOnAbortRef.current = true;
      }

      abortController.abort();
      setAbortController(null);
    },
    [abortController, setAbortController, discardCurrentTurnOnAbortRef],
  );

  const deleteMessage = React.useCallback(
    async (index: number) => {
      const target = messages[index];
      if (!target) return;

      const targetId = target.serverMessageId ?? target.id;
      const serverMessageId = target.serverMessageId;
      const serverMessageVersion = target.serverMessageVersion;
      const historyRole = target.role ?? (target.isBot ? "assistant" : "user");
      const historyContent = target.message ?? "";

      if (replyTarget?.id && targetId && replyTarget.id === targetId) {
        clearReplyTarget();
      }

      try {
        if (serverMessageId) {
          await tldwClient.initialize().catch(() => null);
          let expectedVersion = serverMessageVersion;
          if (expectedVersion == null) {
            const serverMessage = await tldwClient.getMessage(serverMessageId);
            expectedVersion = serverMessage?.version;
          }
          if (expectedVersion == null) {
            throw new Error("Missing server message version");
          }
          await tldwClient.deleteMessage(
            serverMessageId,
            Number(expectedVersion),
            serverChatId ?? undefined,
          );
          invalidateServerChatHistory();
        }

        if (historyId) {
          await removeMessageByIndex(historyId, index);
        }
      } catch (err) {
        console.error("[deleteMessage] Failed to delete message", err);
        return;
      }

      setMessages((prev) => prev.filter((m) => m.id !== targetId));
      setHistory((prev) => {
        let removed = false;
        return prev.filter((h) => {
          if (
            !removed &&
            h.role === historyRole &&
            h.content === historyContent
          ) {
            removed = true;
            return false;
          }
          return true;
        });
      });
    },
    [
      clearReplyTarget,
      historyId,
      invalidateServerChatHistory,
      messages,
      replyTarget?.id,
      serverChatId,
      setHistory,
      setMessages,
    ],
  );

  const toggleMessagePinned = React.useCallback(
    async (index: number) => {
      const target = messages[index];
      if (!target) return;

      const targetId = target.id;
      const nextPinned = !Boolean(target.pinned);
      const serverMessageId = target.serverMessageId;
      const serverMessageVersion = target.serverMessageVersion;
      const messageText = String(target.message || "");

      try {
        if (serverMessageId) {
          await tldwClient.initialize().catch(() => null);
          let expectedVersion = serverMessageVersion;
          if (expectedVersion == null) {
            const serverMessage = await tldwClient.getMessage(serverMessageId);
            expectedVersion = serverMessage?.version;
          }
          if (expectedVersion == null) {
            throw new Error("Missing server message version");
          }
          await tldwClient.editMessage(
            serverMessageId,
            messageText,
            Number(expectedVersion),
            serverChatId ?? undefined,
            { pinned: nextPinned },
          );
          invalidateServerChatHistory();
        }
      } catch (err) {
        console.error("[toggleMessagePinned] Failed to toggle pin", err);
        return;
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === targetId ? { ...m, pinned: nextPinned } : m,
        ),
      );
    },
    [invalidateServerChatHistory, messages, serverChatId, setMessages],
  );

  const editMessage = createEditMessage({
    messages,
    history,
    setMessages,
    setHistory,
    historyId,
    // validateBeforeSubmitFn and onSubmit are bound later via bindSubmit
    validateBeforeSubmitFn: () => true,
    onSubmit: async () => {},
  });

  const createChatBranch = createBranchMessage({
    notification,
    historyId,
    setHistory,
    setHistoryId: setHistoryId as (id: string | null) => void,
    setMessages,
    setContext: setContextFiles,
    setSelectedSystemPrompt,
    setSystemPrompt: currentChatModelSettings.setSystemPrompt,
    serverChatId,
    scope,
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
    characterId: serverChatCharacterId ?? null,
    chatTitle: serverChatTitle ?? null,
    messages,
    history,
  });

  const createServerOnlyChatBranch = createBranchMessage({
    notification,
    historyId,
    setHistory,
    setHistoryId: setHistoryId as (id: string | null) => void,
    setMessages,
    setContext: setContextFiles,
    setSelectedSystemPrompt,
    setSystemPrompt: currentChatModelSettings.setSystemPrompt,
    serverChatId,
    scope,
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
    characterId: serverChatCharacterId ?? null,
    chatTitle: serverChatTitle ?? null,
    messages,
    history,
    serverOnly: true,
  });

  /**
   * bindSubmit wires onSubmit + validateBeforeSubmitFn into
   * editMessage and regenerateLastMessage after onSubmit is created
   * in the facade (breaks the circular dep).
   */
  const bindSubmit = (
    onSubmit: (...args: any[]) => Promise<void>,
    validateBeforeSubmitFn: () => boolean,
  ) => {
    const boundEditMessage = createEditMessage({
      messages,
      history,
      setMessages,
      setHistory,
      historyId,
      validateBeforeSubmitFn,
      onSubmit,
    });

    const boundRegenerateLastMessage = createRegenerateLastMessage({
      validateBeforeSubmitFn,
      history,
      messages,
      setHistory,
      setMessages,
      onSubmit,
      beforeSubmit: async ({ nextMessages }) => {
        if (!serverChatId) return;
        if (
          selectedCharacter?.id == null &&
          serverChatCharacterId == null
        )
          return;

        const branchIndex = nextMessages.length - 1;
        if (branchIndex < 0) return;

        let branchSnapshot:
          | import("@/hooks/handlers/messageHandlers").ServerChatBranchSnapshot
          | undefined;
        const branchedChatId = await createServerOnlyChatBranch(
          branchIndex,
          (snapshot) => {
            branchSnapshot = snapshot;
          },
        );
        if (!branchedChatId) {
          throw new Error("Failed to create branch for regeneration");
        }

        return {
          messages: branchSnapshot?.messages,
          submitExtras: {
            historyIdOverride: branchSnapshot?.historyId,
            serverChatIdOverride: branchedChatId,
          },
        };
      },
    });

    return {
      editMessage: boundEditMessage,
      regenerateLastMessage: boundRegenerateLastMessage,
    };
  };

  return {
    stopStreamingRequest,
    deleteMessage,
    toggleMessagePinned,
    editMessage,
    createChatBranch,
    createServerOnlyChatBranch,
    bindSubmit,
  };
};
