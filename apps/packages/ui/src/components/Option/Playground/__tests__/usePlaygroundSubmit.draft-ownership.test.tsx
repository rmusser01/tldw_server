// @vitest-environment jsdom
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { MemoryRouter, useNavigate } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText";
import {
  HistorySelectionProvider,
  useHistorySelectionContext,
} from "@/hooks/chat/useHistorySelection";
import type { UploadedFile } from "@/db/dexie/types";
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils";
import { useSlashCommands } from "@/hooks/playground/useSlashCommands";
import { useStoreChatModelSettings } from "@/store/model";
import { createSidepanelChatHandoff } from "@/services/sidepanel-chat-handoff";
import { useSidepanelChatHandoffImport } from "../hooks/useSidepanelChatHandoffImport";
import { usePlaygroundSubmit } from "../hooks/usePlaygroundSubmit";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) =>
      typeof fallback === "string" ? fallback : key,
  }),
}));

const services = vi.hoisted(() => ({
  account: "alice" as string | null,
  embedding: vi.fn<() => Promise<string | null>>(),
  simpleSearch: vi.fn<() => Promise<boolean>>(),
  upload: vi.fn(),
}));

vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: async () => {
    if (!services.account) throw new Error("Signed out");
    return {
      scopeKey: services.account,
      userId: services.account,
      config: {
        serverUrl: "http://localhost:8000",
        authMode: "multi-user",
        authSource: "manual",
        orgId: null,
      },
    };
  },
}));
vi.mock("~/services/tldw-server", () => ({
  defaultEmbeddingModelForRag: services.embedding,
}));
vi.mock("@/services/search", () => ({
  getIsSimpleInternetSearch: services.simpleSearch,
}));
vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/background-proxy")>()),
  bgUpload: (...args: unknown[]) => services.upload(...args),
}));

type Input = {
  webSearch?: boolean;
  isSending?: boolean;
  historyId?: string | null;
  selectedDocuments?: Array<{ id: number; title: string; url: string }>;
  uploadedFiles?: UploadedFile[];
  openUIRequestMode?: boolean;
  importHandoff?: boolean;
  route?: string;
};

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

const file = (id: string): UploadedFile => ({
  id,
  filename: `${id}.pdf`,
  type: "application/pdf",
  content: "synthetic document bytes",
  size: 3,
  uploadedAt: 1,
  processed: false,
  processingMode: "add_to_chat",
  processingStatus: "ready",
});

const t = (key: string, fallback?: unknown) =>
  typeof fallback === "string" ? fallback : key;

describe("Playground send owns only its captured composer draft", () => {
  const sendMessage = vi.fn();
  const clearSelectedDocuments = vi.fn();
  const clearUploadedFiles = vi.fn();
  const clearOpenUIRequestMode = vi.fn();
  const clearImportedSidepanelContext = vi.fn();
  const queueSubmission = vi.fn();

  const mount = (initial: Input = {}) =>
    renderHook(
      (input: Input) => {
        const textareaRef = React.useRef<HTMLTextAreaElement>(null);
        const [webSearch, setWebSearch] = React.useState(
          input.webSearch ?? false,
        );
        const [chatMode, setChatMode] = React.useState("normal");
        const composer = useComposerText({
          draftKey: "tldw:playgroundChatDraft:ownership-test",
          tabScopedDraft: true,
          textareaRef,
        });
        const systemPrompt = useStoreChatModelSettings(
          (state) => state.systemPrompt,
        );
        const navigate = useNavigate();
        const handoff = useSidepanelChatHandoffImport({
          draftValue: composer.form.values.message,
          setMessageValue: composer.setMessageValue,
          t,
        });
        const submissions = React.useRef<Array<Promise<void>>>([]);
        const form = {
          ...composer.form,
          onSubmit: (
            handler: (values: {
              message: string;
              image: string;
            }) => void | Promise<void>,
          ) => {
            const invoke = composer.form.onSubmit(handler);
            return (event?: React.FormEvent) => {
              const completion = invoke(event);
              submissions.current.push(completion);
              return completion;
            };
          },
        };
        const historySelection = useHistorySelectionContext()!;
        const slash = useSlashCommands({
          chatMode,
          setChatMode,
          webSearch,
          setWebSearch,
          handleImageUpload: vi.fn(),
          imageBackendDefaultTrimmed: "",
          imageBackendLabel: "none",
          setOpenModelSettings: vi.fn(),
          currentMessage: composer.form.values.message,
        });
        const submit = usePlaygroundSubmit({
          form,
          beginPromptAssistReset: composer.beginPromptAssistReset,
          markPromptAssistAttemptSaved: composer.markPromptAssistAttemptSaved,
          captureDraftOwnership: composer.captureDraftOwnership,
          captureHistorySelection: historySelection.fence,
          isSending: input.isSending ?? false,
          isConnectionReady: true,
          webSearch,
          chatMode,
          systemPrompt,
          compareModeActive: false,
          compareSelectedModels: [],
          selectedModel: "ready-model",
          historyId: input.historyId,
          serverChatId: null,
          fileRetrievalEnabled: false,
          ragPinnedResults: [],
          selectedDocuments: input.selectedDocuments ?? [],
          uploadedFiles: input.uploadedFiles ?? [],
          currentContextSnapshot: {
            historyId: input.historyId ?? null,
            promptSummary: systemPrompt ? "System prompt" : null,
          },
          conversationTokenCount: 0,
          characterContextTokenEstimate: 0,
          pinnedSourceTokenEstimate: 0,
          resolvedMaxContext: 100_000,
          jsonMode: false,
          openUIRequestMode: input.openUIRequestMode ?? false,
          importedSidepanelContext: input.importHandoff
            ? handoff.importedContext
            : null,
          clearImportedSidepanelContext: input.importHandoff
            ? handoff.removeImportedContext
            : clearImportedSidepanelContext,
          sendMessage,
          clearOpenUIRequestMode,
          clearSelectedDocuments,
          clearUploadedFiles,
          textAreaFocus: composer.textAreaFocus,
          setLastSubmittedContext: vi.fn(),
          estimateTokensForText: (text: string) => text.length,
          resolveSubmissionIntent: slash.resolveSubmissionIntent,
          queueSubmission,
          validateSelectedChatModelsAvailability: () => true,
          compareModelsSupportCapability: () => true,
          notificationApi: { error: vi.fn(), warning: vi.fn() },
          t,
        });
        return {
          composer,
          historySelection,
          submit,
          submissions,
          webSearch,
          chatMode,
          navigate,
          importedContext: handoff.importedContext,
          promptSummary: systemPrompt ? "System prompt" : null,
        };
      },
      {
        initialProps: initial,
        wrapper: ({ children }) => (
          <MemoryRouter initialEntries={[initial.route ?? "/chat"]}>
            <HistorySelectionProvider>{children}</HistorySelectionProvider>
          </MemoryRouter>
        ),
      },
    );

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    useStoreChatModelSettings.getState().reset();
    services.account = "alice";
    services.embedding.mockReset().mockResolvedValue("embedding-model");
    services.simpleSearch.mockReset().mockResolvedValue(false);
    services.upload.mockReset();
    sendMessage.mockReset().mockResolvedValue({ status: "submitted" });
    clearSelectedDocuments.mockReset();
    clearUploadedFiles.mockReset();
    clearOpenUIRequestMode.mockReset();
    clearImportedSidepanelContext.mockReset();
    queueSubmission.mockReset().mockReturnValue({ id: "queued-second" });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    { command: "/web", expectedWebSearch: true, expectedChatMode: "normal" },
    { command: "/search", expectedWebSearch: false, expectedChatMode: "rag" },
  ])(
    "applies bare $command and removes the consumed slash text",
    async ({ command, expectedWebSearch, expectedChatMode }) => {
      const view = mount();
      try {
        await waitFor(() =>
          expect(view.result.current.composer.draftReady).toBe(true),
        );
        act(() => view.result.current.composer.setMessageValue(command));
        act(() => view.result.current.submit.submitForm());
        await act(async () => {
          await Promise.all(view.result.current.submissions.current);
        });
        expect(view.result.current.webSearch).toBe(expectedWebSearch);
        expect(view.result.current.chatMode).toBe(expectedChatMode);
        expect(view.result.current.composer.form.values.message).toBe("");
        expect(sendMessage).not.toHaveBeenCalled();
      } finally {
        await act(async () => {
          await Promise.allSettled(view.result.current.submissions.current);
        });
        view.unmount();
      }
    },
  );

  it.each([
    {
      draft: "/web ask online",
      prompt: "ask online",
      webSearch: false,
      chatMode: "normal",
    },
    {
      draft: "/search ask notes",
      prompt: "ask notes",
      webSearch: true,
      chatMode: "rag",
    },
    {
      draft: "ask normally",
      prompt: "ask normally",
      webSearch: true,
      chatMode: "normal",
    },
  ])(
    "sends $draft once after a held embedding check",
    async ({ draft, prompt, webSearch, chatMode }) => {
      const embedding = deferred<string | null>();
      services.embedding.mockReturnValue(embedding.promise);
      const view = mount({ webSearch: true });
      try {
        await waitFor(() =>
          expect(view.result.current.composer.draftReady).toBe(true),
        );
        act(() => view.result.current.composer.setMessageValue(draft));
        act(() => view.result.current.submit.submitForm());
        await waitFor(() =>
          expect(services.embedding).toHaveBeenCalledTimes(1),
        );
        expect(view.result.current.webSearch).toBe(webSearch);
        expect(view.result.current.chatMode).toBe(chatMode);
        expect(sendMessage).not.toHaveBeenCalled();
        await act(async () => {
          embedding.resolve("embedding-model");
          await Promise.all(view.result.current.submissions.current);
        });
        expect(sendMessage).toHaveBeenCalledTimes(1);
        expect(sendMessage).toHaveBeenCalledWith(
          expect.objectContaining({ message: prompt }),
        );
        expect(view.result.current.composer.form.values.message).toBe("");
      } finally {
        await act(async () => {
          embedding.resolve("embedding-model");
          await Promise.allSettled(view.result.current.submissions.current);
        });
        view.unmount();
      }
    },
  );

  it("does not dispatch a held draft after the actual system prompt changes under the same summary", async () => {
    const embedding = deferred<string | null>();
    services.embedding.mockReturnValue(embedding.promise);
    act(() => useStoreChatModelSettings.getState().setSystemPrompt("Prompt A"));
    const view = mount({ webSearch: true });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() =>
        view.result.current.composer.setMessageValue("ask with prompt A"),
      );
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(services.embedding).toHaveBeenCalledTimes(1));
      act(() =>
        useStoreChatModelSettings.getState().setSystemPrompt("Prompt B"),
      );
      expect(view.result.current.promptSummary).toBe("System prompt");
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.all(view.result.current.submissions.current);
      });
      expect(sendMessage).not.toHaveBeenCalled();
      expect(view.result.current.composer.form.values.message).toBe(
        "ask with prompt A",
      );
    } finally {
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it("keeps a newer same-page import when an older held response completes", async () => {
    const pageContext = {
      title: "Same page",
      url: "https://example.test/same",
      snippets: [{ kind: "visible-context" as const, text: "Same notes" }],
    };
    const first = await createSidepanelChatHandoff({
      draftText: "first imported question",
      pageContext,
    });
    const second = await createSidepanelChatHandoff({
      draftText: "second imported question",
      pageContext,
    });
    const response = deferred<ChatSubmitResult>();
    sendMessage.mockReturnValue(response.promise);
    const view = mount({
      importHandoff: true,
      route: `/chat?handoff=${first.id}`,
    });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      await waitFor(() =>
        expect(view.result.current.composer.form.values.message).toBe(
          "first imported question",
        ),
      );
      await waitFor(() =>
        expect(view.result.current.importedContext).not.toBeNull(),
      );
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
      expect(view.result.current.composer.form.values.message).toBe("");
      act(() => view.result.current.navigate(`/chat?handoff=${second.id}`));
      await waitFor(() =>
        expect(view.result.current.composer.form.values.message).toBe(
          "second imported question",
        ),
      );
      const newerContext = view.result.current.importedContext;
      expect(newerContext).not.toBeNull();
      await act(async () => {
        response.resolve({ status: "submitted" });
        await Promise.all(view.result.current.submissions.current);
      });
      expect(view.result.current.importedContext).toBe(newerContext);
      expect(view.result.current.composer.form.values.message).toBe(
        "second imported question",
      );
    } finally {
      await act(async () => {
        response.resolve({ status: "submitted" });
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it.each(["replacement", "ABA", "image"])(
    "does not dispatch an old draft after a held embedding check and %s edit",
    async (change) => {
      const embedding = deferred<string | null>();
      services.embedding.mockReturnValue(embedding.promise);
      const view = mount({ webSearch: true });
      try {
        await waitFor(() =>
          expect(view.result.current.composer.draftReady).toBe(true),
        );
        act(() => view.result.current.composer.setMessageValue("first"));
        act(() => view.result.current.submit.submitForm());
        await waitFor(() =>
          expect(services.embedding).toHaveBeenCalledTimes(1),
        );
        if (change === "image") {
          act(() =>
            view.result.current.composer.form.setFieldValue(
              "image",
              "new-image",
            ),
          );
        } else {
          act(() => view.result.current.composer.setMessageValue("second"));
          if (change === "ABA") {
            act(() => view.result.current.composer.setMessageValue("first"));
          }
        }
        await act(async () => {
          embedding.resolve("embedding-model");
          await Promise.all(view.result.current.submissions.current);
        });
        expect(sendMessage).not.toHaveBeenCalled();
        expect(clearSelectedDocuments).not.toHaveBeenCalled();
        expect(clearUploadedFiles).not.toHaveBeenCalled();
        expect(view.result.current.composer.form.values).toEqual({
          message: change === "replacement" ? "second" : "first",
          image: change === "image" ? "new-image" : "",
        });
      } finally {
        await act(async () => {
          embedding.resolve("embedding-model");
          await Promise.allSettled(view.result.current.submissions.current);
        });
        view.unmount();
      }
    },
  );

  it("does not send Alice's held draft into Bob's account", async () => {
    const embedding = deferred<string | null>();
    services.embedding.mockReturnValue(embedding.promise);
    const view = mount({ webSearch: true });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() => view.result.current.composer.setMessageValue("Alice private"));
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(services.embedding).toHaveBeenCalledTimes(1));
      act(() => {
        services.account = "bob";
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"));
      });
      await waitFor(() =>
        expect(view.result.current.composer.form.values.message).toBe(""),
      );
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() => view.result.current.composer.setMessageValue("Bob private"));
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.all(view.result.current.submissions.current);
      });
      expect(sendMessage).not.toHaveBeenCalled();
      expect(view.result.current.composer.form.values.message).toBe(
        "Bob private",
      );
    } finally {
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it("does not dispatch after the history selection epoch resets at the same history ID", async () => {
    const embedding = deferred<string | null>();
    services.embedding.mockReturnValue(embedding.promise);
    const view = mount({ webSearch: true, historyId: "history-a" });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() => view.result.current.composer.setMessageValue("old view"));
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(services.embedding).toHaveBeenCalledTimes(1));
      act(() => view.result.current.historySelection.reset());
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.all(view.result.current.submissions.current);
      });
      expect(sendMessage).not.toHaveBeenCalled();
      expect(view.result.current.composer.form.values.message).toBe("old view");
    } finally {
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it("does not dispatch a held draft after the current history ID changes", async () => {
    const embedding = deferred<string | null>();
    services.embedding.mockReturnValue(embedding.promise);
    const view = mount({ webSearch: true, historyId: "history-a" });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() => view.result.current.composer.setMessageValue("old view"));
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(services.embedding).toHaveBeenCalledTimes(1));
      view.rerender({ webSearch: true, historyId: "history-b" });
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.all(view.result.current.submissions.current);
      });
      expect(sendMessage).not.toHaveBeenCalled();
      expect(view.result.current.composer.form.values.message).toBe("old view");
    } finally {
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it("admits one turn when Send is pressed twice during the same preparation", async () => {
    const embedding = deferred<string | null>();
    services.embedding.mockReturnValue(embedding.promise);
    const view = mount({ webSearch: true });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() => view.result.current.composer.setMessageValue("one question"));
      act(() => {
        view.result.current.submit.submitForm();
        view.result.current.submit.submitForm();
      });
      await waitFor(() => expect(services.embedding).toHaveBeenCalled());
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.all(view.result.current.submissions.current);
      });
      expect(sendMessage).toHaveBeenCalledTimes(1);
      expect(sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ message: "one question" }),
      );
    } finally {
      await act(async () => {
        embedding.resolve("embedding-model");
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it("keeps replacement tab attachments when old document preparation finishes", async () => {
    const preparation = deferred<{ text: string }>();
    services.upload.mockReturnValue(preparation.promise);
    const oldFile = file("old");
    const firstTab = {
      id: 1,
      title: "First",
      url: "https://example.test/first",
    };
    const nextTab = { id: 2, title: "Next", url: "https://example.test/next" };
    const view = mount({
      uploadedFiles: [oldFile],
      selectedDocuments: [firstTab],
    });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() => view.result.current.composer.setMessageValue("first"));
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(services.upload).toHaveBeenCalledTimes(1));
      view.rerender({ uploadedFiles: [oldFile], selectedDocuments: [nextTab] });
      await act(async () => {
        preparation.resolve({ text: "parsed notes" });
        await Promise.all(view.result.current.submissions.current);
      });
      await waitFor(() =>
        expect(view.result.current.submit.isPreparingDocuments).toBe(false),
      );
      expect(sendMessage).not.toHaveBeenCalled();
      expect(clearSelectedDocuments).not.toHaveBeenCalled();
      expect(clearUploadedFiles).not.toHaveBeenCalled();
      expect(view.result.current.composer.form.values.message).toBe("first");
    } finally {
      await act(async () => {
        preparation.resolve({ text: "parsed notes" });
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it("reaches the queue for a second draft while the first document turn streams", async () => {
    const preparation = deferred<{ text: string }>();
    const provider = deferred<ChatSubmitResult>();
    services.upload.mockReturnValue(preparation.promise);
    sendMessage.mockReturnValue(provider.promise);
    const firstFile = file("first-turn");
    const view = mount({ uploadedFiles: [firstFile] });
    try {
      await waitFor(() =>
        expect(view.result.current.composer.draftReady).toBe(true),
      );
      act(() =>
        view.result.current.composer.setMessageValue("first with document"),
      );
      act(() => view.result.current.submit.submitForm());
      await waitFor(() => expect(services.upload).toHaveBeenCalledTimes(1));
      await act(async () => {
        preparation.resolve({ text: "parsed first document" });
        await preparation.promise;
      });
      await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
      expect(view.result.current.composer.form.values.message).toBe("");
      view.rerender({ isSending: true, uploadedFiles: [] });
      act(() =>
        view.result.current.composer.setMessageValue("second while streaming"),
      );
      act(() => view.result.current.submit.submitForm());
      await act(async () => {
        await view.result.current.submissions.current[1];
      });
      expect(view.result.current.submit.isPreparingDocuments).toBe(false);
      expect(queueSubmission).toHaveBeenCalledTimes(1);
      expect(queueSubmission).toHaveBeenCalledWith(
        expect.objectContaining({ promptText: "second while streaming" }),
      );
      expect(sendMessage).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => {
        preparation.resolve({ text: "parsed first document" });
        provider.resolve({ status: "submitted" });
        await Promise.allSettled(view.result.current.submissions.current);
      });
      view.unmount();
    }
  });

  it.each([
    { label: "submitted", result: { status: "submitted" } as ChatSubmitResult },
    {
      label: "failed",
      result: {
        status: "failed",
        errorMessage: "Provider unavailable",
      } as ChatSubmitResult,
    },
    {
      label: "skipped",
      result: {
        status: "skipped",
        reason: "Request scope changed",
      } as ChatSubmitResult,
    },
  ])(
    "resets before a held $label response without invoking stale mode cleanup",
    async ({ result }) => {
      const response = deferred<ChatSubmitResult>();
      sendMessage.mockReturnValue(response.promise);
      const view = mount({ openUIRequestMode: true });
      try {
        await waitFor(() =>
          expect(view.result.current.composer.draftReady).toBe(true),
        );
        act(() => view.result.current.composer.setMessageValue("first"));
        act(() => view.result.current.submit.submitForm());
        await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
        expect(view.result.current.composer.form.values.message).toBe("");
        const clearsBeforeNewMode = clearOpenUIRequestMode.mock.calls.length;
        view.rerender({ openUIRequestMode: false });
        view.rerender({ openUIRequestMode: true });
        act(() => view.result.current.composer.setMessageValue("second"));
        await act(async () => {
          response.resolve(result);
          await Promise.all(view.result.current.submissions.current);
        });
        expect(view.result.current.composer.form.values.message).toBe("second");
        expect(clearOpenUIRequestMode).toHaveBeenCalledTimes(
          clearsBeforeNewMode,
        );
      } finally {
        await act(async () => {
          response.resolve(result);
          await Promise.allSettled(view.result.current.submissions.current);
        });
        view.unmount();
      }
    },
  );
});
