// @vitest-environment jsdom
import React from "react";
import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStoreMessageOption } from "@/store/option";
import { usePromptInteractions } from "../hooks/usePromptInteractions";

// The deployed library interaction hook, router, QueryClient and option setters
// are real. Usage recording and unrelated library/service discovery are the I/O
// doubles. No full PromptBody, native browser navigation or database assertion.
vi.mock("@/hooks/useMessageOption", async () => {
  const { useStoreMessageOption: store } = await import("@/store/option");
  return { useMessageOption: () => store.getState() };
});
vi.mock("@/services/application", () => ({
  getAllCopilotPrompts: async () => [],
  upsertCopilotPrompts: vi.fn(),
}));
vi.mock("@/services/prompt-studio", () => ({
  hasPromptStudio: async () => false,
  getPrompt: vi.fn(),
  getLlmProviders: async () => ({ providers: [] }),
}));
const clients = new Set<QueryClient>();
const userText = "  Exact library question\n";
const systemText = "Library system instruction";
const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
const mount = (markUsed: (id: string) => Promise<void>) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  clients.add(queryClient);
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter initialEntries={["/prompt"]}>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
  return renderHook(
    () => ({
      interactions: usePromptInteractions({
        queryClient,
        isOnline: false,
        t: (key, options) => String(options?.defaultValue ?? key),
        getPromptTexts: (record: {
          systemText?: string;
          userText?: string;
        }) => ({ systemText: record.systemText, userText: record.userText }),
        getPromptKeywords: () => [],
        getPromptRecordById: () => null,
        getPromptModifiedAt: () => 0,
        getPromptUsageCount: () => 0,
        getPromptLastUsedAt: () => null,
        editorMarkPromptAsUsed: markUsed,
      }),
      path: useLocation().pathname,
    }),
    { wrapper },
  );
};
beforeEach(() => {
  useStoreMessageOption.setState({
    selectedSystemPrompt: null,
    selectedQuickPrompt: null,
    quickPromptRevision: 0,
  });
});
afterEach(() => {
  cleanup();
  clients.forEach((client) => client.clear());
  clients.clear();
  vi.restoreAllMocks();
});

describe("library insertion does not await best-effort usage", () => {
  it.each(["system", "quick", "both"] as const)(
    "applies %s and navigates before held usage recording resolves",
    async (choice) => {
      const usage = deferred();
      const markUsed = vi.fn(() => usage.promise);
      const view = mount(markUsed);
      await act(async () => {
        await view.result.current.interactions.handleUsePromptInChat({
          id: "library-prompt",
          systemText,
          userText,
        });
      });
      expect(view.result.current.interactions.insertPrompt?.id).toBe(
        "library-prompt",
      );
      let completion!: Promise<void>;
      act(() => {
        completion =
          view.result.current.interactions.handleInsertChoice(choice);
      });
      try {
        expect(markUsed).toHaveBeenCalledExactlyOnceWith("library-prompt");
        expect(view.result.current.path).toBe("/chat");
        expect(useStoreMessageOption.getState().selectedSystemPrompt).toBe(
          choice === "quick" ? undefined : "library-prompt",
        );
        expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
          choice === "system" ? null : userText,
        );
        expect(view.result.current.interactions.insertPrompt).toBeNull();
      } finally {
        await act(async () => {
          usage.resolve();
          await completion;
        });
      }
    },
  );
  it("publishes a quick-only library prompt and navigates before its held usage record resolves", async () => {
    const usage = deferred();
    const markUsed = vi.fn(() => usage.promise);
    const view = mount(markUsed);
    let completion!: Promise<void>;
    act(() => {
      completion = view.result.current.interactions.handleUsePromptInChat({
        id: "quick-library-prompt",
        userText,
      });
    });
    try {
      expect(markUsed).toHaveBeenCalledExactlyOnceWith("quick-library-prompt");
      expect(view.result.current.path).toBe("/chat");
      expect(useStoreMessageOption.getState().selectedQuickPrompt).toBe(
        userText,
      );
      expect(
        useStoreMessageOption.getState().selectedSystemPrompt,
      ).toBeUndefined();
    } finally {
      await act(async () => {
        usage.resolve();
        await completion;
      });
    }
  });
});
