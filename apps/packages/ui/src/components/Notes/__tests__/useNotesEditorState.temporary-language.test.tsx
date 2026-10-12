import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  useNotesEditorState,
  type UseNotesEditorStateDeps,
} from "../hooks/useNotesEditorState";

const mocks = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.request }));
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({ id: "alice", is_active: true })),
  },
}));
vi.mock("../hooks/useNotesGraphAuthorityScope", () => ({
  createNotesGraphAuthorityScope: (_origin: string, user: string) => user,
}));
vi.mock("@/services/notes-tasks", () => ({
  listNoteTasks: vi.fn(async () => ({ tasks: [] })),
  listTaskActivity: vi.fn(async () => ({ events: [] })),
  markTaskActivityRead: vi.fn(),
  setNoteTaskStatus: vi.fn(),
}));
vi.mock("@/services/settings/registry", async (original) => ({
  ...(await original<typeof import("@/services/settings/registry")>()),
  getSetting: vi.fn(async () => null),
  setSetting: vi.fn(async () => undefined),
  clearSetting: vi.fn(),
}));
const noop = () => undefined;
function editor() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const deps: UseNotesEditorStateDeps = {
    authorityScope: "alice",
    connectionConfig: {
      serverUrl: "https://notes.test",
      authMode: "single-user",
      apiKey: "synthetic",
    },
    isOnline: true,
    isMobileViewport: false,
    message: {
      error: vi.fn(),
      success: vi.fn(),
      warning: vi.fn(),
    } as unknown as UseNotesEditorStateDeps["message"],
    confirmDanger: async () => true,
    queryClient,
    t: (key) => key,
    listMode: "active",
    setListMode: noop,
    data: [],
    refetch: async () => undefined,
    setPage: noop,
    setQuery: noop,
    setQueryInput: noop,
    setKeywordTokens: noop,
    setSelectedNotebookId: noop,
    setMobileSidebarOpen: noop,
    editorKeywords: [],
    setEditorKeywords: noop,
    keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: noop,
    setKeywordSuggestionSelection: noop,
    editorDisabled: false,
  };
  return renderHook(
    ({ scope }) => useNotesEditorState({ ...deps, authorityScope: scope }),
    {
      initialProps: { scope: "alice" },
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    },
  );
}
describe("temporary language at the actual canonical owner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });
  it("keeps temporary structured mode through ordinary old-peer save and same-note reload without publishing metadata", async () => {
    let stored = {
      id: "a",
      title: "Old peer",
      content: '<>& {"a":1}',
      version: 1,
    };
    const writes: Record<string, unknown>[] = [];
    mocks.request.mockImplementation(async ({ path, method, body }) => {
      if (path.endsWith("/capabilities")) return { content_language: false };
      if (method === "PUT") {
        writes.push(body);
        stored = { ...stored, ...body, version: stored.version + 1 };
        return stored;
      }
      return stored;
    });
    const view = editor();
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    await waitFor(() =>
      expect(view.result.current.languagePersistence).toBe("temporary"),
    );
    act(() => view.result.current.selectContentLanguage("json"));
    expect(view.result.current.contentLanguage).toBeUndefined();
    expect(view.result.current.effectiveContentLanguage).toBe("json");
    expect(view.result.current.isDirty).toBe(false);
    act(() => {
      view.result.current.setTitle("Title changed");
      view.result.current.setIsDirty(true);
    });
    await act(async () => {
      expect(await view.result.current.saveNote()).toBe(true);
    });
    expect(writes).toHaveLength(1);
    expect(writes[0]).not.toHaveProperty("content_language");
    expect(writes[0].content).toBe('<>& {"a":1}');
    expect(view.result.current.effectiveContentLanguage).toBe("json");
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    expect(view.result.current.effectiveContentLanguage).toBe("json");
    view.rerender({ scope: "bob" });
    view.rerender({ scope: "alice" });
    expect(view.result.current.effectiveContentLanguage).not.toBe("json");
  });
  it("never treats missing capability fields or transport failure as unsupported proof", async () => {
    mocks.request.mockImplementation(async ({ path }) =>
      path.endsWith("/capabilities")
        ? {}
        : { id: "a", content: "{}", version: 1 },
    );
    const view = editor();
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    await waitFor(() =>
      expect(mocks.request).toHaveBeenCalledWith(
        expect.objectContaining({ path: "/api/v1/notes/capabilities" }),
      ),
    );
    act(() => view.result.current.selectContentLanguage("json"));
    expect(view.result.current.languagePersistence).toBe("unavailable");
    expect(view.result.current.effectiveContentLanguage).toBeUndefined();
  });
  it("retires temporary view on new canonical metadata and preserves a future mode through an omitted same-note response", async () => {
    let stored: Record<string, unknown> = {
      id: "a",
      content: "{}",
      version: 1,
    };
    mocks.request.mockImplementation(async ({ path }) =>
      path.endsWith("/capabilities") ? { content_language: false } : stored,
    );
    const view = editor();
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    await waitFor(() =>
      expect(view.result.current.languagePersistence).toBe("temporary"),
    );
    act(() => view.result.current.selectContentLanguage("json"));
    stored = { ...stored, content_language: "future" };
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    expect(view.result.current.effectiveContentLanguage).toBe("future");
    delete stored.content_language;
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    act(() => view.result.current.selectContentLanguage("yaml"));
    expect(view.result.current.contentLanguage).toBe("future");
    expect(view.result.current.effectiveContentLanguage).toBe("future");
  });
  it("allows only fresh-default provenance through a first old-peer save", async () => {
    let stored: Record<string, unknown> = {};
    mocks.request.mockImplementation(async ({ path, method, body }) => {
      if (path.endsWith("/capabilities")) return { content_language: false };
      if (method === "POST") {
        expect(body).not.toHaveProperty("content_language");
        stored = { ...body, id: "created", version: 1 };
        return stored;
      }
      return stored;
    });
    const view = editor();
    await waitFor(() =>
      expect(view.result.current.languagePersistence).toBe("temporary"),
    );
    act(() => {
      view.result.current.selectContentLanguage("json");
      view.result.current.setContentDirty('{"a":1}');
    });
    await act(async () => {
      expect(await view.result.current.saveNote()).toBe(true);
    });
    expect(view.result.current.selectedId).toBe("created");
    expect(view.result.current.effectiveContentLanguage).toBe("json");
  });
});
