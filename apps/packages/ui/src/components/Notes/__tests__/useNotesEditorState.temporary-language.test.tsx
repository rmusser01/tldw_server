import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  useNotesEditorState,
  type UseNotesEditorStateDeps,
} from "../hooks/useNotesEditorState";

import { buildPortableNote, buildStructuredSource } from "../export-utils";

const mocks = vi.hoisted(() => ({ request: vi.fn(), activity: vi.fn() }));
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
  listTaskActivity: mocks.activity,
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
function editor(activeDraft = true) {
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
    ({
      scope,
      activeDraft = true,
      serverUrl = "https://notes.test",
    }: {
      scope: string;
      activeDraft?: boolean;
      serverUrl?: string;
    }) =>
      useNotesEditorState({
        ...deps,
        authorityScope: scope,
        hasActiveDraft: activeDraft,
        connectionConfig: { ...deps.connectionConfig!, serverUrl },
      }),
    {
      initialProps: { scope: "alice", activeDraft },
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
    mocks.activity.mockResolvedValue({ events: [] });
    localStorage.clear();
  });
  it.each([null, "future", undefined])(
    "negotiates for a selected note even when no new draft is active (%s)",
    async (language) => {
      mocks.request.mockImplementation(async ({ path }) =>
        path.endsWith("/capabilities")
          ? { content_language: true }
          : {
              id: "a",
              title: "Selected",
              content: "{}",
              version: 1,
              ...(language === undefined ? {} : { content_language: language }),
            },
      );
      const view = editor(false);
      await act(async () => {});
      expect(
        mocks.request.mock.calls.filter(([request]) =>
          request.path.endsWith("/capabilities"),
        ),
      ).toHaveLength(0);
      await act(async () => {
        await view.result.current.loadDetail("a");
      });
      await waitFor(() =>
        expect(view.result.current.languagePersistence).toBe("portable"),
      );
      expect(view.result.current.contentLanguage).toBe(language);
      expect(view.result.current.content).toBe("{}");
    },
  );

  it("negotiates for a recovered structured draft without marking it implicit", async () => {
    localStorage.setItem(
      "tldw:notesOfflineDraftQueue:v1:alice",
      JSON.stringify({
        "draft:new": {
          key: "draft:new",
          noteId: null,
          baseVersion: null,
          draftVersion: 3,
          implicitDefaultLanguage: false,
          title: "Recovered",
          content: "a: 1\n",
          content_language: "yaml",
          revision: 2,
          keywords: [],
          syncState: "conflict",
        },
      }),
    );
    mocks.request.mockImplementation(async ({ path }) =>
      path.endsWith("/capabilities") ? { content_language: true } : {},
    );
    const view = editor(false);
    await waitFor(() => {
      expect(view.result.current.title).toBe("Recovered");
      expect(view.result.current.languagePersistence).toBe("portable");
    });
    expect(view.result.current.content).toBe("a: 1\n");
    expect(view.result.current.contentLanguage).toBe("yaml");
    expect(view.result.current.implicitDefaultLanguage).toBe(false);
    expect(
      mocks.request.mock.calls.filter(
        ([request]) => request.method === "POST" || request.method === "PUT",
      ),
    ).toHaveLength(0);
  });

  it.each(["authority", "connection"] as const)(
    "rejects a late active-session capability reply after %s changes",
    async (transition) => {
      let release!: (value: Record<string, unknown>) => void;
      let signal: AbortSignal | undefined;
      const held = new Promise<Record<string, unknown>>((resolve) => {
        release = resolve;
      });
      mocks.request.mockImplementation(async (request) => {
        if (request.path.endsWith("/capabilities")) {
          signal = request.abortSignal;
          return held;
        }
        return {};
      });
      const view = editor(false);
      try {
        await act(async () => {});
        expect(
          mocks.request.mock.calls.filter(([request]) =>
            request.path.endsWith("/capabilities"),
          ),
        ).toHaveLength(0);
        view.rerender({ scope: "alice", activeDraft: true });
        await waitFor(() => expect(signal).toBeDefined());
        view.rerender({
          scope: transition === "authority" ? "bob" : "alice",
          serverUrl:
            transition === "connection"
              ? "https://other.test"
              : "https://notes.test",
          activeDraft: false,
        });
        await act(async () => {
          release({ content_language: true });
          await held;
        });
        expect(signal!.aborted).toBe(true);
        expect(view.result.current.languagePersistence).toBe("unavailable");
        expect(view.result.current.content).toBe("");
        expect(view.result.current.contentLanguage).toBeNull();
        view.rerender({
          scope: "alice",
          serverUrl: "https://notes.test",
          activeDraft: false,
        });
        await act(async () => {});
        expect(view.result.current.languagePersistence).toBe("unavailable");
        expect(
          mocks.request.mock.calls.filter(([request]) =>
            request.path.endsWith("/capabilities"),
          ),
        ).toHaveLength(1);
      } finally {
        release({ content_language: true });
        view.unmount();
      }
    },
  );

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
  it.each([
    "source",
    "mode",
    "temporary-mode",
    "selection",
    "authority",
    "current-markdown",
  ])(
    "qualifies only its exact internal Studio reload before another render (%s)",
    async (transition) => {
      const unsupported = transition === "temporary-mode";
      let generated = false;
      mocks.request.mockImplementation(async ({ path }) =>
        path.endsWith("/capabilities")
          ? { content_language: !unsupported }
          : {
              id: "a",
              title: "Owned",
              content: generated ? "Reloaded source" : "Saved source",
              version: 1,
              ...(unsupported ? {} : { content_language: "markdown" }),
            },
      );
      const view = editor();
      await act(async () => {
        await view.result.current.loadDetail("a");
      });
      await waitFor(() =>
        expect(view.result.current.languagePersistence).toBe(
          unsupported ? "temporary" : "portable",
        ),
      );
      if (unsupported)
        act(() => view.result.current.selectContentLanguage("markdown"));
      const admission = view.result.current.captureMarkdownWriter()!;
      expect(admission).not.toBeNull();
      let release!: (value: { events: [] }) => void;
      const pending = new Promise<{ events: [] }>((resolve) => {
        release = resolve;
      });
      mocks.activity.mockReturnValueOnce(pending);
      generated = true;
      let reloading!: ReturnType<typeof admission.reload>;
      act(() => {
        reloading = admission.reload("a");
      });
      await waitFor(() =>
        expect(view.result.current.content).toBe("Reloaded source"),
      );
      const rendered = view.result.current;
      await act(async () => {
        if (transition === "source")
          rendered.setContentDirty("New native source");
        if (transition === "mode" || transition === "temporary-mode")
          rendered.selectContentLanguage("yaml");
        if (transition === "selection") rendered.resetEditor();
        if (transition === "authority")
          window.dispatchEvent(new Event("tldw:auth-principal-changed"));
        expect(view.result.current).toBe(rendered);
        release({ events: [] });
        const completed = await reloading;
        expect(view.result.current).toBe(rendered);
        expect(completed?.isCurrent()).toBe(transition === "current-markdown");
        expect(completed?.loaded).toBe(transition === "current-markdown");
      });
      if (transition === "source")
        expect(view.result.current.content).toBe("New native source");
      if (transition === "temporary-mode")
        expect(view.result.current.effectiveContentLanguage).toBe("yaml");
    },
  );
});

describe("export capture at the actual canonical owner", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.activity.mockResolvedValue({ events: [] });
    localStorage.clear();
  });
  it.each([null, "json", "future/😀", undefined])(
    "reads current canonical source and mode before another render (%s)",
    async (language) => {
      mocks.request.mockImplementation(async ({ path }) =>
        path.endsWith("/capabilities")
          ? { content_language: true }
          : {
              id: "a",
              title: "Data",
              content: "old",
              version: 1,
              ...(language === undefined ? {} : { content_language: language }),
            },
      );
      const view = editor();
      await act(async () => {
        await view.result.current.loadDetail("a");
      });
      const owner = view.result.current;
      await act(async () => {
        owner.setContentDirty('{"x":9007199254740993}\n');
        expect(view.result.current).toBe(owner);
        expect(owner.captureExport()).toMatchObject({
          id: "a",
          content: '{"x":9007199254740993}\n',
          content_language: language,
        });
      });
    },
  );
  it("never exports a temporary old-peer view as portable metadata", async () => {
    mocks.request.mockImplementation(async ({ path }) =>
      path.endsWith("/capabilities")
        ? { content_language: false }
        : {
            id: "a",
            title: "Old",
            content: "old",
            version: 1,
          },
    );
    const view = editor();
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    await waitFor(() =>
      expect(view.result.current.languagePersistence).toBe("temporary"),
    );
    const owner = view.result.current;
    await act(async () => {
      owner.selectContentLanguage("json");
      owner.setContentDirty('{"x":9007199254740993}\n');
      expect(view.result.current).toBe(owner);
      expect(owner.captureExport()?.content_language).toBeUndefined();
      expect(owner.captureExport()?.content).toBe('{"x":9007199254740993}\n');
    });
  });
  it("keeps fresh old-peer temporary intent absent but preserves explicit null", async () => {
    mocks.request.mockResolvedValue({ content_language: false });
    const view = editor();
    await waitFor(() =>
      expect(view.result.current.languagePersistence).toBe("temporary"),
    );
    const owner = view.result.current;
    await act(async () => {
      owner.selectContentLanguage("json");
      owner.setContentDirty('{"x":9007199254740993}\n');
      expect(view.result.current).toBe(owner);
      expect(owner.captureExport()?.content_language).toBeUndefined();
      expect(buildStructuredSource(owner.captureExport()!).text).toBe(
        '{"x":9007199254740993}\n',
      );
      expect(buildStructuredSource(owner.captureExport()!).filename).toMatch(
        /\.txt$/,
      );
      expect(() => buildPortableNote(owner.captureExport()!)).toThrow(
        "known language metadata",
      );
      owner.setContentLanguage(null);
      expect(owner.captureExport()?.content_language).toBeNull();
      expect(
        JSON.parse(buildPortableNote(owner.captureExport()!)).content_language,
      ).toBeNull();
    });
  });
  it("retires a stale action on authority or selection change", async () => {
    mocks.request.mockResolvedValue({
      id: "a",
      title: "A",
      content: "a",
      version: 1,
      content_language: null,
    });
    const view = editor();
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    const owner = view.result.current;
    act(() => view.result.current.resetEditor());
    expect(owner.captureExport()).toBeNull();
    const current = view.result.current;
    view.rerender({ scope: "bob" });
    expect(current.captureExport()).toBeNull();
  });
});
