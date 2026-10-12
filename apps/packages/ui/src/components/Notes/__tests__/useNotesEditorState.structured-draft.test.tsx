import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  normalizeOfflineDraftQueue,
  NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY,
} from "../notes-manager-utils";
import {
  useNotesEditorState,
  type UseNotesEditorStateDeps,
} from "../hooks/useNotesEditorState";

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  tasks: vi.fn(),
  activity: vi.fn(),
  error: vi.fn(),
}));
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.request }));
vi.mock("@/services/notes-tasks", () => ({
  listNoteTasks: mocks.tasks,
  listTaskActivity: mocks.activity,
  markTaskActivityRead: vi.fn(),
  setNoteTaskStatus: vi.fn(),
}));
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({ id: "alice", is_active: true })),
  },
}));
vi.mock("../hooks/useNotesGraphAuthorityScope", () => ({
  createNotesGraphAuthorityScope: (_origin: string, user: string) => user,
}));
vi.mock("@/services/settings/registry", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/services/settings/registry")>()),
  getSetting: vi.fn(async () => null),
  setSetting: vi.fn(async () => undefined),
  clearSetting: vi.fn(),
}));

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};

const renderEditor = (
  isOnline = false,
  refetch = vi.fn(async () => undefined),
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const deps: UseNotesEditorStateDeps = {
    authorityScope: "alice",
    isOnline,
    connectionConfig: {
      serverUrl: "https://notes.example.test",
      authMode: "single-user",
      apiKey: "test-key",
    },
    isMobileViewport: false,
    message: {
      error: mocks.error,
      success: vi.fn(),
      warning: vi.fn(),
    } as unknown as UseNotesEditorStateDeps["message"],
    confirmDanger: vi.fn(async () => true),
    queryClient,
    t: (key) => key,
    listMode: "active",
    setListMode: vi.fn(),
    data: [],
    refetch,
    setPage: vi.fn(),
    setQuery: vi.fn(),
    setQueryInput: vi.fn(),
    setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(),
    setMobileSidebarOpen: vi.fn(),
    editorKeywords: [],
    setEditorKeywords: vi.fn(),
    keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(),
    setKeywordSuggestionSelection: vi.fn(),
    editorDisabled: false,
  };
  return renderHook(
    ({ scope, online }) =>
      useNotesEditorState({ ...deps, authorityScope: scope, isOnline: online }),
    {
      initialProps: { scope: "alice", online: isOnline },
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      ),
    },
  );
};

describe("structured drafts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mocks.tasks.mockResolvedValue({ tasks: [], reconciliation: null });
    mocks.activity.mockResolvedValue({ events: [] });
  });

  it("keeps legacy absent language unknown and preserves future language exactly", () => {
    const queue = normalizeOfflineDraftQueue({
      legacy: { content: "<>&" },
      future: { content: "雪", content_language: " future ", revision: 7 },
    });
    expect(queue.legacy.content_language).toBeUndefined();
    expect(queue.future).toMatchObject({
      content: "雪",
      content_language: " future ",
      revision: 7,
      draftVersion: 3,
    });
  });

  it("retains an empty language-bearing draft through a hook restart", async () => {
    mocks.request.mockResolvedValue({
      id: "n",
      title: "Saved",
      content: "Original",
      content_language: "json",
      version: 1,
    });
    const first = renderEditor();
    await act(async () => {
      await first.result.current.loadDetail("n");
    });
    act(() => {
      first.result.current.setContent("");
      first.result.current.setContentLanguage("yaml");
      first.result.current.setIsDirty(true);
    });
    await act(async () => {
      await first.result.current.saveNote({ showSuccessMessage: false });
    });
    first.unmount();
    const second = renderEditor();
    await act(async () => {
      await second.result.current.loadDetail("n");
    });
    expect(second.result.current.content).toBe("");
    expect(second.result.current.contentLanguage).toBe("yaml");
    expect(second.result.current.markdownAllowed).toBe(false);
  });

  it("does not derive WYSIWYG before language is known or for future modes", async () => {
    const loading = deferred<unknown>();
    mocks.request.mockReturnValueOnce(loading.promise);
    const view = renderEditor();
    let pending!: Promise<boolean>;
    act(() => {
      view.result.current.setEditorInputMode("wysiwyg");
      pending = view.result.current.loadDetail("n");
    });
    expect(view.result.current.editorInputMode).toBe("markdown");
    expect(view.result.current.markdownAllowed).toBe(false);
    await act(async () => {
      loading.resolve({
        id: "n",
        content: "<>&",
        content_language: "future-mode",
        version: 1,
      });
      await pending;
    });
    mocks.request.mockResolvedValue({
      id: "n",
      title: "Note",
      content: "<>&",
      content_language: "future-mode",
      version: 1,
    });
    await act(async () => {
      await view.result.current.loadDetail("n");
    });
    expect(view.result.current.content).toBe("<>&");
    expect(view.result.current.editorInputMode).toBe("markdown");
  });

  it("retains a newer language revision after an older complete acknowledgement", async () => {
    const response = deferred<unknown>();
    mocks.request.mockImplementation(({ method, path }) => {
      if (path.endsWith("/capabilities"))
        return Promise.resolve({ content_language: true });
      if (method === "PUT") return response.promise;
      return Promise.resolve({
        id: "n",
        title: "Saved",
        content: "old",
        content_language: "json",
        version: 1,
      });
    });
    const view = renderEditor(true);
    await act(async () => {
      await view.result.current.loadDetail("n");
    });
    act(() => {
      view.result.current.setContentLanguage("yaml");
    });
    let saving!: Promise<boolean>;
    act(() => {
      saving = view.result.current.saveNote({ showSuccessMessage: false });
    });
    await waitFor(() =>
      expect(
        mocks.request.mock.calls.some(([arg]) => arg.method === "PUT"),
      ).toBe(true),
    );
    act(() => {
      view.result.current.setContentLanguage("jsonl");
      view.result.current.setContent("new");
    });
    await act(async () => {
      response.resolve({
        id: "n",
        content: "old",
        content_language: "yaml",
        version: 2,
      });
      await saving;
    });
    expect(view.result.current.contentLanguage).toBe("jsonl");
    expect(view.result.current.isDirty).toBe(true);
    expect(
      view.result.current.offlineDraftQueue["note:n"].content_language,
    ).toBe("jsonl");
    expect(view.result.current.offlineDraftQueue["note:n"].content).toBe("new");
  });

  it("keeps a language draft after an incomplete acknowledgement", async () => {
    mocks.request.mockImplementation(({ method, path }) => {
      if (path.endsWith("/capabilities"))
        return Promise.resolve({ content_language: true });
      if (method === "PUT")
        return Promise.resolve({ id: "n", content: "old", version: 2 });
      return Promise.resolve({
        id: "n",
        title: "Saved",
        content: "old",
        content_language: "json",
        version: 1,
      });
    });
    const view = renderEditor(true);
    await act(async () => {
      await view.result.current.loadDetail("n");
    });
    act(() => {
      view.result.current.setContentLanguage("yaml");
    });
    await act(async () => {
      expect(
        await view.result.current.saveNote({ showSuccessMessage: false }),
      ).toBe(false);
    });
    expect(
      view.result.current.offlineDraftQueue["note:n"].content_language,
    ).toBe("yaml");
    expect(view.result.current.saveIndicator).toBe("error");
  });

  it("blocks navigation on quota failure and keeps an empty draft", async () => {
    mocks.request.mockResolvedValue({
      id: "n",
      title: "Saved",
      content: "old",
      content_language: "json",
      version: 1,
    });
    const view = renderEditor();
    await act(async () => {
      await view.result.current.loadDetail("n");
    });
    act(() => {
      view.result.current.setContent("");
      view.result.current.setIsDirty(true);
    });
    const storage = vi
      .spyOn(Object.getPrototypeOf(window.localStorage), "setItem")
      .mockImplementation(() => {
        throw new Error("quota");
      });
    try {
      await act(async () => {
        expect(await view.result.current.handleSelectNote("other")).toBe(false);
      });
      expect(view.result.current.content).toBe("");
      expect(view.result.current.selectedId).toBe("n");
      expect(storage).toHaveBeenCalled();
    } finally {
      storage.mockRestore();
    }
  });
  it("starts fresh offline notes in known default mode and resets to that mode", () => {
    const view = renderEditor();
    expect(view.result.current.contentLanguage).toBeNull();
    expect(view.result.current.markdownAllowed).toBe(true);
    act(() => {
      view.result.current.setEditorInputMode("wysiwyg");
    });
    expect(view.result.current.editorInputMode).toBe("wysiwyg");
    act(() => {
      view.result.current.setContentLanguage("yaml");
    });
    act(() => {
      view.result.current.resetEditor();
    });
    expect(view.result.current.contentLanguage).toBeNull();
    expect(view.result.current.markdownAllowed).toBe(true);
  });

  it.each([1, 2])(
    "only fills legacy language from its matching baseline (remote v%s)",
    async (remoteVersion) => {
      localStorage.setItem(
        `${NOTES_OFFLINE_DRAFT_QUEUE_STORAGE_KEY}:alice`,
        JSON.stringify({
          "note:n": {
            key: "note:n",
            noteId: "n",
            content: "legacy <>&",
            baseVersion: 1,
          },
        }),
      );
      mocks.request.mockResolvedValue({
        id: "n",
        content: "remote",
        content_language: "yaml",
        version: remoteVersion,
      });
      const view = renderEditor();
      await act(async () => {
        await view.result.current.loadDetail("n");
      });
      expect(view.result.current.content).toBe("legacy <>&");
      expect(view.result.current.contentLanguage).toBe(
        remoteVersion === 1 ? "yaml" : undefined,
      );
      expect(view.result.current.markdownAllowed).toBe(false);
    },
  );

  it("does not persist newly selected note B into note A after a deferred queue reply", async () => {
    const response = deferred<unknown>();
    mocks.request.mockImplementation(({ method, path }) => {
      if (path.endsWith("/capabilities"))
        return Promise.resolve({ content_language: true });
      if (method === "PUT") return response.promise;
      return Promise.resolve(
        path.includes("/b")
          ? {
              id: "b",
              title: "B",
              content: "B source",
              content_language: "json",
              version: 8,
            }
          : {
              id: "a",
              title: "A",
              content: "A source",
              content_language: "yaml",
              version: 1,
            },
      );
    });
    const view = renderEditor();
    await act(async () => {
      await view.result.current.loadDetail("a");
    });
    act(() => {
      view.result.current.setContent("A edit");
      view.result.current.setIsDirty(true);
    });
    await act(async () => {
      await view.result.current.saveNote({ showSuccessMessage: false });
    });
    view.rerender({ scope: "alice", online: true });
    await waitFor(() =>
      expect(
        mocks.request.mock.calls.some(([arg]) => arg.method === "PUT"),
      ).toBe(true),
    );
    await act(async () => {
      await view.result.current.handleSelectNote("b");
    });
    await act(async () => {
      response.resolve({
        id: "a",
        content: "A edit",
        content_language: "yaml",
        version: 2,
      });
    });
    expect(view.result.current.selectedId).toBe("b");
    expect(view.result.current.content).toBe("B source");
    expect(view.result.current.selectedVersion).toBe(8);
    expect(view.result.current.offlineDraftQueue["note:a"]).toBeUndefined();
    expect(
      mocks.request.mock.calls.filter(([arg]) => arg.method === "PUT"),
    ).toHaveLength(1);
  });

  it.each([null, "n"])(
    "excludes ordinary save while queued %s request is pending",
    async (noteId) => {
      const response = deferred<unknown>();
      const method = noteId ? "PUT" : "POST";
      mocks.request.mockImplementation(({ method: requestMethod, path }) => {
        if (path.endsWith("/capabilities"))
          return Promise.resolve({ content_language: true });
        if (requestMethod === method) return response.promise;
        return Promise.resolve({
          id: "n",
          title: "Note",
          content: "source",
          content_language: "yaml",
          version: 1,
        });
      });
      const view = renderEditor();
      if (noteId)
        await act(async () => {
          await view.result.current.loadDetail(noteId);
        });
      act(() => {
        view.result.current.setContent("queued");
        view.result.current.setContentLanguage("yaml");
        view.result.current.setIsDirty(true);
      });
      await act(async () => {
        await view.result.current.saveNote({ showSuccessMessage: false });
      });
      view.rerender({ scope: "alice", online: true });
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.filter(([arg]) => arg.method === method),
        ).toHaveLength(1),
      );
      let ordinary!: Promise<boolean>;
      act(() => {
        ordinary = view.result.current.saveNote({ showSuccessMessage: false });
      });
      await act(async () => {
        await Promise.resolve();
      });
      const count = mocks.request.mock.calls.filter(
        ([arg]) => arg.method === method,
      ).length;
      await act(async () => {
        response.resolve({
          id: "n",
          content: "queued",
          content_language: "yaml",
          version: 2,
        });
        await ordinary;
      });
      expect(count).toBe(1);
    },
  );
  it.each(["save", "edit"])(
    "keeps an incompatible explicitly selected null draft stable until %s retries",
    async (retry) => {
      let compatible = false;
      let probes = 0;
      mocks.request.mockImplementation(({ path, body }) => {
        if (path.endsWith("/capabilities")) {
          probes += 1;
          return Promise.resolve(compatible ? { content_language: true } : {});
        }
        return Promise.resolve({
          id: "n",
          title: "New",
          content: body?.content ?? "default source",
          content_language: null,
          version: 1,
        });
      });
      const view = renderEditor();
      act(() => {
        view.result.current.setContent("default source");
        view.result.current.setContentLanguage(null);
        view.result.current.setIsDirty(true);
      });
      await act(async () => {
        await view.result.current.saveNote({ showSuccessMessage: false });
      });
      view.rerender({ scope: "alice", online: true });
      await waitFor(() =>
        expect(
          view.result.current.offlineDraftQueue["draft:new"]?.syncState,
        ).toBe("error"),
      );
      expect(probes).toBe(2);
      view.rerender({ scope: "alice", online: false });
      view.rerender({ scope: "alice", online: true });
      await act(async () => {
        await Promise.resolve();
      });
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 300));
      });
      expect(probes).toBe(3);
      expect(
        view.result.current.offlineDraftQueue["draft:new"].content_language,
      ).toBeNull();
      expect(view.result.current.offlineDraftQueue["draft:new"].content).toBe(
        "default source",
      );
      expect(
        view.result.current.offlineDraftQueue["draft:new"].lastError,
      ).toContain("Portable language is unavailable");
      expect(
        mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
      ).toBe(false);
      compatible = true;
      if (retry === "save") {
        await act(async () => {
          expect(
            await view.result.current.saveNote({ showSuccessMessage: false }),
          ).toBe(true);
        });
      } else {
        act(() => {
          view.result.current.setContent("edited default source");
          view.result.current.setIsDirty(true);
        });
        await waitFor(() =>
          expect(
            mocks.request.mock.calls.filter(([arg]) => arg.method === "POST"),
          ).toHaveLength(1),
        );
      }
      expect(
        mocks.request.mock.calls.filter(([arg]) => arg.method === "POST"),
      ).toHaveLength(1);
    },
  );
  it("resets loading ownership when New interrupts a deferred detail request", async () => {
    const response = deferred<unknown>();
    mocks.request.mockReturnValue(response.promise);
    const view = renderEditor();
    let loading!: Promise<boolean>;
    act(() => {
      loading = view.result.current.loadDetail("old");
    });
    expect(view.result.current.markdownAllowed).toBe(false);
    act(() => {
      view.result.current.resetEditor();
      view.result.current.setEditorInputMode("wysiwyg");
    });
    expect(view.result.current.markdownAllowed).toBe(true);
    expect(view.result.current.editorInputMode).toBe("wysiwyg");
    await act(async () => {
      response.resolve({
        id: "old",
        content: "old",
        content_language: "yaml",
        version: 1,
      });
      await loading;
    });
    expect(view.result.current.selectedId).toBeNull();
    expect(view.result.current.contentLanguage).toBeNull();
    expect(view.result.current.markdownAllowed).toBe(true);
  });

  it.each(["ordinary", "restart"])(
    "creates an untouched default on a legacy server via %s",
    async (pathway) => {
      mocks.request.mockImplementation(({ body }) =>
        Promise.resolve({
          id: "legacy",
          content: body?.content ?? "<>& legacy",
          version: 1,
        }),
      );
      let view = renderEditor(pathway === "ordinary");
      act(() => {
        view.result.current.setContent("<>& legacy");
        view.result.current.setIsDirty(true);
      });
      if (pathway === "restart") {
        await act(async () => {
          await view.result.current.saveNote({ showSuccessMessage: false });
        });
        view.unmount();
        view = renderEditor();
        await waitFor(() =>
          expect(view.result.current.content).toBe("<>& legacy"),
        );
        view.rerender({ scope: "alice", online: true });
        await waitFor(() =>
          expect(
            mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
          ).toBe(true),
        );
        await waitFor(() =>
          expect(
            view.result.current.offlineDraftQueue["draft:new"],
          ).toBeUndefined(),
        );
      } else {
        await act(async () => {
          expect(
            await view.result.current.saveNote({ showSuccessMessage: false }),
          ).toBe(true);
        });
      }
      const posts = mocks.request.mock.calls.filter(
        ([arg]) => arg.method === "POST",
      );
      expect(posts).toHaveLength(1);
      expect(posts[0][0].body).not.toHaveProperty("content_language");
      expect(posts[0][0].body.content).toBe("<>& legacy");
    },
  );

  it("treats explicit same-null selection as a portable mutation and retains it on legacy servers", async () => {
    mocks.request.mockResolvedValue({});
    const view = renderEditor(true);
    act(() => {
      view.result.current.setContentLanguage(null);
    });
    expect(view.result.current.isDirty).toBe(true);
    act(() => {
      view.result.current.setContent("explicit null");
    });
    await act(async () => {
      expect(
        await view.result.current.saveNote({ showSuccessMessage: false }),
      ).toBe(false);
    });
    expect(
      mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
    ).toBe(false);
    expect(
      view.result.current.offlineDraftQueue["draft:new"].content_language,
    ).toBeNull();
  });

  it.each(
    [
      { id: "n", version: 1 },
      { id: "n", version: 1, content: "wrong" },
      { id: {}, version: 1, content: "exact" },
      { id: "", version: 1, content: "exact" },
      { id: "n", content: "exact" },
      { id: "n", version: -1, content: "exact" },
      {
        id: "n",
        version: 1,
        content: "exact",
        content_language: { invalid: true },
      },
      {
        id: "n",
        version: 1,
        content: "exact",
        content_language: "😀".repeat(101),
      },
    ].flatMap((reply) =>
      ["ordinary", "queue"].map((pathway) => ({ reply, pathway })),
    ),
  )(
    "retains an untouched default after an incomplete legacy acknowledgement %j",
    async ({ reply, pathway }) => {
      mocks.request.mockImplementation(({ method }) =>
        Promise.resolve(method === "POST" ? reply : {}),
      );
      const view = renderEditor(pathway === "ordinary");
      act(() => {
        view.result.current.setContent("exact");
        view.result.current.setIsDirty(true);
      });
      if (pathway === "ordinary") {
        await act(async () => {
          expect(
            await view.result.current.saveNote({ showSuccessMessage: false }),
          ).toBe(false);
        });
      } else {
        await act(async () => {
          await view.result.current.saveNote({ showSuccessMessage: false });
        });
        view.rerender({ scope: "alice", online: true });
        await waitFor(() =>
          expect(
            view.result.current.offlineDraftQueue["draft:new"]?.syncState,
          ).toBe("error"),
        );
      }
      expect(
        mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
      ).toBe(true);
      expect(view.result.current.offlineDraftQueue["draft:new"].content).toBe(
        "exact",
      );
      expect(view.result.current.saveIndicator).not.toBe("saved");
    },
  );

  it("does not infer fresh-default provenance for historical or existing-note entries", () => {
    const queue = normalizeOfflineDraftQueue({
      historical: { content: "x", content_language: null },
      old: {
        draftVersion: 2,
        implicitDefaultLanguage: true,
        content: "x",
        content_language: null,
      },
      "note:n": {
        draftVersion: 3,
        implicitDefaultLanguage: true,
        noteId: "n",
        content: "x",
        content_language: null,
      },
      "draft:new": {
        draftVersion: 3,
        implicitDefaultLanguage: true,
        noteId: null,
        baseVersion: null,
        content: "x",
        content_language: null,
      },
    });
    expect(queue.historical.implicitDefaultLanguage).not.toBe(true);
    expect(queue.old.implicitDefaultLanguage).not.toBe(true);
    expect(queue["note:n"].implicitDefaultLanguage).not.toBe(true);
    expect(queue["draft:new"].implicitDefaultLanguage).toBe(true);
  });

  it("retains a same-null selection made while an implicit-default POST is pending", async () => {
    const response = deferred<unknown>();
    mocks.request.mockImplementation(({ method }) =>
      method === "POST" ? response.promise : Promise.resolve({}),
    );
    const view = renderEditor(true);
    act(() => {
      view.result.current.setContent("exact");
      view.result.current.setIsDirty(true);
    });
    let saving!: Promise<boolean>;
    act(() => {
      saving = view.result.current.saveNote({ showSuccessMessage: false });
    });
    await waitFor(() =>
      expect(
        mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
      ).toBe(true),
    );
    act(() => {
      view.result.current.setContentLanguage(null);
    });
    await act(async () => {
      response.resolve({ id: "n", content: "exact", version: 1 });
      await saving;
    });
    expect(view.result.current.offlineDraftQueue["note:n"]).toMatchObject({
      content: "exact",
      content_language: null,
      implicitDefaultLanguage: false,
    });
    expect(view.result.current.isDirty).toBe(true);
    expect(view.result.current.saveIndicator).not.toBe("saved");
  });

  it.each(["ordinary", "queue"])(
    "rejects a different note identity in a protected %s acknowledgement",
    async (pathway) => {
      mocks.request.mockImplementation(({ method, path }) => {
        if (path.endsWith("/capabilities"))
          return Promise.resolve({ content_language: true });
        return Promise.resolve({
          id: method === "PUT" ? "wrong-note" : "n",
          content: "exact",
          content_language: "yaml",
          version: method === "PUT" ? 2 : 1,
        });
      });
      const view = renderEditor(pathway === "ordinary");
      await act(async () => {
        await view.result.current.loadDetail("n");
      });
      act(() => {
        view.result.current.setIsDirty(true);
      });
      if (pathway === "ordinary") {
        await act(async () => {
          expect(
            await view.result.current.saveNote({ showSuccessMessage: false }),
          ).toBe(false);
        });
      } else {
        await act(async () => {
          await view.result.current.saveNote({ showSuccessMessage: false });
        });
        view.rerender({ scope: "alice", online: true });
        await waitFor(() =>
          expect(
            view.result.current.offlineDraftQueue["note:n"]?.syncState,
          ).toBe("error"),
        );
      }
      expect(view.result.current.offlineDraftQueue["note:n"].content).toBe(
        "exact",
      );
    },
  );
  it.each(
    ["ordinary", "queue"].flatMap((pathway) =>
      [undefined, null, "yaml", "future-mode", "   ", "😀".repeat(51)].map(
        (language) => ({
          pathway,
          language,
        }),
      ),
    ),
  )(
    "saves newer body-only edits after a deferred legacy first POST via $pathway $language",
    async ({ pathway, language }) => {
      const firstReply = deferred<unknown>();
      const firstRefresh = deferred<void>();
      let remote = {
        id: "created",
        content: "older <>&",
        version: 7,
        content_language: language,
      };
      mocks.request.mockImplementation(({ method, path, body }) => {
        if (path.endsWith("/capabilities"))
          return Promise.resolve({ content_language: true });
        if (method === "POST") return firstReply.promise;
        if (method === "PUT") {
          remote = {
            id: "created",
            content: body.content,
            version: 8,
            content_language: language,
          };
          return Promise.resolve(remote);
        }
        return Promise.resolve(remote);
      });
      const refetch = vi.fn(async () => {
        if (pathway === "ordinary") await firstRefresh.promise;
      });
      const view = renderEditor(pathway === "ordinary", refetch);
      act(() => {
        view.result.current.setContent("older <>&");
        view.result.current.setIsDirty(true);
      });
      let saving!: Promise<boolean>;
      if (pathway === "ordinary") {
        act(() => {
          saving = view.result.current.saveNote({ showSuccessMessage: false });
        });
      } else {
        await act(async () => {
          await view.result.current.saveNote({ showSuccessMessage: false });
        });
        view.rerender({ scope: "alice", online: true });
      }
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
        ).toBe(true),
      );
      act(() => {
        view.result.current.setContent("newer <>&\n雪");
        view.result.current.setIsDirty(true);
      });
      if (pathway === "queue") view.rerender({ scope: "alice", online: false });
      await act(async () => {
        firstReply.resolve(remote);
      });
      await waitFor(() =>
        expect(view.result.current.selectedId).toBe("created"),
      );
      expect(view.result.current.content).toBe("newer <>&\n雪");
      expect(view.result.current.contentLanguage).toBe(language);
      expect(view.result.current.selectedVersion).toBe(7);
      expect(
        view.result.current.offlineDraftQueue["note:created"],
      ).toMatchObject({
        content: "newer <>&\n雪",
        baseVersion: 7,
        implicitDefaultLanguage: false,
      });
      expect(
        view.result.current.offlineDraftQueue["note:created"].content_language,
      ).toBe(language);
      if (pathway === "ordinary") {
        await act(async () => {
          firstRefresh.resolve();
          await saving;
        });
        await act(async () => {
          expect(
            await view.result.current.saveNote({ showSuccessMessage: false }),
          ).toBe(true);
        });
      } else {
        view.rerender({ scope: "alice", online: true });
        await waitFor(() =>
          expect(
            view.result.current.offlineDraftQueue["note:created"],
          ).toBeUndefined(),
        );
      }
      const puts = mocks.request.mock.calls.filter(
        ([arg]) => arg.method === "PUT",
      );
      expect(puts).toHaveLength(1);
      expect(puts[0][0].body.content).toBe("newer <>&\n雪");
      if (language === undefined)
        expect(puts[0][0].body).not.toHaveProperty("content_language");
      else expect(puts[0][0].body.content_language).toBe(language);
      expect(puts[0][0].headers["expected-version"]).toBe("7");
      expect(puts[0][0].path).toBe("/api/v1/notes/created");
      expect(
        view.result.current.offlineDraftQueue["note:created"],
      ).toBeUndefined();
      if (language !== undefined) {
        expect(
          mocks.request.mock.calls.some(([arg]) =>
            arg.path.endsWith("/capabilities"),
          ),
        ).toBe(true);
      }
      expect(view.result.current.isDirty).toBe(false);
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 300));
      });
      expect(
        view.result.current.offlineDraftQueue["note:created"],
      ).toBeUndefined();
    },
  );

  it.each(
    ["ordinary", "queue"].flatMap((pathway) =>
      [null, "yaml", "future-mode"].map((language) => ({ pathway, language })),
    ),
  )(
    "preserves explicit mode edits during first POST $pathway $language",
    async ({ pathway, language }) => {
      const reply = deferred<unknown>();
      mocks.request.mockImplementation(({ method }) =>
        method === "POST" ? reply.promise : Promise.resolve({}),
      );
      const view = renderEditor(pathway === "ordinary");
      act(() => {
        view.result.current.setContent("older");
        view.result.current.setIsDirty(true);
      });
      let saving!: Promise<boolean>;
      if (pathway === "ordinary")
        act(() => {
          saving = view.result.current.saveNote({ showSuccessMessage: false });
        });
      else {
        await act(async () => {
          await view.result.current.saveNote({ showSuccessMessage: false });
        });
        view.rerender({ scope: "alice", online: true });
      }
      await waitFor(() =>
        expect(
          mocks.request.mock.calls.some(([arg]) => arg.method === "POST"),
        ).toBe(true),
      );
      act(() => {
        view.result.current.setContent("newer");
        view.result.current.setContentLanguage(language);
      });
      if (pathway === "queue") view.rerender({ scope: "alice", online: false });
      await act(async () => {
        reply.resolve({ id: "created", content: "older", version: 7 });
        if (saving) await saving;
      });
      await waitFor(() =>
        expect(view.result.current.selectedId).toBe("created"),
      );
      expect(view.result.current.contentLanguage).toBe(language);
      expect(
        view.result.current.offlineDraftQueue["note:created"],
      ).toMatchObject({
        content: "newer",
        content_language: language,
        baseVersion: 7,
        implicitDefaultLanguage: false,
      });
      expect(view.result.current.isDirty).toBe(true);
    },
  );

  it.each(
    ["ordinary", "queue"].flatMap((pathway) =>
      [
        { id: "n", version: 2 },
        { id: "n", version: 2, content: "wrong" },
        { id: "other", version: 2, content: "newer" },
        { id: "n", content: "newer" },
        { id: "n", version: -1, content: "newer" },
      ].map((reply) => ({ pathway, reply })),
    ),
  )(
    "retains unknown-mode source after incomplete acknowledgement $pathway $reply",
    async ({ pathway, reply }) => {
      mocks.request.mockImplementation(({ method }) =>
        Promise.resolve(
          method === "PUT" ? reply : { id: "n", content: "older", version: 1 },
        ),
      );
      const view = renderEditor(pathway === "ordinary");
      await act(async () => {
        await view.result.current.loadDetail("n");
      });
      act(() => {
        view.result.current.setContent("newer");
        view.result.current.setIsDirty(true);
      });
      if (pathway === "ordinary")
        await act(async () => {
          expect(
            await view.result.current.saveNote({ showSuccessMessage: false }),
          ).toBe(false);
        });
      else {
        await act(async () => {
          await view.result.current.saveNote({ showSuccessMessage: false });
        });
        view.rerender({ scope: "alice", online: true });
        await waitFor(() =>
          expect(
            view.result.current.offlineDraftQueue["note:n"]?.syncState,
          ).toBe("error"),
        );
      }
      expect(view.result.current.contentLanguage).toBeUndefined();
      expect(view.result.current.offlineDraftQueue["note:n"].content).toBe(
        "newer",
      );
      expect(view.result.current.saveIndicator).not.toBe("saved");
    },
  );
  it.each(["   ", "😀".repeat(51), "😀".repeat(100)])(
    "retains raw opaque mode in setter and recovery %s",
    async (language) => {
      const view = renderEditor(false);
      act(() => {
        view.result.current.setContentLanguage(language);
        view.result.current.setContent("exact");
      });
      await act(async () => {
        await view.result.current.saveNote({ showSuccessMessage: false });
      });
      expect(view.result.current.contentLanguage).toBe(language);
      const restored = normalizeOfflineDraftQueue(
        JSON.parse(JSON.stringify(view.result.current.offlineDraftQueue)),
      );
      expect(restored["draft:new"].content_language).toBe(language);
    },
  );
  it("refuses raw opaque modes over 100 code points", () => {
    const view = renderEditor(false);
    expect(() =>
      view.result.current.setContentLanguage("😀".repeat(101)),
    ).toThrow("Invalid note language");
    expect(
      normalizeOfflineDraftQueue({
        "draft:new": { content: "exact", content_language: "😀".repeat(101) },
      })["draft:new"].content_language,
    ).toBeUndefined();
  });
});
