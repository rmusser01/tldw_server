import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import NotesManagerPage from "../NotesManagerPage";

const notesConnectionConfig = {
  serverUrl: "https://notes.example.test",
  authMode: "multi-user" as const,
  accessToken: "test-access-token",
};

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: notesConnectionConfig,
    loading: false,
    authorityLoading: false,
  }),
}));

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({ id: 1, is_active: true })),
  },
}));

vi.mock(
  "@/components/Notes/hooks/useNotesGraphAuthorityScope",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/components/Notes/hooks/useNotesGraphAuthorityScope")
      >();
    return {
      ...actual,
      useNotesGraphAuthorityScope: () =>
        actual.createNotesGraphAuthorityScope(
          notesConnectionConfig.serverUrl,
          1,
        ),
    };
  },
);

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  mockCapabilities,
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
    mockClearSetting: vi.fn(),
    mockCapabilities: { hasNotes: true },
  };
});

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string;
            [key: string]: unknown;
          },
    ) => {
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions;
      if (defaultValueOrOptions?.defaultValue)
        return defaultValueOrOptions.defaultValue;
      return key;
    },
  }),
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate,
}));

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest,
}));

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true,
}));

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false }),
}));

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: mockCapabilities,
    loading: false,
  }),
}));

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger,
}));

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: vi.fn(),
  }),
}));

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: vi.fn(async () => []),
  searchNoteKeywords: vi.fn(async () => []),
}));

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: Record<string, unknown>) => unknown,
  ) =>
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn(),
    }),
}));

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>();
  return {
    ...actual,
    getSetting: mockGetSetting,
    clearSetting: mockClearSetting,
  };
});

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null),
  },
}));

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />,
}));

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  queryClients.push(queryClient);
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>,
  );
};

const queryClients: QueryClient[] = [];

type NoteRequest = {
  path?: string;
  method?: string;
  body?: Record<string, unknown>;
};
const noteSnapshot = (
  request: NoteRequest,
  id: string | number = 11,
  version = 1,
) => ({
  id,
  title: request.body?.title,
  content: request.body?.content,
  content_language: request.body?.content_language ?? null,
  metadata: {
    keywords: [...((request.body?.keywords as string[] | undefined) ?? [])],
  },
  version,
});

const createCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || "");
    const method = String(request?.method || "GET").toUpperCase();
    return path === "/api/v1/notes/" && method === "POST";
  });

const updateCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || "");
    const method = String(request?.method || "GET").toUpperCase();
    return path === "/api/v1/notes/11" && method === "PUT";
  });

const callsFor = (path: string, method: string) =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const requestPath = String(request?.path || "");
    const requestMethod = String(request?.method || "GET").toUpperCase();
    return requestPath === path && requestMethod === method.toUpperCase();
  });

describe("NotesManagerPage stage 1 editor reliability", () => {
  beforeEach(() => {
    cleanup();
    vi.clearAllMocks();
    mockCapabilities.hasNotes = true;
    mockConfirmDanger.mockResolvedValue(true);
    mockGetSetting.mockResolvedValue(null);
    mockClearSetting.mockResolvedValue(undefined);

    let savedNote: ReturnType<typeof noteSnapshot> | null = null;
    mockBgRequest.mockImplementation(async (request: NoteRequest) => {
      const path = String(request.path || "");
      const method = String(request.method || "GET").toUpperCase();

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 },
        };
      }

      if (path === "/api/v1/notes/capabilities")
        return { content_language: true };

      if (path === "/api/v1/notes/" && method === "POST") {
        savedNote = noteSnapshot(request);
        return savedNote;
      }

      if (path === "/api/v1/notes/11" && method === "GET") return savedNote;

      if (path === "/api/v1/notes/11" && method === "PUT") {
        savedNote = noteSnapshot(request, 11, 2);
        return savedNote;
      }

      return {};
    });
  });

  afterEach(async () => {
    cleanup();
    while (queryClients.length > 0) {
      const queryClient = queryClients.pop();
      if (!queryClient) continue;
      await queryClient.cancelQueries();
      queryClient.clear();
    }
    vi.useRealTimers();
  });

  it("does not negotiate language while Notes has no active editor session", async () => {
    renderPage();
    await act(async () => {});
    expect(screen.getByTestId("notes-editor-empty-state")).toBeInTheDocument();
    expect(callsFor("/api/v1/notes/capabilities", "GET")).toHaveLength(0);
  });

  it("negotiates language immediately after Create note before typing", async () => {
    renderPage();
    await act(async () => {});
    expect(callsFor("/api/v1/notes/capabilities", "GET")).toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "Create note" }));
    await waitFor(() => {
      expect(callsFor("/api/v1/notes/capabilities", "GET")).toHaveLength(1);
      expect(screen.getByRole("combobox", { name: "Language" })).toBeEnabled();
    });
    expect(screen.getByPlaceholderText("Title")).toHaveValue("");
    expect(
      screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)",
      ),
    ).toHaveValue("");
    expect(createCalls()).toHaveLength(0);
  });

  it("saves with Ctrl+S and Cmd+S keyboard shortcuts", async () => {
    renderPage();

    const titleInput = screen.getByPlaceholderText("Title");
    const contentInput = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)",
    );

    fireEvent.change(titleInput, {
      target: { value: "Shortcut note" },
    });
    fireEvent.change(contentInput, {
      target: { value: "Saved from keyboard" },
    });

    fireEvent.focus(contentInput);
    fireEvent.keyDown(contentInput, { key: "s", ctrlKey: true });

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1);
    });

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note created");
      expect(
        callsFor("/api/v1/notes/11", "GET").filter(
          ([request]) => request.servicePromptConfig,
        ),
      ).toHaveLength(1);
      expect(screen.getByTestId("notes-save-status")).not.toHaveAttribute(
        "data-state",
        "saving",
      );
    });

    fireEvent.change(contentInput, {
      target: { value: "Updated via cmd shortcut" },
    });

    fireEvent.focus(titleInput);
    fireEvent.keyDown(titleInput, { key: "s", metaKey: true });

    await waitFor(() => {
      expect(updateCalls()).toHaveLength(1);
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note updated");
    });
  });

  it("saves a draft and immediately opens a blank focused draft for repeated capture", async () => {
    renderPage();

    const titleInput = screen.getByPlaceholderText("Title");
    const contentInput = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)",
    );

    fireEvent.change(titleInput, {
      target: { value: "First quick capture" },
    });
    fireEvent.change(contentInput, {
      target: { value: "Capture this and keep moving." },
    });

    fireEvent.click(screen.getByTestId("notes-save-and-new-button"));

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1);
    });

    await waitFor(() => {
      expect(titleInput).toHaveValue("");
      expect(contentInput).toHaveValue("");
    });
    expect(titleInput).toHaveFocus();
  });

  it("keeps the current draft in place when save and new fails", async () => {
    mockBgRequest.mockImplementation(async (request: NoteRequest) => {
      const path = String(request.path || "");
      const method = String(request.method || "GET").toUpperCase();

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 },
        };
      }

      if (path === "/api/v1/notes/" && method === "POST") {
        throw new Error("Save failed");
      }

      return {};
    });

    renderPage();

    const titleInput = screen.getByPlaceholderText("Title");
    const contentInput = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)",
    );

    fireEvent.change(titleInput, {
      target: { value: "Do not lose this" },
    });
    fireEvent.change(contentInput, {
      target: { value: "The save failed, so this text must remain." },
    });

    fireEvent.click(screen.getByTestId("notes-save-and-new-button"));

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1);
      // One surface for the failure (NS-03): the issue panel, not a toast too.
      expect(screen.getByTestId("notes-save-issue")).toBeInTheDocument();
    });
    expect(mockMessageError).not.toHaveBeenCalled();

    expect(titleInput).toHaveValue("Do not lose this");
    expect(contentInput).toHaveValue(
      "The save failed, so this text must remain.",
    );
  });

  it("does not save when Ctrl/Cmd+S is pressed outside the editor region", async () => {
    renderPage();

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Shortcut note" },
    });
    fireEvent.change(
      screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)",
      ),
      {
        target: { value: "Saved from keyboard" },
      },
    );

    const searchInput = screen.getByPlaceholderText(
      "Search notes... (use quotes for exact match)",
    );
    fireEvent.focus(searchInput);
    fireEvent.keyDown(searchInput, { key: "s", ctrlKey: true });
    fireEvent.keyDown(searchInput, { key: "s", metaKey: true });

    await waitFor(() => {
      expect(createCalls()).toHaveLength(0);
    });
  });

  it("preserves draft content and exposes retry after save failure", async () => {
    let createAttempts = 0;
    mockBgRequest.mockImplementation(async (request: NoteRequest) => {
      const path = String(request.path || "");
      const method = String(request.method || "GET").toUpperCase();

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 },
        };
      }

      if (path === "/api/v1/notes/" && method === "POST") {
        createAttempts += 1;
        if (createAttempts === 1) {
          throw new Error("Server temporarily unavailable");
        }
        return noteSnapshot(request);
      }

      if (path === "/api/v1/notes/11" && method === "GET") {
        return {
          id: 11,
          title: "Recoverable note",
          content: "Draft survives failure",
          content_language: null,
          metadata: { keywords: [] },
          version: 1,
        };
      }

      return {};
    });

    renderPage();

    const titleInput = screen.getByPlaceholderText("Title");
    const contentInput = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)",
    );

    fireEvent.change(titleInput, {
      target: { value: "Recoverable note" },
    });
    fireEvent.change(contentInput, {
      target: { value: "Draft survives failure" },
    });

    fireEvent.click(screen.getByTestId("notes-save-button"));

    // One surface for the failure (NS-03): the issue panel, not a toast too.
    const issue = await screen.findByTestId("notes-save-issue");
    expect(issue).toHaveAttribute("data-kind", "retrying");
    expect(mockMessageError).not.toHaveBeenCalled();

    expect(titleInput).toHaveValue("Recoverable note");
    expect(contentInput).toHaveValue("Draft survives failure");
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute(
      "data-state",
      "retrying",
    );

    fireEvent.click(screen.getByTestId("notes-save-retry"));

    await waitFor(() => {
      expect(createAttempts).toBe(2);
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note created");
    });
  });

  it("shows saving state and blocks duplicate submits while save is in flight", async () => {
    let resolveCreate: ((value: unknown) => void) | null = null;
    const createPromise = new Promise((resolve) => {
      resolveCreate = resolve;
    });

    mockBgRequest.mockImplementation(async (request: NoteRequest) => {
      const path = String(request.path || "");
      const method = String(request.method || "GET").toUpperCase();

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 },
        };
      }

      if (path === "/api/v1/notes/" && method === "POST") {
        const acknowledged = noteSnapshot(request);
        return createPromise.then(() => acknowledged);
      }

      if (path === "/api/v1/notes/11" && method === "GET") {
        return {
          id: 11,
          title: "Slow save",
          content: "Saving should be visible",
          content_language: null,
          metadata: { keywords: [] },
          version: 1,
        };
      }

      return {};
    });

    renderPage();

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Slow save" },
    });
    fireEvent.change(
      screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)",
      ),
      {
        target: { value: "Saving should be visible" },
      },
    );

    const saveButton = screen.getByTestId("notes-save-button");
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1);
    });
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute(
      "data-state",
      "saving",
    );

    fireEvent.click(saveButton);
    expect(createCalls()).toHaveLength(1);

    fireEvent.change(
      screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)",
      ),
      {
        target: { value: "Newer exact draft remains" },
      },
    );
    resolveCreate?.(undefined);

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note created");
      expect(
        screen.getByPlaceholderText(
          "Write your note here... (Markdown supported)",
        ),
      ).toHaveValue("Newer exact draft remains");
      expect(screen.getByTestId("notes-save-status")).toHaveAttribute(
        "data-state",
        "dirty",
      );
    });
    expect(createCalls()).toHaveLength(1);
    expect(createCalls()[0][0].body.content).toBe("Saving should be visible");
  });

  it("opens a writable, focused title field after starting a new draft from the empty state", async () => {
    renderPage();

    expect(screen.getByTestId("notes-editor-empty-state")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Create note" }));

    await waitFor(() => {
      expect(
        screen.queryByTestId("notes-editor-empty-state"),
      ).not.toBeInTheDocument();
    });

    const titleInput = screen.getByRole("textbox", { name: "Note title" });
    expect(titleInput).toBeEnabled();
    expect(screen.getByTestId("notes-save-button")).toBeDisabled();

    await waitFor(() => {
      expect(titleInput).toHaveFocus();
    });
  });

  it("keeps the empty-state create CTA disabled when the notes capability is unavailable", async () => {
    mockCapabilities.hasNotes = false;

    renderPage();

    const createButton = screen.getByTestId("notes-editor-empty-create");
    expect(createButton).toBeDisabled();

    fireEvent.click(createButton);

    await waitFor(() => {
      expect(createCalls()).toHaveLength(0);
    });
    expect(screen.getByTestId("notes-editor-empty-state")).toBeInTheDocument();
  });

  it("encodes reserved note ids for save refresh requests", async () => {
    const reservedId = "folder/note?image#1";
    const rawPath = `/api/v1/notes/${reservedId}`;
    const encodedPath = `/api/v1/notes/${encodeURIComponent(reservedId)}`;
    let detailLoadCount = 0;
    let savedNote: ReturnType<typeof noteSnapshot> | null = null;

    mockBgRequest.mockImplementation(async (request: NoteRequest) => {
      const path = String(request.path || "");
      const method = String(request.method || "GET").toUpperCase();

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 },
        };
      }

      if (path === "/api/v1/notes/capabilities")
        return { content_language: true };

      if (path === "/api/v1/notes/" && method === "POST") {
        savedNote = noteSnapshot(request, reservedId, 1);
        return savedNote;
      }

      if ((path === rawPath || path === encodedPath) && method === "GET") {
        detailLoadCount += 1;
        if (detailLoadCount === 1) {
          const { version: _version, ...withoutVersion } = savedNote!;
          return withoutVersion;
        }
        if (detailLoadCount === 2) savedNote = { ...savedNote!, version: 2 };
        return savedNote;
      }

      if (path === encodedPath && method === "PUT") {
        savedNote = noteSnapshot(request, reservedId, 3);
        return savedNote;
      }

      return {};
    });

    renderPage();

    const titleInput = screen.getByPlaceholderText("Title");
    const contentInput = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)",
    );

    fireEvent.change(titleInput, {
      target: { value: "Reserved route note" },
    });
    fireEvent.change(contentInput, {
      target: { value: "Saved first pass" },
    });

    fireEvent.focus(contentInput);
    fireEvent.keyDown(contentInput, { key: "s", ctrlKey: true });

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1);
      expect(
        callsFor(encodedPath, "GET").filter(
          ([request]) => request.servicePromptConfig,
        ),
      ).toHaveLength(1);
      expect(screen.getByTestId("notes-save-status")).not.toHaveAttribute(
        "data-state",
        "saving",
      );
    });

    fireEvent.change(contentInput, {
      target: { value: "Updated reserved route note" },
    });

    fireEvent.focus(titleInput);
    fireEvent.keyDown(titleInput, { key: "s", metaKey: true });

    await waitFor(() => {
      expect(callsFor(encodedPath, "PUT")).toHaveLength(1);
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note updated");
    });

    expect(callsFor(rawPath, "GET")).toHaveLength(0);
    await waitFor(() => {
      expect(
        callsFor(encodedPath, "GET").map(([request]) =>
          Boolean(request.servicePromptConfig),
        ),
      ).toEqual([true, true, false]);
    });
    expect(callsFor(encodedPath, "PUT")[0]?.[0]?.headers).toMatchObject({
      "expected-version": "2",
    });
  });
});
