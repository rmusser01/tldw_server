import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import { useNotesImport } from "../hooks/useNotesImport";
import {
  readKnowledgeNoteProvenance,
  retainKnowledgeNoteProvenance,
} from "@/utils/knowledge-note-provenance";
import { writeFileSync, mkdirSync } from "node:fs";


const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockSetSetting: vi.fn(),
  mockClearSetting: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}));

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: {
      serverUrl: "https://notes.test",
      authMode: "single-user",
      apiKey: "synthetic",
    },
    loading: false,
    authorityLoading: false,
  }),
}));
vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", () => ({
  useNotesGraphAuthorityScope: () => "synthetic-authority",
  createNotesGraphAuthorityScope: () => "synthetic-authority",
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: mockMessageInfo
  })
}))

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: vi.fn(async () => []),
  searchNoteKeywords: vi.fn(async () => [])
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn()
    })
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mockGetSetting,
    setSetting: mockSetSetting,
    clearSetting: mockClearSetting
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: { onSelectNote: (id: string) => void }) => (
    <button data-testid="notes-list-panel" onClick={() => onSelectNote("n")}>
      Load
    </button>
  ),
}));

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: ({ onCopy, onExport }: any) => (
    <div>
      <button
        data-testid="copy-content-action"
        onClick={() => onCopy("content")}
      >
        Copy content
      </button>
      <button
        data-testid="copy-markdown-action"
        onClick={() => onCopy("markdown")}
      >
        Copy markdown
      </button>
      <button
        data-testid="export-source-action"
        onClick={() => onExport("source")}
      >
        Source
      </button>
      <button data-testid="export-md-action" onClick={() => onExport("md")}>
        Export md
      </button>
      <button data-testid="export-json-action" onClick={() => onExport("json")}>
        Export json
      </button>
    </div>
  ),
}));

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

describe("NotesManagerPage stage 31 single-note export/copy parity", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear();
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      return {}
    })
  })

  it("routes copy modes to clipboard payloads and routes single-note export formats", async () => {
    const clipboardWriteText = vi.fn(async () => undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: clipboardWriteText
      }
    })

    const exportedBlobs: Blob[] = []
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn((blob: Blob) => {
        exportedBlobs.push(blob)
        return `blob:notes-${exportedBlobs.length}`
      })
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn()
    })
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined)

    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Export parity note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Single note body" }
    })

    fireEvent.click(screen.getByTestId("copy-content-action"))
    fireEvent.click(screen.getByTestId("copy-markdown-action"))

    await waitFor(() => {
      expect(clipboardWriteText).toHaveBeenCalledTimes(2)
    })
    const clipboardCalls = clipboardWriteText.mock.calls as unknown as Array<[string]>
    expect(clipboardCalls[0]?.[0]).toBe("Single note body")
    expect(String(clipboardCalls[1]?.[0])).toContain("# Export parity note")
    expect(String(clipboardCalls[1]?.[0])).toContain("Single note body")

    fireEvent.click(screen.getByTestId("export-md-action"))
    fireEvent.click(screen.getByTestId("export-json-action"))

    await waitFor(() => {
      expect(exportedBlobs.length).toBe(2)
    })
    expect(exportedBlobs[0].type).toContain("text/markdown")
    expect(exportedBlobs[1].type).toContain("application/json")
    expect(await exportedBlobs[0].text()).toContain("# Export parity note")
    expect(await exportedBlobs[1].text()).toContain('"title": "Export parity note"')
  })
  it.each([
    "active",
    "legacy",
    "disagreement",
    "deleted",
    "unsupported",
    "json",
    "future-mode",
    "json-deleted",
    "json-unsupported",
  ])(
    "preserves %s provenance through actual capture, actions and import preparation",
    async (kind) => {
      const evidence = {
        origin: "knowledge_qa",
        trust_state: "uncited_degraded_answer",
        thread_id: "canonical",
      };
      const legacy = { ...evidence, thread_id: "legacy" };
      const structured = kind.startsWith("json") || kind === "future-mode";
      const deleted = kind.endsWith("deleted"),
        unsupported = kind.endsWith("unsupported");
      const body = structured
        ? ' {"n":9007199254740993,"s":"<>&"}\r\n'
        : "Body";
      const markerBody = retainKnowledgeNoteProvenance(body, legacy);
      const record = {
        id: "n",
        title: "Evidence",
        content: structured || kind === "active" ? body : markerBody,
        content_language: structured
          ? kind.startsWith("json")
            ? "json"
            : kind
          : null,
        version: 1,
        ...(kind === "legacy"
          ? {}
          : {
              knowledge_provenance_state: deleted
                ? "deleted"
                : unsupported
                  ? "unsupported"
                  : "active",
              knowledge_provenance_version: 3,
              knowledge_provenance: deleted || unsupported ? null : evidence,
            }),
      };
      mockBgRequest.mockImplementation(
        async ({ path, method }: { path: string; method?: string }) => {
          if (path === "/api/v1/notes/n") return record;
          if (path.endsWith("/capabilities")) return { content_language: true };
          if (path === "/api/v1/notes/import" && method === "POST")
            return { created_count: 1 };
          return {
            items: [],
            notes: [],
            tasks: [],
            events: [],
            pagination: { total_items: 0, total_pages: 1 },
          };
        },
      );
      const blobs: Blob[] = [];
      Object.defineProperty(URL, "createObjectURL", {
        configurable: true,
        value: vi.fn((blob: Blob) => {
          blobs.push(blob);
          return "blob:export";
        }),
      });
      Object.defineProperty(URL, "revokeObjectURL", {
        configurable: true,
        value: vi.fn(),
      });
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(() => {});
      const page = renderPage();
      fireEvent.click(screen.getByTestId("notes-list-panel"));
      await waitFor(() =>
        expect(screen.getByPlaceholderText("Title")).toHaveValue("Evidence"),
      );
      const expected =
        deleted || (structured && unsupported)
          ? null
          : kind === "legacy" || kind === "unsupported"
            ? legacy
            : evidence;
      const importer = renderHook(() =>
        useNotesImport({
          isOnline: true,
          listMode: "active",
          message: {
            error: mockMessageError,
            success: mockMessageSuccess,
          } as unknown as Parameters<typeof useNotesImport>[0]["message"],
          t: (key: string) => key,
          refetch: vi.fn(),
        }),
      );
      for (const format of structured
        ? ["json", "source"]
        : ["md", "json", "source"]) {
        fireEvent.click(screen.getByTestId(`export-${format}-action`));
        const blob = blobs.at(-1)!;
        expect(blob).toBeDefined();
        const text = await blob.text();
        if (format === "source") {
          expect(text).toBe(body);
          continue;
        }
        const portable = format === "json" ? JSON.parse(text) : null;
        if (portable && !structured) {
          expect(portable).not.toHaveProperty("knowledge_provenance");
          expect(portable).not.toHaveProperty("expected_provenance_version");
        }
        if (structured) {
          expect(portable.content).toBe(body);
          expect(portable.knowledge_provenance ?? null).toEqual(expected);
        } else
          expect(
            readKnowledgeNoteProvenance(portable ? portable.content : text),
          ).toEqual(expected);
        const filename =
          format === "json" ? "Evidence.note.json" : "Evidence.md";
        await act(async () => {
          await importer.result.current.prepareImportFiles([
            {
              name: filename,
              arrayBuffer: async () => new TextEncoder().encode(text).buffer,
            } as File,
          ]);
        });
        await act(async () => {
          await importer.result.current.confirmImport();
        });
        const submitted = mockBgRequest.mock.calls
          .filter(([x]) => x.path === "/api/v1/notes/import")
          .at(-1)![0].body;
        expect(submitted.items[0].content).toBe(text);
        if (process.env.REVIEW_EXPORT_ARTIFACT_DIR) {
          mkdirSync(process.env.REVIEW_EXPORT_ARTIFACT_DIR, {
            recursive: true,
          });
          writeFileSync(
            `${process.env.REVIEW_EXPORT_ARTIFACT_DIR}/${kind}-${format}.json`,
            JSON.stringify({
              request: submitted,
              expected,
              source: body,
              structured,
            }),
          );
        }
      }
      importer.unmount();
      page.unmount();
      click.mockRestore();
    },
  );

})
