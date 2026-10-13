import type { MessageInstance } from "antd/es/message/interface";
import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useNotesImport } from "../hooks/useNotesImport";
import { useNotesExport } from "../hooks/useNotesExport";

const { bgRequest } = vi.hoisted(() => ({ bgRequest: vi.fn() }));
vi.mock("@/services/background-proxy", () => ({ bgRequest }));
const source = '\ufeff  {"x":9007199254740993,"huge":1e400,"html":"<>&"}\r\n ';
const message = {
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn(),
} as unknown as MessageInstance;
const file = (name: string, text: string) =>
  ({
    name,
    arrayBuffer: async () => new TextEncoder().encode(text).buffer,
  }) as File;
const exportDeps = () => ({
  message,
  confirmDanger: vi.fn(async () => true),
  t: (_: string, opts?: Record<string, unknown>) =>
    typeof opts?.defaultValue === "string" ? opts.defaultValue : "Exported",
  listMode: "active" as const,
  query: "",
  effectiveKeywordTokens: [],
  total: 1,
  filteredCount: 1,
  hasActiveFilters: false,
  selectedBulkNotes: [],
  fetchFilteredNotesRaw: vi.fn(),
  selectedId: "n-1",
  title: "Data",
  content: source,
  contentLanguage: "json",
  editorKeywords: [],
});

beforeEach(() => {
  vi.clearAllMocks();
});

describe("source and portable import contracts", () => {
  it("loads and submits exact raw UTF-8 source with explicit filename context", async () => {
    bgRequest.mockResolvedValue({ created_count: 1 });
    const { result } = renderHook(() =>
      useNotesImport({
        isOnline: true,
        listMode: "active",
        message,
        t: (key) => key,
        refetch: vi.fn(),
      }),
    );
    await act(async () => {
      await result.current.prepareImportFiles(
        [file("Data.json", source)],
        true,
      );
    });
    expect(result.current.pendingImportFiles[0]).toMatchObject({
      format: "source",
      content: source,
      parseError: null,
    });
    await act(async () => {
      await result.current.confirmImport();
    });
    expect(bgRequest.mock.calls[0][0].body.items[0]).toEqual({
      file_name: "Data.json",
      format: "source",
      content: source,
    });
  });

  it("keeps only the latest import choice and blocks submit during re-reading", async () => {
    const { result } = renderHook(() =>
      useNotesImport({
        isOnline: true,
        listMode: "active",
        message,
        t: (key) => key,
        refetch: vi.fn(),
      }),
    );
    const text = '\ufeff{"title":"Container","content":"literal"}';
    const bytes = new TextEncoder().encode(text).buffer;
    let releaseSource!: (value: ArrayBuffer) => void;
    let releaseContainer!: (value: ArrayBuffer) => void;
    const arrayBuffer = vi
      .fn()
      .mockResolvedValueOnce(bytes)
      .mockImplementationOnce(
        () =>
          new Promise<ArrayBuffer>((resolve) => {
            releaseSource = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise<ArrayBuffer>((resolve) => {
            releaseContainer = resolve;
          }),
      );
    await act(async () => {
      await result.current.prepareImportFiles([
        { name: "Data.json", arrayBuffer } as unknown as File,
      ]);
    });
    let sourceRead!: Promise<void>;
    let containerRead!: Promise<void>;
    act(() => {
      sourceRead = result.current.changeImportSourceMode(true);
    });
    expect(result.current.importPreparing).toBe(true);
    await act(async () => {
      await result.current.confirmImport();
    });
    expect(bgRequest).not.toHaveBeenCalled();
    act(() => {
      containerRead = result.current.changeImportSourceMode(false);
    });
    await act(async () => {
      releaseContainer(bytes);
      await containerRead;
    });
    expect(result.current.pendingImportFiles[0].format).toBe("json");
    await act(async () => {
      releaseSource(bytes);
      await sourceRead;
    });
    expect(result.current.importSourceMode).toBe(false);
    expect(result.current.pendingImportFiles[0]).toMatchObject({
      format: "json",
      content: text.slice(1),
    });
    expect(result.current.importPreparing).toBe(false);
  });

  it("retains the legacy BOM-bearing JSON container reader", async () => {
    const { result } = renderHook(() =>
      useNotesImport({
        isOnline: true,
        listMode: "active",
        message,
        t: (key) => key,
        refetch: vi.fn(),
      }),
    );
    await act(async () => {
      await result.current.prepareImportFiles([
        file("notes.json", '\ufeff[{"title":"Legacy","content":"body"}]'),
      ]);
    });
    expect(result.current.pendingImportFiles[0]).toMatchObject({
      format: "json",
      detectedNotes: 1,
      parseError: null,
    });
  });

  it("declares portable contracts inside old wrappers and preserves unsupported-server errors", async () => {
    const envelope = JSON.stringify({
      notes: [
        {
          format_version: "tldw.note.v99",
          title: "Data",
          content: source,
          content_language: "future",
        },
      ],
    });
    bgRequest.mockRejectedValue(new Error("Unsupported import format"));
    const { result } = renderHook(() =>
      useNotesImport({
        isOnline: true,
        listMode: "active",
        message,
        t: (key) => key,
        refetch: vi.fn(),
      }),
    );
    await act(async () => {
      await result.current.prepareImportFiles([
        file("portable.json", envelope),
      ]);
    });
    expect(result.current.pendingImportFiles[0].format).toBe("portable_note");
    await act(async () => {
      await result.current.confirmImport();
    });
    expect(bgRequest.mock.calls[0][0].body.items[0].format).toBe(
      "portable_note",
    );
    expect(message.error).toHaveBeenCalledWith("Unsupported import format");
    expect(result.current.importModalOpen).toBe(true);
  });

  it.each(["notes", "data", "items", "results"])(
    "previews the root envelope rather than its %s extra field",
    async (wrapper) => {
      const outer = {
        format_version: "tldw.note.v1",
        title: "Outer",
        content: source,
        content_language: "yaml",
        [wrapper]: [
          { format_version: "tldw.note.v99" },
          { format_version: "tldw.note.v99" },
        ],
      };
      const serialized = JSON.stringify(outer);
      const { result } = renderHook(() =>
        useNotesImport({
          isOnline: true,
          listMode: "active",
          message,
          t: (key) => key,
          refetch: vi.fn(),
        }),
      );
      await act(async () => {
        await result.current.prepareImportFiles([
          file("outer.json", serialized),
        ]);
      });
      expect(result.current.pendingImportFiles[0]).toMatchObject({
        format: "portable_note",
        content: serialized,
        detectedNotes: 1,
        parseError: null,
      });
    },
  );

  it("blocks unreadable UTF-8 without substituting source", async () => {
    const { result } = renderHook(() =>
      useNotesImport({
        isOnline: true,
        listMode: "active",
        message,
        t: (key) => key,
        refetch: vi.fn(),
      }),
    );
    await act(async () => {
      await result.current.prepareImportFiles(
        [
          {
            name: "Data.json",
            arrayBuffer: async () => Uint8Array.of(0xff).buffer,
          } as File,
        ],
        true,
      );
    });
    expect(result.current.pendingImportFiles[0].parseError).toMatch(/UTF-8/);
    await act(async () => {
      await result.current.confirmImport();
    });
    expect(bgRequest).not.toHaveBeenCalled();
  });
});

describe("complete source export callers", () => {
  it("refuses a truncated list response instead of fabricating a portable body", async () => {
    bgRequest.mockResolvedValue({
      notes: [
        {
          id: "n-1",
          title: "Data",
          content_preview: "truncated…",
          content_language: "json",
        },
      ],
    });
    const { result } = renderHook(() => useNotesExport(exportDeps()));
    await act(async () => {
      await result.current.exportAllJSON();
    });
    expect(message.error).toHaveBeenCalledWith(
      "Portable export requires complete source records",
    );
    expect(message.success).not.toHaveBeenCalled();
  });

  it("fetches full pages using the canonical list contract and retains metadata", async () => {
    const blobs: Blob[] = [];
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn((blob) => {
        blobs.push(blob);
        return "blob:note";
      }),
      revokeObjectURL: vi.fn(),
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    const first = Array.from({ length: 100 }, (_, id) => ({
      id: `n-${id}`,
      title: "Data",
      content: source,
      content_language: "json",
      keywords: ["data"],
      created_at: "2026-09-01",
      last_modified: "2026-09-02",
    }));
    const final = { ...first[0], id: "n-100", content_language: "future/mode" };
    bgRequest
      .mockResolvedValueOnce({ notes: first, pagination: { total_pages: 2 } })
      .mockResolvedValueOnce({
        notes: [final],
        pagination: { total_pages: 2 },
      });
    const { result } = renderHook(() => useNotesExport(exportDeps()));
    await act(async () => {
      await result.current.exportAllJSON();
    });
    expect(bgRequest.mock.calls.map(([request]) => request.path)).toEqual([
      "/api/v1/notes/?limit=100&offset=0&include_keywords=true&sort_by=created_at&sort_order=asc",
      "/api/v1/notes/?limit=100&offset=100&include_keywords=true&sort_by=created_at&sort_order=asc",
    ]);
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(blobs[0]);
    });
    const records = JSON.parse(text);
    expect(records).toHaveLength(101);
    expect(records[100]).toMatchObject({
      id: "n-100",
      content: source,
      content_language: "future/mode",
      keywords: ["data"],
      created_at: "2026-09-01",
      updated_at: "2026-09-02",
    });
    click.mockRestore();
    vi.unstubAllGlobals();
  });

  it("refuses portable export for an undefined editor mode", () => {
    const { result } = renderHook(() =>
      useNotesExport({ ...exportDeps(), contentLanguage: undefined }),
    );
    act(() => {
      result.current.exportSelected("json");
    });
    expect(message.error).toHaveBeenCalledWith(
      "Portable export requires known language metadata",
    );
  });

  it("downloads current source and mode without numeric or whitespace conversion", () => {
    const blobs: Blob[] = [];
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn((blob) => {
        blobs.push(blob);
        return "blob:note";
      }),
      revokeObjectURL: vi.fn(),
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    const { result } = renderHook(() => useNotesExport(exportDeps()));
    act(() => {
      result.current.exportSelected("source");
    });
    expect(blobs[0].size).toBe(new TextEncoder().encode(source).length);
    expect(click.mock.instances[0].download).toBe("Data.json");
    expect(blobs[0].type).toBe("text/plain;charset=utf-8");
    click.mockRestore();
    vi.unstubAllGlobals();
  });
});
