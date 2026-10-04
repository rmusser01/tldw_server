/**
 * NL-02 (#3103): "Export matching notes" must page the way the server does.
 *
 * The bgRequest mock mirrors list_notes in
 * tldw_Server_API/app/api/v1/endpoints/notes.py: GET /api/v1/notes/ reads only
 * `limit` (default 100, max 1000) and `offset`, ignores unknown params such as
 * page/results_per_page, and reports the library size as `pagination.total`.
 */
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useNotesExport, type UseNotesExportDeps } from "../useNotesExport";
import { bgRequest } from "@/services/background-proxy";

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }));

type ServerNote = { id: number; title: string; content: string };

const makeNotes = (count: number): ServerNote[] =>
  Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    title: `Note ${index + 1}`,
    content: `Body ${index + 1}`,
  }));

type ServeOptions = {
  /** Drop every total/pagination field, like an older or proxied server. */
  omitTotal?: boolean;
  /** Always serve the first slice, like a server that ignores the offset. */
  ignoreOffset?: boolean;
};

const serveNotesList = (
  path: string,
  notes: ServerNote[],
  { omitTotal = false, ignoreOffset = false }: ServeOptions = {},
) => {
  const params = new URL(path, "https://notes.test").searchParams;
  const limit = Math.min(Number(params.get("limit") ?? 100), 1000);
  const offset = ignoreOffset ? 0 : Number(params.get("offset") ?? 0);
  const page = notes.slice(offset, offset + limit);
  if (omitTotal) {
    return { items: page, notes: page, results: page, count: page.length };
  }
  const hasMore = offset + page.length < notes.length;
  return {
    notes: page,
    items: page,
    results: page,
    count: page.length,
    limit,
    offset,
    total: notes.length,
    pagination: {
      limit,
      offset,
      total: notes.length,
      has_more: hasMore,
      next_offset: hasMore ? offset + limit : null,
    },
  };
};

const listPaths: string[] = [];
let exportedBlobs: Blob[] = [];

const installServer = (notes: ServerNote[], options: ServeOptions = {}) => {
  vi.mocked(bgRequest).mockImplementation(async ({ path }) => {
    listPaths.push(String(path));
    // Test-only circuit breaker so a runaway client cannot hang the suite.
    if (listPaths.length > 50) throw new Error("runaway export pagination");
    return serveNotesList(String(path), notes, options);
  });
};

const exportedJson = async () =>
  JSON.parse(await exportedBlobs[0].text()) as Array<{ id: number }>;

const deps = (overrides: Partial<UseNotesExportDeps> = {}): UseNotesExportDeps => ({
  message: {
    info: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
  } as unknown as UseNotesExportDeps["message"],
  confirmDanger: vi.fn(async () => true),
  t: (key) => key,
  listMode: "active",
  query: "",
  effectiveKeywordTokens: [],
  total: 0,
  filteredCount: 0,
  hasActiveFilters: false,
  selectedBulkNotes: [],
  fetchFilteredNotesRaw: vi.fn(),
  selectedId: null,
  title: "",
  content: "",
  editorKeywords: [],
  ...overrides,
});

beforeEach(() => {
  listPaths.length = 0;
  exportedBlobs = [];
  vi.mocked(bgRequest).mockReset();
  vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
    exportedBlobs.push(blob as Blob);
    return "blob:notes";
  });
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
});
afterEach(() => vi.restoreAllMocks());

describe("useNotesExport paging (NL-02)", () => {
  it("exports 500 notes as 500 unique notes in ceil(500/limit) limit/offset requests", async () => {
    installServer(makeNotes(500));
    const options = deps({ total: 500 });
    const { result } = renderHook(() => useNotesExport(options));

    await act(async () => {
      await result.current.exportAllJSON();
    });

    const params = listPaths.map((path) => new URL(path, "https://notes.test").searchParams);
    const limit = Number(params[0].get("limit"));
    expect(limit).toBeGreaterThan(0);
    expect(listPaths).toHaveLength(Math.ceil(500 / limit));
    expect(params.map((query) => Number(query.get("offset")))).toEqual(
      Array.from({ length: Math.ceil(500 / limit) }, (_, index) => index * limit),
    );
    const exported = await exportedJson();
    expect(exported).toHaveLength(500);
    expect(new Set(exported.map((note) => note.id)).size).toBe(500);
    expect(options.message.warning).not.toHaveBeenCalled();
    expect(result.current.exportProgress).toBeNull();
  });

  it("reports determinate progress and Cancel stops further requests without a download", async () => {
    const notes = makeNotes(500);
    let secondPageSignal: AbortSignal | undefined;
    let releaseSecondPage: (() => void) | undefined;
    vi.mocked(bgRequest).mockImplementation(async ({ path, abortSignal }) => {
      listPaths.push(String(path));
      if (listPaths.length === 2) {
        secondPageSignal = abortSignal;
        await new Promise<void>((resolve, reject) => {
          releaseSecondPage = resolve;
          abortSignal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          );
        });
      }
      return serveNotesList(String(path), notes);
    });
    const options = deps({ total: 500 });
    const { result } = renderHook(() => useNotesExport(options));

    let exportPromise: Promise<void> | undefined;
    act(() => {
      exportPromise = result.current.exportAllJSON();
    });
    await waitFor(() => expect(listPaths).toHaveLength(2));
    expect(result.current.exportProgress).toMatchObject({
      format: "json",
      fetchedNotes: 100,
      totalNotes: 500,
    });

    act(() => {
      result.current.cancelExport();
    });
    releaseSecondPage?.();
    await act(async () => {
      await exportPromise;
    });

    expect(secondPageSignal?.aborted).toBe(true);
    expect(listPaths).toHaveLength(2);
    expect(exportedBlobs).toHaveLength(0);
    expect(options.message.success).not.toHaveBeenCalled();
    expect(options.message.error).not.toHaveBeenCalled();
    expect(options.message.info).toHaveBeenCalledWith(expect.stringMatching(/cancel/i));
    expect(result.current.exportProgress).toBeNull();
  });

  it("terminates on a short page when the response has no total", async () => {
    installServer(makeNotes(250), { omitTotal: true });
    const { result } = renderHook(() => useNotesExport(deps()));

    await act(async () => {
      await result.current.exportAllJSON();
    });

    expect(listPaths).toHaveLength(3);
    const exported = await exportedJson();
    expect(new Set(exported.map((note) => note.id)).size).toBe(250);
    expect(exported).toHaveLength(250);
  });

  it("terminates on an empty page when the response has no total and the size is a page multiple", async () => {
    installServer(makeNotes(200), { omitTotal: true });
    const { result } = renderHook(() => useNotesExport(deps()));

    await act(async () => {
      await result.current.exportAllJSON();
    });

    expect(listPaths).toHaveLength(3);
    expect(await exportedJson()).toHaveLength(200);
  });

  it("stops when a page adds no new notes and warns that the export is incomplete", async () => {
    installServer(makeNotes(250), { ignoreOffset: true });
    const options = deps({ total: 250 });
    const { result } = renderHook(() => useNotesExport(options));

    await act(async () => {
      await result.current.exportAllJSON();
    });

    expect(listPaths).toHaveLength(2);
    const exported = await exportedJson();
    expect(exported).toHaveLength(100);
    expect(new Set(exported.map((note) => note.id)).size).toBe(100);
    expect(options.message.warning).toHaveBeenCalledWith(
      expect.stringContaining("100 of 250"),
    );
  });

  it("stops a filtered export at the reported total", async () => {
    const matches = makeNotes(200);
    const fetchFilteredNotesRaw = vi.fn(
      async (_q: string, _toks: string[], page: number, pageSize: number) => ({
        items: matches.slice((page - 1) * pageSize, page * pageSize),
        total: matches.length,
      }),
    );
    const options = deps({
      query: "Body",
      hasActiveFilters: true,
      total: 200,
      filteredCount: 200,
      fetchFilteredNotesRaw,
    });
    const { result } = renderHook(() => useNotesExport(options));

    await act(async () => {
      await result.current.exportAllJSON();
    });

    expect(fetchFilteredNotesRaw).toHaveBeenCalledTimes(2);
    const exported = await exportedJson();
    expect(exported).toHaveLength(200);
    expect(new Set(exported.map((note) => note.id)).size).toBe(200);
  });
});
