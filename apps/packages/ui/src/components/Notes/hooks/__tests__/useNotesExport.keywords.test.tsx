import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useNotesExport, type UseNotesExportDeps } from "../useNotesExport";
import { bgRequest } from "@/services/background-proxy";

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }));

let exported: Blob;
// Larger than one export page, so tags must survive every page.
const LIBRARY_SIZE = 150;
beforeEach(() => {
  vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
    exported = blob as Blob;
    return "blob:notes";
  });
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  // Mirrors list_notes: limit/offset paging, pagination.total, and keywords
  // only when include_keywords=true.
  vi.mocked(bgRequest).mockImplementation(async ({ path }) => {
    const query = new URL(String(path), "https://notes.test").searchParams;
    const limit = Number(query.get("limit") ?? 100);
    const offset = Number(query.get("offset") ?? 0);
    const ids = Array.from({ length: LIBRARY_SIZE }, (_, index) => index + 1);
    const items = ids.slice(offset, offset + limit).map((id) => ({
      id,
      title: `Note ${id}`,
      content: "Body",
      ...(query.get("include_keywords") === "true"
        ? { keywords: [`tag-${id}`] }
        : {}),
    }));
    return { items, pagination: { limit, offset, total: LIBRARY_SIZE } };
  });
});
afterEach(() => vi.restoreAllMocks());

const deps = (): UseNotesExportDeps => ({
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
  total: LIBRARY_SIZE,
  filteredCount: LIBRARY_SIZE,
  hasActiveFilters: false,
  selectedBulkNotes: [],
  fetchFilteredNotesRaw: vi.fn(),
  selectedId: null,
  title: "",
  content: "",
  editorKeywords: [],
});

it("retains linked tags on every page of an unfiltered JSON export", async () => {
  const { result } = renderHook(() => useNotesExport(deps()));
  await act(async () => {
    await result.current.exportAllJSON();
  });
  expect(
    JSON.parse(await exported.text()).map(
      (note: { keywords: string[] }) => note.keywords,
    ),
  ).toEqual(
    Array.from({ length: LIBRARY_SIZE }, (_, index) => [`tag-${index + 1}`]),
  );
});

it("retains linked tags in filtered JSON exports", async () => {
  const options = deps();
  options.query = "Body";
  options.hasActiveFilters = true;
  options.fetchFilteredNotesRaw = vi.fn(async () => ({
    items: [
      { id: 1, title: "Note", content: "Body", keywords: ["filtered-tag"] },
    ],
    total: 1,
  }));
  const { result } = renderHook(() => useNotesExport(options));
  await act(async () => {
    await result.current.exportAllJSON();
  });
  expect(JSON.parse(await exported.text())[0].keywords).toEqual([
    "filtered-tag",
  ]);
});
