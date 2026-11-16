import { getSourceFeedbackKey } from "../feedback";
import { describe, expect, it } from "vitest";
import {
  projectWorkspaceRagSources,
  readWorkspaceRagSourceProvenance,
  restoreWorkspaceRagSources,
} from "../workspace-rag-sources";

const source = {
  name: "Rowan",
  type: "text",
  pageContent: "Selene is the telescope.",
  metadata: { media_id: 42, secret: "do-not-retain" },
};
describe("S04 recorded workspace source references", () => {
  const spans = [
    { page_number: 871, start_char: 100, end_char: 120, printed_page_label: "838" },
    { page_number: 872, start_char: 122, end_char: 140, printed_page_label: "839" },
  ];
  it("preserves every canonical page span and exact source identity through migration", () => {
    const row = { ...source, type: "pdf", metadata: {
      media_id: 42, page_number: 871, citation_index: 6, chunk_id: "chunk-2843", page_spans: spans,
    } };
    const saved = projectWorkspaceRagSources([row])!;
    expect(saved.sources![0]).toMatchObject({ pageSpans: spans.map(span => ({
      pageNumber: span.page_number, startChar: span.start_char, endChar: span.end_char, printedPageLabel: span.printed_page_label,
    })) });
    expect(restoreWorkspaceRagSources(JSON.parse(JSON.stringify(saved)))[0]).toMatchObject({
      pageContent: source.pageContent, metadata: { page_spans: spans, chunk_id: "chunk-2843", citation_index: 6, media_id: 42 },
    });
  });
  it.each([
    [], Array(21).fill(spans[0]), [{ ...spans[0], page_number: 0 }],
    [{ ...spans[0], start_char: -1 }], [{ ...spans[0], end_char: 100 }],
    [spans[1], spans[0]], [spans[0], { ...spans[1], start_char: 119 }],
    [{ ...spans[0], page_number: "871" }], [{ ...spans[0], printed_page_label: "x".repeat(513) }],
  ])("drops malformed raw spans while retaining the legacy page fallback", page_spans => {
    const saved = projectWorkspaceRagSources([{ ...source, metadata: { media_id: 42, page_number: 871, page_spans } }])!;
    expect(saved.sources![0]).toMatchObject({ pageNumber: 871 });
    expect(saved.sources![0]).not.toHaveProperty("pageSpans");
  });
  it("rejects malformed persisted spans instead of treating them as evidence", () => {
    const saved = projectWorkspaceRagSources([source])!;
    expect(readWorkspaceRagSourceProvenance({ ...saved, sources: [{ ...saved.sources![0],
      pageSpans: [{ pageNumber: 871, startChar: 120, endChar: 100 }],
    }] })).toBeNull();
  });
  it("keeps same-document passages distinct after save and reopen", () => {
    const rows = [1, 2].map(index => ({ ...source, type: "pdf", metadata: {
      media_id: 42, chunk_id: `chunk-${index}`, citation_index: index,
      page_number: 352 + index, page_spans: [{ page_number: 352 + index, printed_page_label: String(319 + index) }]
    }}));
    const restored = restoreWorkspaceRagSources(projectWorkspaceRagSources(rows));
    expect(restored.map((row, index) => getSourceFeedbackKey(row, index))).toEqual(["chunk:chunk-1", "chunk:chunk-2"]);
    expect(restored[0].metadata).toMatchObject({ chunk_id: "chunk-1", printed_page_label: "320", page_number: 353, citation_index: 1 });
  });
  it("keeps legacy filename-only citation cards distinct", () => {
    expect(getSourceFeedbackKey(source, 0)).not.toBe(getSourceFeedbackKey(source, 1));
  });
  it("preserves physical page and citation identity through a sourced note roundtrip", () => {
    const recorded = projectWorkspaceRagSources([{
      ...source, type: "pdf", metadata: { media_id: 42, page_number: 352, citation_index: 2 }
    }])!;
    expect(recorded.sources![0]).toMatchObject({ pageNumber: 352, citationIndex: 2 });
    expect(restoreWorkspaceRagSources(recorded)[0].metadata).toMatchObject({ page_number: 352, citation_index: 2 });
  });
  it("does not turn malformed location metadata into a saved citation", () => {
    const recorded = projectWorkspaceRagSources([{
      ...source, metadata: { media_id: 42, page_number: -4, citation_index: "2oops" }
    }])!;
    expect(recorded.sources![0]).not.toHaveProperty("pageNumber");
    expect(recorded.sources![0]).not.toHaveProperty("citationIndex");
  });
  it("round-trips only the bounded reference fields, without treating arbitrary metadata as evidence", () => {
    const recorded = projectWorkspaceRagSources([source])!;
    expect(recorded.sources).toEqual([
      {
        originalId: 42,
        mediaId: 42,
        title: "Rowan",
        excerpt: source.pageContent,
        type: "text",
        sourceType: "text",
      },
    ]);
    expect(restoreWorkspaceRagSources(recorded)).toEqual([
      {
        name: "Rowan",
        type: "text",
        mode: "rag",
        url: "",
        pageContent: source.pageContent,
        metadata: { media_id: 42, title: "Rowan", type: "text" },
      },
    ]);
  });
  it.each(["42junk", 0, -1, true, Number.MAX_SAFE_INTEGER + 1])(
    "rejects an invalid Media identity %j",
    (id) => {
      expect(
        projectWorkspaceRagSources([{ ...source, metadata: { media_id: id } }]),
      ).toBeNull();
    },
  );
  it.each([
    [],
    Array(65).fill(source),
    [{ ...source, pageContent: "" }],
    [{ ...source, pageContent: "x".repeat(100001) }],
    [{ ...source, pageContent: "😀".repeat(30000) }],
    [{ ...source, pageContent: "\ud800" }],
  ])("rejects missing, excessive or nonportable evidence", (rows) => {
    expect(projectWorkspaceRagSources(rows)).toBeNull();
  });
  it("omits unsafe retrieval URLs and refuses unsafe or extended persisted envelopes", () => {
    const recorded = projectWorkspaceRagSources([
      { ...source, url: "javascript:alert(1)" },
    ])!;
    expect(recorded.sources![0]).not.toHaveProperty("url");
    expect(
      readWorkspaceRagSourceProvenance({ ...recorded, authority: true }),
    ).toBeNull();
    expect(
      restoreWorkspaceRagSources({
        ...recorded,
        sources: [{ ...recorded.sources![0], url: "file:///private/secret" }],
      }),
    ).toEqual([]);
  });
});
