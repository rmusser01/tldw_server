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
