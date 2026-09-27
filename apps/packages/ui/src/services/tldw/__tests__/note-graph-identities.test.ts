import { buildNotesGraphRelationshipGroups } from "@/components/Notes/NotesGraphRelationshipsView";
import {
  fetchNotesGraph,
  type NotesGraphNode,
  type NotesGraphResponse,
} from "@/services/note-graph-suggestions";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ bgRequest: vi.fn() }));
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }));

const node = (id: string, type: NotesGraphNode["type"]): NotesGraphNode => ({
  id,
  type,
  label: id,
  created_at: null,
  deleted: false,
  degree: 1,
  tag_count: 1,
  primary_source_id: null,
});

const graphPayload = (): NotesGraphResponse => ({
  nodes: [node("alpha", "note"), node("tag:amber", "tag")],
  edges: [
    {
      id: "tag-edge",
      source: "alpha",
      target: "tag:amber",
      type: "tag_membership",
      directed: false,
      weight: 1,
      label: "Tag",
    },
  ],
  truncated: false,
  truncated_by: [],
  has_more: false,
  cursor: null,
  limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
  radius_cap_applied: false,
  active_note_count: 1,
  all_notes_note_cap: 100,
  all_notes_eligible: true,
  suggestions_authorized: true,
});

describe("Notes graph identities at the API boundary", () => {
  beforeEach(() => vi.resetAllMocks());

  it("resolves the focused note and its real relationship from raw API note IDs", async () => {
    mocks.bgRequest.mockResolvedValueOnce(graphPayload());
    const graph = await fetchNotesGraph({ centerNoteId: "alpha" });
    const selectedNodeId =
      graph.nodes.find(({ id }) => id === "note:alpha")?.id ?? null;

    expect(
      buildNotesGraphRelationshipGroups({
        graph,
        selectedNodeId,
        provisionalOverlays: [],
        suggestions: [],
      }),
    ).toEqual([
      {
        id: "connected",
        rows: [
          {
            id: "tag-edge",
            group: "connected",
            edgeType: "tag_membership",
            counterpart: { id: "tag:amber", label: "tag:amber" },
            suggestion: null,
          },
        ],
      },
    ]);
  });

  it("normalizes both note endpoints while preserving tag and source identities", async () => {
    const payload = graphPayload();
    payload.nodes.push(
      node("note:beta", "note"),
      node("source:book", "source"),
    );
    payload.edges.push(
      {
        ...payload.edges[0],
        id: "manual-edge",
        source: "note:beta",
        target: "alpha",
        type: "manual",
        directed: true,
      },
      {
        ...payload.edges[0],
        id: "source-edge",
        source: "alpha",
        target: "source:book",
        type: "source_membership",
      },
    );
    mocks.bgRequest.mockResolvedValueOnce(payload);

    const graph = await fetchNotesGraph({ centerNoteId: "alpha" });
    expect(graph.nodes.map(({ id }) => id)).toEqual([
      "note:alpha",
      "tag:amber",
      "note:beta",
      "source:book",
    ]);
    expect(
      graph.edges.map(({ id, source, target }) => ({ id, source, target })),
    ).toEqual([
      { id: "tag-edge", source: "note:alpha", target: "tag:amber" },
      { id: "manual-edge", source: "note:beta", target: "note:alpha" },
      { id: "source-edge", source: "note:alpha", target: "source:book" },
    ]);
    expect(payload.nodes[0].id).toBe("alpha");
    expect(payload.edges[0].source).toBe("alpha");
  });

  it("keeps already-normalized graphs unchanged", async () => {
    const payload = graphPayload();
    payload.nodes[0].id = "note:alpha";
    payload.edges[0].source = "note:alpha";
    mocks.bgRequest.mockResolvedValueOnce(payload);
    expect(await fetchNotesGraph({ centerNoteId: "alpha" })).toEqual(payload);
  });

  it("normalizes each cursor page without changing metadata or cursor requests", async () => {
    const first = {
      ...graphPayload(),
      has_more: true,
      cursor: "opaque cursor",
      truncated: true,
      truncated_by: ["max_nodes"],
    };
    const second = graphPayload();
    second.nodes[0] = node("gamma", "note");
    second.edges[0].source = "gamma";
    mocks.bgRequest.mockResolvedValueOnce(first).mockResolvedValueOnce(second);

    const page1 = await fetchNotesGraph({ centerNoteId: "alpha" });
    const page2 = await fetchNotesGraph({
      centerNoteId: "alpha",
      cursor: page1.cursor!,
    });
    expect(page1).toEqual({
      ...first,
      nodes: [{ ...first.nodes[0], id: "note:alpha" }, first.nodes[1]],
      edges: [{ ...first.edges[0], source: "note:alpha" }],
    });
    expect(page2.nodes[0].id).toBe("note:gamma");
    expect(page2.edges[0].source).toBe("note:gamma");
    expect(mocks.bgRequest.mock.calls[1][0].path).toContain(
      "cursor=opaque+cursor",
    );
  });

  it("does not invent note identities for unrelated or missing endpoints", async () => {
    const payload = graphPayload();
    payload.nodes = [
      node("unprefixed-tag", "tag"),
      node("unprefixed-source", "source"),
    ];
    payload.edges[0] = {
      ...payload.edges[0],
      source: "unprefixed-tag",
      target: "missing-node",
    };
    mocks.bgRequest.mockResolvedValueOnce(payload);
    expect(await fetchNotesGraph({})).toEqual(payload);
  });
});
