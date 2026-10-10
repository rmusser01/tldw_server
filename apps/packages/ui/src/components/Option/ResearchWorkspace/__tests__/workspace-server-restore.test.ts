import { beforeEach, describe, expect, it, vi } from "vitest";

const boundary = vi.hoisted(() => ({
  request: vi.fn(),
  captures: [] as unknown[],
  captureRead: vi.fn(),
  controller: new AbortController(),
  release: vi.fn(),
  scopeKey: "original-alice",
  userId: "alice" as string | null,
  captureOwner: "original-alice",
  config: { serverUrl: "https://original.test", authMode: "multi-user" as "single-user" | "multi-user" },
}));
vi.mock("@/utils/research-workspace-prefill", async (original) => ({
  ...(await original<typeof import("@/utils/research-workspace-prefill")>()),
  readResearchWebCaptures: boundary.captureRead,
}));
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }));
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    scopeKey: boundary.scopeKey,
    requestScope: {
      config: boundary.config,
      userId: boundary.userId,
    },
    scopeSignal: boundary.controller.signal,
    scopeInvalidatedSignal: boundary.controller.signal,
    release: boundary.release,
  }),
}));

import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope";
import {
  buildResearchWorkspaceMigrationTombstone,
  buildResearchWorkspaceMigrationTombstoneKey,
} from "@/store/workspace-migration";
import { useWorkspaceStore } from "@/store/workspace";
import { retainKnowledgeNoteProvenance } from "@/utils/knowledge-note-provenance";
import {
  readMigratedResearchWorkspaceId,
  restoreMigratedResearchWorkspace,
} from "../workspace-server-restore";
import { isServicePromptRequestPath } from "@/services/tldw/service-prompt-scope-error";

const receiptKey = "tldw:research-workspace:migration:tombstone:original";
const context = () => ({
  workspace_id: "original",
  workspace: {
    id: "original",
    name: "Saved Research",
    created_at: "2026-10-01T00:00:00Z",
    version: 1,
    deleted: false,
    archived: false,
    workspace_profile: "research",
    study_materials_policy: "workspace",
    banner_title: "Research",
    banner_subtitle: "Two sources",
  },
  sources: {
    items: [
      {
        id: "source-1",
        workspace_id: "original",
        media_id: 7,
        title: "Larch",
        source_type: "text",
        selected: true,
        added_at: "2026-10-01T00:00:00Z",
        state: "queryable",
        readiness: { fts_ready: true },
      },
    ],
  },
  partial_errors: [] as Array<{ scope: string; code: string; message: string }>,
});
const restore = (apply?: Parameters<typeof restoreMigratedResearchWorkspace>[0]["apply"]) => {
  const origin = useWorkspaceStore.getState().workspaceId;
  return restoreMigratedResearchWorkspace({
    signal: new AbortController().signal,
    apply: apply ?? ((workspace, scopeKey) => useWorkspaceStore.getState().installServerWorkspace(
      workspace, { scopeKey, expectedWorkspaceId: origin },
    )),
  });
};

beforeEach(() => {
  localStorage.clear();
  boundary.captures = [];
  useWorkspaceStore.getState().reset();
  boundary.controller = new AbortController();
  boundary.scopeKey = "original-alice";
  boundary.userId = "alice";
  boundary.config = { serverUrl: "https://original.test", authMode: "multi-user" };
  boundary.captureOwner = buildChatSurfaceScopeKeyFromConfig(boundary.config, { userId: "alice" });
  boundary.captureRead.mockReset().mockImplementation(async (owner: string, workspace: string) =>
    owner === boundary.captureOwner && workspace === "original" ? boundary.captures : []);
  boundary.release.mockReset();
  boundary.request
    .mockReset()
    .mockImplementation(async ({ path }: { path: string }) =>
      path === "/api/v1/workspaces/"
        ? { items: [context().workspace] }
        : path.endsWith("/context")
          ? context()
          : path.endsWith("/notes")
            ? [
                {
                  id: 1,
                  workspace_id: "original",
                  title: "Saved note",
                  content: "Keep this note",
                  keywords_json: '["larch"]',
                  version: 1,
                },
              ]
            : [],
    );
  localStorage.setItem(
    receiptKey,
    JSON.stringify({
      legacyWorkspaceId: "original",
      serverWorkspaceId: "original",
      migrationId: "migration-original",
      serverScopeKey: "original-alice",
      contentRetained: false,
      deletedAt: "2026-10-03T00:00:00Z",
    }),
  );
});

describe("migrated Research Workspace restoration", () => {
  it("restores a scoped UUID note without granting provenance source membership", async () => {
    const evidence = { importId: "knowledge-import", threadId: null, snapshot: false, sources: [] };
    const content = retainKnowledgeNoteProvenance("Canonical UUID note", {
      origin: "knowledge_qa",
      research: {
        workspace_id: "original", import_id: "knowledge-import",
        sources: [{ mediaId: 7, evidence }, { mediaId: 99, evidence }],
      },
    });
    const originalRequest = boundary.request.getMockImplementation()!;
    boundary.request.mockImplementation(async (request: { path: string }) =>
      request.path.startsWith("/api/v1/notes/search/")
        ? { notes: [{ id: "12345678-1234-4123-8123-123456789abc", title: "Knowledge note", content, version: 4, keywords: ["workspace:original", "larch"] }] }
        : request.path === "/api/v1/notes/12345678-1234-4123-8123-123456789abc"
          ? { id: "12345678-1234-4123-8123-123456789abc", title: "Knowledge note", content, version: 4, keywords: ["workspace:original", "larch"] }
        : originalRequest(request),
    );
    await expect(restore()).resolves.toBe(true);
    const state = useWorkspaceStore.getState();
    expect(state.currentNote).toMatchObject({
      id: "12345678-1234-4123-8123-123456789abc", content, version: 4,
      serverWorkspaceId: "original", serverScopeKey: "original-alice", isDirty: false,
    });
    expect(state.sources).toMatchObject([{ id: "source-1", mediaId: 7, knowledgeQaEvidence: evidence }]);
    expect(state.sources).toHaveLength(1);
    expect(state.selectedSourceIds).toEqual(["source-1"]);
    expect(state.serverWorkspace?.notes).toMatchObject([{ id: 1, workspace_id: "original" }]);
    expect(boundary.request).toHaveBeenCalledWith(expect.objectContaining({
      path: expect.stringContaining("/api/v1/notes/search/"),
      servicePromptConfig: expect.objectContaining({ expectedUserId: "alice" }),
      headers: { "X-TLDW-Expected-User-ID": "alice" },
      abortSignal: boundary.controller.signal,
      method: "GET",
    }));
  });

  it("retains canonical account provenance, complete notes and reconciliation baseline", async () => {
    await restore();
    expect(useWorkspaceStore.getState().serverWorkspace).toMatchObject({
      scopeKey: "original-alice",
      metadata: { id: "original", workspace_profile: "research" },
      notes: [{ id: 1, workspace_id: "original", content: "Keep this note" }],
      selectedSourceSignature: "source-1",
    });
  });

  it("refuses restoration over a retained empty-ID draft without replacing it", async () => {
    useWorkspaceStore.getState().updateNoteContent("Retained unsaved draft");
    await expect(restore()).rejects.toThrow(/install|draft/i);
    expect(useWorkspaceStore.getState().workspaceId).toBe("");
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      content: "Retained unsaved draft", isDirty: true,
    });
  });

  it("does not report success when the final installation is refused", async () => {
    await expect(restore(() => false)).rejects.toThrow(/install/i);
    expect(useWorkspaceStore.getState().workspaceId).toBe("");
  });

  it.each([
    "jobs_unavailable",
    "media_db_unavailable",
    "membership_summary_unavailable",
  ])("restores retained sources despite informational %s", async (code) => {
    const retained = context();
    retained.partial_errors = [
      { scope: "sources", code, message: "Optional service is unavailable" },
    ];
    boundary.request.mockImplementation(async ({ path }: { path: string }) =>
      path.endsWith("/context") ? retained : [],
    );
    await expect(restore()).resolves.toBe(true);
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.id),
    ).toEqual(["source-1"]);
  });

  it.each(["deleted", "archived"] as const)(
    "invalidates a %s receipt without installing or deleting server state",
    async (state) => {
      const retained = context();
      retained.workspace[state] = true;
      retained.partial_errors = [
        {
          scope: "sources",
          code: "sources_unavailable",
          message: "Retired workspace sources unavailable",
        },
      ];
      boundary.request.mockResolvedValue(retained);
      const apply = vi.fn();
      await expect(restore(apply)).resolves.toBe(false);
      expect(readMigratedResearchWorkspaceId(localStorage)).toBeNull();
      expect(apply).not.toHaveBeenCalled();
    },
  );

  it("invalidates a receipt for a server workspace that is no longer found", async () => {
    boundary.request.mockRejectedValue(
      Object.assign(new Error("Workspace not found"), { status: 404 }),
    );
    await expect(restore()).resolves.toBe(false);
    expect(readMigratedResearchWorkspaceId(localStorage)).toBeNull();
  });

  it("retains a valid identity on a transient context failure", async () => {
    boundary.request.mockRejectedValue(
      Object.assign(new Error("Temporary outage"), { status: 503 }),
    );
    await expect(restore()).rejects.toMatchObject({ status: 503 });
    expect(readMigratedResearchWorkspaceId(localStorage)).toBe("original");
  });

  it("retains a valid identity when a source-level context error prevents complete restoration", async () => {
    const retained = context();
    retained.partial_errors = [
      {
        scope: "sources",
        code: "sources_unavailable",
        message: "Source rows unavailable",
      },
    ];
    boundary.request.mockResolvedValue(retained);
    await expect(restore()).rejects.toThrow("restored completely");
    expect(readMigratedResearchWorkspaceId(localStorage)).toBe("original");
  });

  it("retains full canonical note content even when keywords JSON is malformed", async () => {
    boundary.request.mockImplementation(async ({ path }: { path: string }) =>
      path.endsWith("/context")
        ? context()
        : path.endsWith("/notes")
          ? [
              {
                id: 1,
                workspace_id: "original",
                title: "Saved note",
                content: "Keep this note",
                keywords_json: "bad JSON",
                version: 1,
              },
            ]
          : [],
    );
    await expect(restore()).resolves.toBe(true);
    expect(useWorkspaceStore.getState().serverWorkspace?.notes[0]).toMatchObject({
      content: "Keep this note", keywords_json: "bad JSON",
    });
    expect(useWorkspaceStore.getState().currentNote.id).toBeUndefined();
  });

  it("confirms an unbound legacy identity in the scoped server list before binding and restoring it", async () => {
    const receipt = JSON.parse(localStorage.getItem(receiptKey)!);
    delete receipt.serverScopeKey;
    localStorage.setItem(receiptKey, JSON.stringify(receipt));
    await expect(restore()).resolves.toBe(true);
    expect(boundary.request.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/workspaces/",
      servicePromptConfig: { expectedUserId: "alice" },
    });
    expect(JSON.parse(localStorage.getItem(receiptKey)!)).toMatchObject({
      serverScopeKey: "original-alice",
    });
    expect(useWorkspaceStore.getState().workspaceId).toBe("original");
  });

  it("ignores Alice's scope-bound receipt after switching to Bob", async () => {
    boundary.scopeKey = "original-bob";
    boundary.userId = "bob";
    const apply = vi.fn();
    await expect(restore(apply)).resolves.toBe(false);
    expect(boundary.request).not.toHaveBeenCalled();
    expect(apply).not.toHaveBeenCalled();
  });

  it("does not treat an unbound legacy receipt as Bob's mandatory workspace", async () => {
    const receipt = JSON.parse(localStorage.getItem(receiptKey)!);
    delete receipt.serverScopeKey;
    localStorage.setItem(receiptKey, JSON.stringify(receipt));
    boundary.scopeKey = "original-bob";
    boundary.userId = "bob";
    boundary.request.mockResolvedValue({ items: [] });
    const apply = vi.fn();
    await expect(restore(apply)).resolves.toBe(false);
    expect(boundary.request).toHaveBeenCalledOnce();
    expect(boundary.request).toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/workspaces/" }),
    );
    expect(apply).not.toHaveBeenCalled();
  });

  it("uses the receipt hint and restores the authorized server state atomically", async () => {
    await restore();
    expect(useWorkspaceStore.getState()).toMatchObject({
      workspaceId: "original",
      workspaceName: "Saved Research",
      workspaceChatReferenceId: "original",
      sources: [
        {
          id: "source-1",
          mediaId: 7,
          status: "ready",
          readiness: { fts_ready: true },
        },
      ],
      selectedSourceIds: ["source-1"],
      serverWorkspace: { scopeKey: "original-alice", notes: [{ title: "Saved note", content: "Keep this note" }] },
    });
    expect(readMigratedResearchWorkspaceId(localStorage)).toBe("original");
    expect(
      boundary.request.mock.calls.every(
        ([request]) =>
          request.servicePromptConfig.expectedUserId === "alice" &&
          request.headers["X-TLDW-Expected-User-ID"] === "alice" &&
          request.abortSignal === boundary.controller.signal,
      ),
    ).toBe(true);
    expect(boundary.release).toHaveBeenCalledOnce();
  });

  it("retains the server identity hint when deleted legacy persistence is rehydrated again", async () => {
    await restore();
    await useWorkspaceStore.persist.rehydrate();
    const restoredId = readMigratedResearchWorkspaceId(localStorage);
    expect(restoredId).toBe("original");
    await restore();
    expect(
      useWorkspaceStore.getState().sources.map((source) => source.id),
    ).toEqual(["source-1"]);
  });

  it("does not install a server response after its account lease is invalidated", async () => {
    const apply = vi.fn();
    boundary.request.mockImplementation(async ({ path }: { path: string }) => {
      if (path.endsWith("/context")) return context();
      boundary.controller.abort();
      return [];
    });
    await expect(restore(apply)).rejects.toMatchObject({ status: 412 });
    expect(apply).not.toHaveBeenCalled();
    expect(boundary.release).toHaveBeenCalledOnce();
  });

  it("does not install sources belonging to another workspace", async () => {
    const apply = vi.fn();
    const foreign = context();
    foreign.sources.items[0].workspace_id = "another";
    boundary.request.mockResolvedValue(foreign);
    await expect(restore(apply)).rejects.toThrow("restored completely");
    expect(apply).not.toHaveBeenCalled();
  });

  it("ignores malformed receipt hints", () => {
    localStorage.setItem(receiptKey, "invalid JSON");
    expect(readMigratedResearchWorkspaceId(localStorage)).toBeNull();
  });

  it.each(["context", "artifacts", "notes"])(
    "permits scoped restoration GET for %s without permitting mutations",
    (resource) => {
      const path = `/api/v1/workspaces/original/${resource}`;
      expect(isServicePromptRequestPath(path, "GET")).toBe(true);
      expect(isServicePromptRequestPath(path, "POST")).toBe(false);
      expect(isServicePromptRequestPath(path, "PUT")).toBe(false);
      expect(
        isServicePromptRequestPath(
          `/api/v1/workspaces/original%2Fanother/${resource}`,
          "GET",
        ),
      ).toBe(false);
    },
  );
});

it.each(["active", "deleted"])("resolves exact owned %s history after a lightweight search", async state => {
  const history = { origin: "knowledge_qa", research: { workspace_id: "original", import_id: "import-a", sources: [] } }
  boundary.request.mockImplementation(async ({ path }: { path: string }) => {
    if (path.endsWith("/context")) return context()
    if (path.endsWith("/notes")) return [{ id: 1, workspace_id: "original", title: "Legacy", content: `Legacy body\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`, keywords_json: "[]", version: 1 }]
    if (path.includes("/notes/search/")) return { notes: [{ id: "canonical-note", title: "Research" }] }
    if (path === "/api/v1/notes/canonical-note") return { id: "canonical-note", title: "Research", content: `Edited body\n\n<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`, version: 8,
      keywords: ["workspace:original"], knowledge_provenance_state: state,
      knowledge_provenance_version: 4, knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
      knowledge_provenance: state === "active" ? history : null,
    }
    return []
  })
  await expect(restore()).resolves.toBe(true)
  expect(boundary.request).toHaveBeenCalledWith(expect.objectContaining({ path: "/api/v1/notes/canonical-note", servicePromptConfig: expect.objectContaining({ serverUrl: "https://original.test" }) }))
  if (state === "active") expect(useWorkspaceStore.getState().currentNote).toMatchObject({ id: "canonical-note", knowledge_provenance_version: 4 })
  else {
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({ id: "canonical-note", content: "Edited body", knowledge_provenance_state: "deleted", knowledge_provenance_version: 4 })
    expect(useWorkspaceStore.getState().sources[0].knowledgeQaEvidence).toBeUndefined()
    expect(useWorkspaceStore.getState().notes).toBe("Edited body")
  }
})
it("allows only an owned explicit source-history restore route", () => {
  expect(isServicePromptRequestPath("/api/v1/notes/owned/provenance/restore", "POST")).toBe(true)
  expect(isServicePromptRequestPath("/api/v1/notes/owned%2Fforeign/provenance/restore", "POST")).toBe(false)
})

it.each(["single-user", "multi-user"] as const)("restores exact pins from the unchanged public namespace for %s", async (mode) => {
  boundary.config = { serverUrl: "https://original.test", authMode: mode };
  boundary.userId = mode === "single-user" ? null : "alice";
  const config = { ...boundary.config, apiKey: "synthetic-test-key" };
  boundary.scopeKey = buildChatSurfaceScopeKeyFromConfig(config, { userId: boundary.userId });
  boundary.captureOwner = buildChatSurfaceScopeKeyFromConfig({ ...config, apiKey: undefined }, { userId: boundary.userId });
  if (mode === "single-user") expect(boundary.scopeKey).not.toBe(boundary.captureOwner);
  localStorage.setItem(receiptKey, JSON.stringify({ ...JSON.parse(localStorage.getItem(receiptKey)!), serverScopeKey: boundary.scopeKey }));
  const pin = {
    clipId: "clip",
    requestedUrl: "https://example.org",
    capturedAt: "2026-10-07T00:00:00Z",
    contentSha256: "digest",
    refreshOf: "prior",
    mediaId: 7,
    versionNumber: 9,
    versionUuid: "version-nine",
  };
  boundary.captures = [{ pin }];
  const ctx = context();
  Object.assign(ctx.sources.items[0], {
    id: "web-clipper:clip",
    url: pin.requestedUrl,
  });
  boundary.request.mockImplementation(async ({ path }: { path: string }) =>
    path.endsWith("/context") ? ctx : [],
  );
  await restore();
  expect(useWorkspaceStore.getState().sources[0].webCapture).toEqual(pin);
  ctx.sources.items = [];
  await restore();
  expect(useWorkspaceStore.getState().sources).toEqual([]);
});

const restoreCaptureFixture = async () => {
  const pin = {
    clipId: "capture-clip",
    requestedUrl: "https://example.org/article",
    capturedAt: "2026-10-07T00:00:00Z",
    contentSha256: "digest",
    refreshOf: null,
    mediaId: 7,
    versionNumber: 1,
    versionUuid: "version-one",
  };
  boundary.captures = [{ pin }];
  const ctx = context();
  Object.assign(ctx.sources.items[0], {
    id: "web-clipper:capture-clip",
    url: pin.requestedUrl,
  });
  boundary.request.mockImplementation(async ({ path }: { path: string }) =>
    path.endsWith("/context") ? ctx : [],
  );
  await restore();
  useWorkspaceStore
    .getState()
    .setSourceStatusById(
      "web-clipper:capture-clip",
      "error",
      "Snapshot changed outside refresh",
      undefined,
      { statusReason: "capture_head_changed", retryEligible: false },
    );
  return { pin, ctx };
};

it.each(["current", "persisted", "inactive"])(
  "canonical restore retains a known capture refusal from %s display state",
  async (display) => {
    const { pin } = await restoreCaptureFixture();
    if (display === "persisted") {
      // The canonical workspace is distinct from the intentionally tombstoned legacy cache.
      const receipt = JSON.parse(localStorage.getItem(receiptKey)!);
      localStorage.removeItem(receiptKey);
      localStorage.setItem(
        buildResearchWorkspaceMigrationTombstoneKey("legacy"),
        JSON.stringify({
          ...receipt,
          ...buildResearchWorkspaceMigrationTombstone({
            ...receipt,
            legacyWorkspaceId: "legacy",
          }),
        }),
      );
      // Await the public native persistence write; the earlier identical-ID tombstone suppressed it.
      const persistence = useWorkspaceStore.persist.getOptions();
      await persistence.storage!.setItem(persistence.name, {
        state: persistence.partialize!(useWorkspaceStore.getState()),
        version: persistence.version,
      });
      await useWorkspaceStore.persist.rehydrate();
    }
    if (display === "inactive") {
      useWorkspaceStore.getState().saveCurrentWorkspace();
      useWorkspaceStore.setState({ workspaceId: "another", sources: [] });
    }
    await restore();
    const state = useWorkspaceStore.getState();
    expect(state.sources[0]).toMatchObject({
      webCapture: pin,
      status: "error",
      statusMessage: "Snapshot changed outside refresh",
      statusDetails: { statusReason: "capture_head_changed" },
    });
    expect(state.selectedSourceIds).toEqual([]);
    expect(state.getSelectedMediaIds()).toEqual([]);
    state.toggleSourceSelection("web-clipper:capture-clip");
    expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([]);
  },
);

it.each([
  "workspace",
  "snapshot-workspace",
  "pin",
  "ordinary",
  "transient",
  "removed",
])(
  "canonical restore does not transfer refusal across %s boundaries",
  async (change) => {
    const { pin, ctx } = await restoreCaptureFixture();
    if (change === "workspace")
      useWorkspaceStore.setState({
        workspaceId: "another",
        workspaceSnapshots: {},
      });
    if (change === "snapshot-workspace") {
      useWorkspaceStore.getState().saveCurrentWorkspace();
      const snapshot = useWorkspaceStore.getState().workspaceSnapshots.original;
      useWorkspaceStore.setState({
        workspaceId: "another",
        sources: [],
        workspaceSnapshots: {
          original: { ...snapshot, workspaceId: "another" },
        },
      });
    }
    if (change === "pin")
      boundary.captures = [
        { pin: { ...pin, versionNumber: 2, versionUuid: "version-two" } },
      ];
    if (change === "ordinary") boundary.captures = [];
    if (change === "transient")
      useWorkspaceStore
        .getState()
        .setSourceStatusById("web-clipper:capture-clip", "ready");
    if (change === "removed") ctx.sources.items = [];
    await restore();
    const state = useWorkspaceStore.getState();
    if (change === "removed") expect(state.sources).toEqual([]);
    else {
      expect(state.sources[0].status).toBe("ready");
      expect(state.sources[0].statusMessage).toBeUndefined();
      expect(state.getSelectedMediaIds()).toEqual([7]);
      if (change === "ordinary")
        expect(state.sources[0].webCapture).toBeUndefined();
      if (change === "pin")
        expect(state.sources[0].webCapture?.versionNumber).toBe(2);
    }
  },
);

it("refuses Bob installation over Alice's protected same-ID cache, then activates after retirement", async () => {
  await restoreCaptureFixture();
  const cached = useWorkspaceStore.getState();
  boundary.userId = "bob";
  boundary.scopeKey = "original-bob";
  localStorage.setItem(receiptKey, JSON.stringify({
    ...JSON.parse(localStorage.getItem(receiptKey)!), serverScopeKey: boundary.scopeKey,
  }));
  const persistence = useWorkspaceStore.persist.getOptions();
  await persistence.storage!.setItem(persistence.name, {
    state: persistence.partialize!(cached), version: persistence.version,
  });
  await useWorkspaceStore.persist.rehydrate();
  const retained = useWorkspaceStore.getState();
  expect(retained.serverWorkspace?.scopeKey).toBe("original-alice");
  expect(retained.sources).toEqual(cached.sources);
  expect(retained.currentNote).toEqual(cached.currentNote);
  await expect(restore()).rejects.toThrow(/install/i);
  expect(useWorkspaceStore.getState().sources).toEqual(retained.sources);
  expect(useWorkspaceStore.getState().currentNote).toEqual(retained.currentNote);
  expect(useWorkspaceStore.getState().workspaceSnapshots).toEqual(retained.workspaceSnapshots);
  expect(useWorkspaceStore.getState().serverWorkspace).toEqual(retained.serverWorkspace);
  useWorkspaceStore.getState().deleteWorkspace("original");
  expect(useWorkspaceStore.getState().workspaceSnapshots.original).toBeUndefined();
  await expect(restore()).resolves.toBe(true);
  expect(useWorkspaceStore.getState().serverWorkspace?.scopeKey).toBe("original-bob");
  expect(useWorkspaceStore.getState().sources[0]).toMatchObject({ status: "ready" });
  expect(useWorkspaceStore.getState().sources[0].webCapture).toBeUndefined();
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([7]);
});

it("canonical restore never revives refusal display from a deleted legacy same-ID cache", async () => {
  await restoreCaptureFixture();
  // A retired legacy cache has no installed canonical ownership baseline.
  useWorkspaceStore.setState({ serverWorkspace: null });
  useWorkspaceStore.getState().saveCurrentWorkspace();
  const persistence = useWorkspaceStore.persist.getOptions();
  await persistence.storage!.setItem(persistence.name, {
    state: persistence.partialize!(useWorkspaceStore.getState()),
    version: persistence.version,
  });
  await useWorkspaceStore.persist.rehydrate();
  expect(
    useWorkspaceStore
      .getState()
      .sources.every(
        (source) =>
          source.statusDetails?.statusReason !== "capture_head_changed",
      ),
  ).toBe(true);
  await restore();
  expect(useWorkspaceStore.getState().sources[0].status).toBe("ready");
  expect(useWorkspaceStore.getState().sources[0].statusMessage).toBeUndefined();
  expect(JSON.parse(localStorage.getItem(receiptKey)!).contentRetained).toBe(
    false,
  );
});

it("preserves a verified canonical same-ID cache and refusal despite a legacy tombstone", async () => {
  const { pin } = await restoreCaptureFixture();
  const persistence = useWorkspaceStore.persist.getOptions();
  await persistence.storage!.setItem(persistence.name, {
    state: persistence.partialize!(useWorkspaceStore.getState()), version: persistence.version,
  });
  await useWorkspaceStore.persist.rehydrate();
  expect(useWorkspaceStore.getState().serverWorkspace?.scopeKey).toBe("original-alice");
  expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
    webCapture: pin, statusDetails: { statusReason: "capture_head_changed" },
  });
  await expect(restore()).resolves.toBe(true);
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
});

it.each(["account-aba", "workspace-aba"])("migration cannot install after %s during capture hydration", async change => {
  await restoreCaptureFixture();
  let resolve!: (records: unknown[]) => void;
  boundary.captureRead.mockReturnValue(new Promise<unknown[]>(done => { resolve = done; }));
  const signal = new AbortController();
  const origin = useWorkspaceStore.getState().workspaceId;
  const stopWorkspace = useWorkspaceStore.subscribe((state, previous) => {
    if (state.workspaceId !== previous.workspaceId) signal.abort();
  });
  const apply = vi.fn();
  const pending = restoreMigratedResearchWorkspace({ signal: signal.signal, apply })
    .then(() => null, error => error);
  await vi.waitFor(() => expect(boundary.captureRead).toHaveBeenCalledTimes(2));
  if (change === "account-aba") {
    boundary.controller.abort(); boundary.userId = "bob"; boundary.userId = "alice";
  } else {
    useWorkspaceStore.getState().createNewWorkspace("Intervening");
    useWorkspaceStore.getState().switchWorkspace(origin);
  }
  resolve(boundary.captures);
  expect(await pending).toMatchObject({
    message: change === "workspace-aba" ? "Workspace restoration cancelled" : expect.stringMatching(/scope|account/i),
  });
  expect(apply).not.toHaveBeenCalled();
  expect(useWorkspaceStore.getState().workspaceId).toBe(origin);
  expect(useWorkspaceStore.getState().sources[0].statusDetails?.statusReason).toBe("capture_head_changed");
  stopWorkspace();
});
