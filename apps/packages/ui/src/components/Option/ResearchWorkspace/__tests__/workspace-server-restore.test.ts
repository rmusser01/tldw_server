import { createElement } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const boundary = vi.hoisted(() => ({
  request: vi.fn(),
  captures: [] as unknown[],
  controller: new AbortController(),
  release: vi.fn(),
  scopeKey: "original-alice",
  userId: "alice" as string | null,
  captureOwner: "original-alice",
  config: { serverUrl: "https://original.test", authMode: "multi-user" as "single-user" | "multi-user" },
}));
vi.mock("@/utils/research-workspace-prefill", async (original) => ({
  ...(await original<typeof import("@/utils/research-workspace-prefill")>()),
  readResearchWebCaptures: async (owner: string, workspace: string) =>
    owner === boundary.captureOwner && workspace === "original"
      ? boundary.captures
      : [],
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
const restore = (apply = useWorkspaceStore.getState().restoreServerWorkspace) =>
  restoreMigratedResearchWorkspace({
    signal: new AbortController().signal,
    apply,
  });

beforeEach(() => {
  localStorage.clear();
  boundary.captures = [];
  useWorkspaceStore.getState().reset();
  boundary.controller = new AbortController();
  boundary.scopeKey = "original-alice";
  boundary.userId = "alice";
  boundary.config = { serverUrl: "https://original.test", authMode: "multi-user" };
  boundary.captureOwner = buildChatSurfaceScopeKeyFromConfig(boundary.config, { userId: "alice" });
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

  it("restores note content with empty keywords when legacy keywords JSON is malformed", async () => {
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
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      content: "Keep this note",
      keywords: [],
    });
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
      currentNote: {
        title: "Saved note",
        content: "Keep this note",
        isDirty: false,
      },
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
  "owner",
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
    if (change === "owner") {
      boundary.userId = "bob";
      boundary.scopeKey = "original-bob";
      localStorage.setItem(
        receiptKey,
        JSON.stringify({
          ...JSON.parse(localStorage.getItem(receiptKey)!),
          serverScopeKey: boundary.scopeKey,
        }),
      );
    }
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
    if (change === "ordinary") {
      boundary.captures = [];
      Object.assign(ctx.sources.items[0], {
        id: "web-clipper:ordinary-clip",
        url: "https://example.org/ordinary",
      });
    }
    if (change === "owner" || change === "ordinary") {
      const { tldwClient } = await import("@/services/tldw/TldwApiClient");
      const row = ctx.sources.items[0] as (typeof ctx.sources.items)[0] & {
        url: string;
      };
      const clipId = row.id.slice("web-clipper:".length);
      vi.spyOn(tldwClient, "getWebClipStatus").mockResolvedValue({
        clip_id: clipId,
        status: "saved" as const,
        note: { id: "owned-note", title: "Ordinary", version: 1 },
        workspace_placements: [
          {
            workspace_id: "original",
            source_note_id: "owned-note",
            workspace_note_id: 1,
          },
        ],
        attachments: [],
        analysis: {},
        content_budget: {},
      });
      vi.spyOn(tldwClient, "listMediaDocumentVersions").mockResolvedValue([
        {
          media_id: 7,
          version_number: 1,
          created_at: pin.capturedAt,
          safe_metadata: {
            source: "web_clipper",
            clip_id: clipId,
            workspace_id: "original",
            source_url: row.url,
          },
        },
      ]);
    }
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
      if (change === "owner" || change === "ordinary")
        expect(state.sources[0].webCapture).toBeUndefined();
      if (change === "pin")
        expect(state.sources[0].webCapture?.versionNumber).toBe(2);
    }
  },
);

it("canonical restore never revives refusal display intentionally scrubbed by an identical-ID tombstone", async () => {
  await restoreCaptureFixture();
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


const intactCapture = async () => {
  const clipId = "9c442db9-65b4-409a-b4f7-3cfe80cc963b";
  const text = "Original accepted article";
  const { sha256Text } = await import("@/store/workspace-migration");
  const descriptor = {
    mode: "server_article",
    requested_url: "https://example.org/article",
    captured_at: "2026-10-07T00:00:00Z",
    content_sha256: await sha256Text(text),
    refresh_of: null,
  };
  const version = {
    media_id: 7,
    version_number: 1,
    uuid: "44d2295a-7165-4e28-bbb0-b488a2bc404e",
    created_at: descriptor.captured_at,
    content: text,
    safe_metadata: {
      source: "web_clipper",
      clip_type: "article",
      clip_id: clipId,
      workspace_id: "original",
      source_url: descriptor.requested_url,
      capture_metadata: { web_capture_v1: descriptor },
    },
  };
  const ctx = context();
  Object.assign(ctx.sources.items[0], {
    id: `web-clipper:${clipId}`,
    url: descriptor.requested_url,
  });
  const status = {
    clip_id: clipId,
    status: "saved" as const,
    note: { id: "canonical", title: "Article", version: 1 },
    workspace_placements: [
      {
        workspace_id: "original",
        source_note_id: "canonical",
        workspace_note_id: 1,
      },
    ],
    attachments: [],
    analysis: {},
    content_budget: {},
  };
  const { tldwClient } = await import("@/services/tldw/TldwApiClient");
  vi.spyOn(tldwClient, "getWebClipStatus").mockResolvedValue(status);
  vi.spyOn(tldwClient, "listMediaDocumentVersions").mockResolvedValue([
    version,
  ]);
  vi.spyOn(tldwClient, "getMediaDocumentVersion").mockResolvedValue(version);
  vi.spyOn(tldwClient, "getWorkspaceSources").mockResolvedValue(
    ctx.sources.items as never,
  );
  boundary.request.mockImplementation(async ({ path }: { path: string }) =>
    path.endsWith("/context") ? ctx : [],
  );
  return { ctx, version, descriptor, tldwClient, status };
};

it("restores the accepted version from owned history without a local checkpoint or current workspace", async () => {
  const { version, tldwClient } = await intactCapture();
  expect(useWorkspaceStore.getState().workspaceId).toBe("");
  await restore();
  const source = useWorkspaceStore.getState().sources[0];
  expect(source.webCapture).toMatchObject({
    versionNumber: 1,
    versionUuid: version.uuid,
    contentSha256:
      version.safe_metadata.capture_metadata.web_capture_v1.content_sha256,
  });
  expect(tldwClient.getMediaDocumentVersion).toHaveBeenCalledWith(
    7,
    1,
    expect.objectContaining({
      requestScope: expect.objectContaining({ userId: "alice" }),
    }),
  );
  expect(
    boundary.request.mock.calls.every(([request]) => request.method === "GET"),
  ).toBe(true);
  vi.mocked(tldwClient.listMediaDocumentVersions).mockResolvedValue([
    { ...version, version_number: 2, content: "Edited current body" },
    version,
  ]);
  const { assertWebCaptureHeadCurrent } =
    await import("@/utils/research-web-capture");
  await expect(
    assertWebCaptureHeadCurrent(source, "original", {
      requestScope: { config: boundary.config, userId: "alice" },
    }),
  ).rejects.toThrow("no longer current");
});

it.each([
  "ambiguous",
  "digest",
  "binding",
  "missing-exact",
  "changed-uuid",
  "pagination",
  "status-binding",
])("blocks checkpointless capture with %s evidence", async (change) => {
  const { version, tldwClient, status } = await intactCapture();
  if (change === "ambiguous")
    vi.mocked(tldwClient.listMediaDocumentVersions).mockResolvedValue([
      {
        ...version,
        version_number: 2,
        uuid: "66d2295a-7165-4e28-bbb0-b488a2bc404e",
      },
      version,
    ]);
  if (change === "digest") version.content = "Tampered";
  if (change === "binding") version.safe_metadata.workspace_id = "foreign";
  if (change === "missing-exact")
    vi.mocked(tldwClient.getMediaDocumentVersion).mockRejectedValue(
      new Error("Deleted"),
    );
  if (change === "changed-uuid")
    vi.mocked(tldwClient.getMediaDocumentVersion).mockResolvedValue({
      ...version,
      uuid: "66d2295a-7165-4e28-bbb0-b488a2bc404e",
    });
  if (change === "pagination")
    vi.mocked(tldwClient.listMediaDocumentVersions).mockRejectedValue(
      new Error("Page unavailable"),
    );
  if (change === "status-binding")
    status.workspace_placements[0].workspace_id = "foreign";
  await restore();
  expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
    status: "error",
    statusDetails: { statusReason: "capture_unavailable" },
  });
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
});

it("does not apply a recovered pin after owner scope invalidation", async () => {
  const { tldwClient, version } = await intactCapture();
  vi.mocked(tldwClient.getMediaDocumentVersion).mockImplementation(async () => {
    boundary.controller.abort();
    return version;
  });
  const apply = vi.fn();
  await expect(restore(apply)).rejects.toThrow();
  expect(apply).not.toHaveBeenCalled();
});

it("retains a known owned pin after checkpoint loss even when its exact version becomes unavailable", async () => {
  const { tldwClient } = await intactCapture();
  await restore();
  const pin = useWorkspaceStore.getState().sources[0].webCapture;
  expect(pin).toBeDefined();
  vi.mocked(tldwClient.getMediaDocumentVersion).mockRejectedValue(
    new Error("Deleted version"),
  );
  await restore();
  expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
    webCapture: pin,
    status: "error",
    statusDetails: { statusReason: "capture_unavailable" },
  });
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
});

it("reads every history page without treating a later edit as accepted", async () => {
  const { version, tldwClient } = await intactCapture();
  const page = Array.from({ length: 10 }, (_, i) => ({
    ...version,
    version_number: 11 - i,
    safe_metadata: null,
    content: "Later edit",
  }));
  vi.mocked(tldwClient.listMediaDocumentVersions)
    .mockResolvedValueOnce(page)
    .mockResolvedValueOnce([version]);
  await restore();
  expect(
    useWorkspaceStore.getState().sources[0].webCapture?.versionNumber,
  ).toBe(1);
  expect(tldwClient.listMediaDocumentVersions).toHaveBeenCalledWith(
    7,
    expect.anything(),
    { limit: 10, page: 2 },
  );
});

it("keeps an authoritative ordinary clip usable without inventing capture semantics", async () => {
  const { version } = await intactCapture();
  delete (version.safe_metadata as Record<string, unknown>).capture_metadata;
  await restore();
  expect(useWorkspaceStore.getState().sources[0].webCapture).toBeUndefined();
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([7]);
});

it("keeps unavailable capture blocked after ordinary Media readiness reconciliation", async () => {
  const { tldwClient } = await intactCapture();
  vi.mocked(tldwClient.getMediaDocumentVersion).mockRejectedValue(
    new Error("Deleted"),
  );
  await restore();
  useWorkspaceStore.getState().setSourceStatusByMediaId(7, "ready");
  expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
    status: "error",
    statusDetails: { statusReason: "capture_unavailable" },
  });
  useWorkspaceStore
    .getState()
    .toggleSourceSelection(useWorkspaceStore.getState().sources[0].id);
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
});

it.each(["owner", "workspace", "source", "media", "url"])(
  "never retains a previous pin across %s identity changes after checkpoint loss",
  async (change) => {
    const { ctx, version } = await intactCapture();
    await restore();
    const previous = useWorkspaceStore.getState().sources[0];
    if (change === "owner")
      useWorkspaceStore.setState({
        sources: [{ ...previous, captureOwnerScope: "foreign" }],
      });
    if (change === "workspace")
      useWorkspaceStore.setState({
        workspaceId: "foreign",
        workspaceSnapshots: {},
      });
    if (change === "source")
      useWorkspaceStore.setState({ sources: [{ ...previous, id: "foreign" }] });
    if (change === "media")
      useWorkspaceStore.setState({ sources: [{ ...previous, mediaId: 900 }] });
    if (change === "url")
      useWorkspaceStore.setState({
        sources: [{ ...previous, url: "https://example.org/foreign" }],
      });
    delete (version.safe_metadata as Record<string, unknown>).capture_metadata;
    await restore();
    expect(useWorkspaceStore.getState().sources[0].webCapture).toBeUndefined();
    expect(useWorkspaceStore.getState().sources[0].id).toBe(
      ctx.sources.items[0].id,
    );
  },
);

it.each(["missing-metadata", "malformed-descriptor", "overlapping-pages"])(
  "blocks a known capture when %s cannot establish its history",
  async (change) => {
    const { version, tldwClient } = await intactCapture();
    if (change === "missing-metadata")
      (version as { safe_metadata: unknown }).safe_metadata = null;
    if (change === "malformed-descriptor")
      version.safe_metadata.capture_metadata.web_capture_v1.content_sha256 =
        "invalid";
    if (change === "overlapping-pages")
      vi.mocked(tldwClient.listMediaDocumentVersions).mockResolvedValue(
        Array.from({ length: 10 }, (_, i) => ({
          ...version,
          version_number: 10 - i,
          safe_metadata: null,
        })),
      );
    await restore();
    expect(
      useWorkspaceStore.getState().sources[0].statusDetails?.statusReason,
    ).toBe("capture_unavailable");
    expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
  },
);

it("retains known unavailable capture semantics when later history loses its descriptor", async () => {
  const { version } = await intactCapture();
  version.content = "Tampered";
  await restore();
  expect(
    useWorkspaceStore.getState().sources[0].statusDetails?.statusReason,
  ).toBe("capture_unavailable");
  delete (version.safe_metadata as Record<string, unknown>).capture_metadata;
  await restore();
  expect(
    useWorkspaceStore.getState().sources[0].statusDetails?.statusReason,
  ).toBe("capture_unavailable");
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
});


it("allows ordinary clip recovery after a transient history lookup failure", async () => {
  const { tldwClient, version } = await intactCapture();
  delete (version.safe_metadata as Record<string, unknown>).capture_metadata;
  vi.mocked(tldwClient.listMediaDocumentVersions).mockRejectedValueOnce(
    new Error("Unavailable"),
  );
  await restore();
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
  await restore();
  expect(useWorkspaceStore.getState().sources[0].webCapture).toBeUndefined();
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([7]);
});


it.each(["queryable", "missing_media"])(
  "clears stale capture unavailability after exact recovery using authoritative %s status",
  async (state) => {
    const { ctx, version, tldwClient } = await intactCapture();
    vi.mocked(tldwClient.getMediaDocumentVersion).mockRejectedValueOnce(
      new Error("Temporarily unavailable"),
    );
    await restore();
    expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
      status: "error",
      statusDetails: { statusReason: "capture_unavailable" },
      captureOwnerScope: boundary.captureOwner,
    });
    expect(useWorkspaceStore.getState().sources[0].webCapture).toBeUndefined();
    ctx.sources.items[0].state = state;
    await restore();
    const recovered = useWorkspaceStore.getState().sources[0];
    expect(recovered.webCapture).toMatchObject({
      versionNumber: 1,
      versionUuid: version.uuid,
    });
    expect(recovered.status).toBe(state === "queryable" ? "ready" : "error");
    expect(recovered.statusDetails).toBeUndefined();
    expect(recovered.statusMessage).toBeUndefined();
    expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual(
      state === "queryable" ? [7] : [],
    );
    if (state === "queryable") {
      const preview = vi
        .spyOn(tldwClient, "getWorkspaceSourcePreview")
        .mockResolvedValue({
          document_version_number: 1,
          text_preview: "Original accepted article",
          content_available: true,
          snippets: [],
        } as Awaited<ReturnType<typeof tldwClient.getWorkspaceSourcePreview>>);
      const { SourcesPane } = await import("../SourcesPane");
      render(createElement(SourcesPane));
      fireEvent.click(screen.getByTestId(`preview-source-${recovered.id}`));
      expect(
        (await screen.findByText("Original accepted article")).textContent,
      ).toBe("Original accepted article");
      expect(preview).toHaveBeenCalledWith(
        "original",
        recovered.id,
        expect.objectContaining({ version_number: 1 }),
        expect.objectContaining({
          requestScope: expect.objectContaining({ userId: "alice" }),
        }),
      );
    }
  },
);

it("preserves intentional changed-head refusal when canonical recovery verifies the existing pin", async () => {
  await intactCapture();
  await restore();
  const source = useWorkspaceStore.getState().sources[0];
  useWorkspaceStore
    .getState()
    .setSourceStatusById(
      source.id,
      "error",
      "Snapshot changed outside refresh",
      undefined,
      { statusReason: "capture_head_changed", retryEligible: false },
    );
  await restore();
  expect(useWorkspaceStore.getState().sources[0]).toMatchObject({
    webCapture: source.webCapture,
    status: "error",
    statusMessage: "Snapshot changed outside refresh",
    statusDetails: { statusReason: "capture_head_changed" },
  });
  expect(useWorkspaceStore.getState().getSelectedMediaIds()).toEqual([]);
});
