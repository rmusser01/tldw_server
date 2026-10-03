import { beforeEach, describe, expect, it, vi } from "vitest";

const boundary = vi.hoisted(() => ({
  request: vi.fn(),
  controller: new AbortController(),
  release: vi.fn(),
  scopeKey: "original-alice",
  userId: "alice",
}));
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }));
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    scopeKey: boundary.scopeKey,
    requestScope: {
      config: { serverUrl: "https://original.test", authMode: "multi-user" },
      userId: boundary.userId,
    },
    scopeSignal: boundary.controller.signal,
    scopeInvalidatedSignal: boundary.controller.signal,
    release: boundary.release,
  }),
}));

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
  partial_errors: [],
});
const restore = (apply = useWorkspaceStore.getState().restoreServerWorkspace) =>
  restoreMigratedResearchWorkspace({
    signal: new AbortController().signal,
    apply,
  });

beforeEach(() => {
  localStorage.clear();
  useWorkspaceStore.getState().reset();
  boundary.controller = new AbortController();
  boundary.scopeKey = "original-alice";
  boundary.userId = "alice";
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
