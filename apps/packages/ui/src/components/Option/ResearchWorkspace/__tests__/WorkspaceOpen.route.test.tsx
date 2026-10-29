import React from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

const boundary = vi.hoisted(() => ({
  request: vi.fn(),
  api: { listWorkspaces: vi.fn(), getWorkspaceContext: vi.fn() },
  controller: new AbortController(),
  release: vi.fn(),
  userId: "1",
  scopeKey: "alice-open-scope",
}));
vi.mock("@/hooks/useTldwApiClient", () => ({
  useTldwApiClient: () => boundary.api,
}));
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }));
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    scopeKey: boundary.scopeKey,
    requestScope: {
      config: {
        serverUrl: "https://open.example.test",
        authMode: "multi-user",
      },
      userId: boundary.userId,
    },
    scopeSignal: boundary.controller.signal,
    scopeInvalidatedSignal: boundary.controller.signal,
    release: boundary.release,
  }),
}));
vi.mock("@/utils/research-workspace-prefill", async (original) => ({
  ...(await original<typeof import("@/utils/research-workspace-prefill")>()),
  readResearchWebCaptures: async () => [],
}));
// Inspect the real store at the production route boundary; full pane rendering is a separate live check.
vi.mock("../index", async () => {
  const { useWorkspaceStore } = await import("@/store/workspace");
  const { useLocation } = await import("react-router-dom");
  return {
    ResearchWorkspace: () => {
      const state = useWorkspaceStore();
      const location = useLocation();
      return (
        <>
          <h1>{state.workspaceName || "No active workspace"}</h1>
          <output data-testid="opened-sources">
            {state.sources.map((source) => source.id).join(",")}
          </output>
          <output data-testid="open-location">{location.search}</output>
        </>
      );
    },
  };
});

import {
  createEmptyWorkspaceSnapshot,
  useWorkspaceStore,
} from "@/store/workspace";
import { WorkspacesManagerPage } from "../../Workspaces/WorkspacesManagerPage";
import { ResearchWorkspaceRouteGate } from "../ResearchWorkspaceRouteGate";

const TARGET = "workspace-A";
const PREVIOUS = "workspace-B";
const serverWorkspace = {
  id: TARGET,
  name: "Alpha",
  archived: false,
  deleted: false,
  workspace_profile: "research",
  study_materials_policy: "workspace",
  created_at: "2026-10-01T00:00:00Z",
  last_modified: "2026-10-02T00:00:00Z",
  version: 1,
  banner_title: null,
  banner_subtitle: null,
  banner_color: null,
  audio_provider: null,
  audio_model: null,
  audio_voice: null,
  audio_speed: null,
};
const canonical = () => ({
  workspace_id: TARGET,
  workspace: serverWorkspace,
  workspace_profile: "research",
  workspace_kind: "research_workspace",
  schema_version: 2,
  generated_at: "2026-10-03T00:00:00Z",
  attention_state: "ready",
  resolution: { status: "complete", partial_errors: [] },
  project_root: {
    state: "not_configured",
    root_id: null,
    backend: null,
    display_name: null,
    path_hint: null,
    git_state: null,
    file_inventory_state: "not_started",
    file_inventory: {
      state: "not_started",
      indexed_file_count: null,
      total_file_count: null,
      updated_at: null,
      available: false,
    },
    indexing_state: null,
    sandbox_mount_state: null,
    mcp_trust_state: null,
  },
  sources: {
    items: [
      {
        id: "source-A",
        workspace_id: TARGET,
        media_id: 11,
        title: "Alpha source",
        source_type: "text",
        selected: true,
        added_at: "2026-10-01T00:00:00Z",
        version: 1,
        state: "queryable",
        readiness: { fts_ready: true, text_extracted: true },
      },
    ],
    summary: {
      total: 1,
      selected: 1,
      queryable: 1,
      partially_queryable: 0,
      processing: 0,
      failed: 0,
      missing: 0,
    },
  },
  capabilities: {},
  services: {},
  allowed_actions: {},
  active_jobs: [],
  active_operations: [],
  partial_errors: [] as Array<{ code: string }>,
});
const reply = async ({ path }: { path: string }) => {
  if (path === `/api/v1/workspaces/${TARGET}/context`) return canonical();
  if (
    path === `/api/v1/workspaces/${TARGET}/artifacts` ||
    path === `/api/v1/workspaces/${TARGET}/notes`
  )
    return [];
  if (path.startsWith("/api/v1/notes/search/")) return { notes: [] };
  throw new Error("Unexpected canonical workspace request");
};
const snapshot = (id: string, name: string) =>
  createEmptyWorkspaceSnapshot({
    id,
    name,
    tag: `workspace:${name.toLowerCase()}`,
    createdAt: new Date("2026-10-01T00:00:00Z"),
  });
const prepare = (cached: boolean) => {
  if (cached) {
    const alpha = snapshot(TARGET, "Alpha");
    alpha.sources = [
      {
        id: "source-A",
        mediaId: 11,
        title: "Alpha source",
        type: "text",
        status: "ready",
        addedAt: new Date(),
      },
    ];
    alpha.selectedSourceIds = ["source-A"];
    useWorkspaceStore.getState().restoreServerWorkspace(alpha);
  }
  useWorkspaceStore
    .getState()
    .restoreServerWorkspace(snapshot(PREVIOUS, "Beta"));
  useWorkspaceStore.getState().setCurrentNote({
    title: "Beta draft",
    content: "Keep unsaved Beta",
    keywords: [],
    isDirty: true,
  });
};
const mount = async (entry = "/workspaces") => {
  const view = render(
    <MemoryRouter initialEntries={[entry]}>
      <Routes>
        <Route path="/workspaces" element={<WorkspacesManagerPage />} />
        <Route
          path="/research-workspace"
          element={<ResearchWorkspaceRouteGate />}
        />
      </Routes>
    </MemoryRouter>,
  );
  await act(async () => {
    await vi.dynamicImportSettled();
  });
  return view;
};

beforeEach(() => {
  localStorage.clear();
  useWorkspaceStore.getState().reset();
  boundary.userId = "1";
  boundary.scopeKey = "alice-open-scope";
  boundary.controller = new AbortController();
  boundary.release.mockReset();
  boundary.request.mockReset().mockImplementation(reply);
  boundary.api.listWorkspaces
    .mockReset()
    .mockResolvedValue({ items: [serverWorkspace], total: 1 });
  boundary.api.getWorkspaceContext.mockReset().mockResolvedValue(canonical());
});
afterEach(() => {
  cleanup();
  useWorkspaceStore.getState().reset();
});

it.each([true, false])(
  "opens requested canonical workspace through manager with cached target=%s",
  async (cached) => {
    prepare(cached);
    await mount();
    fireEvent.click(await screen.findByRole("button", { name: "Open Alpha" }));
    await act(async () => {
      await vi.dynamicImportSettled();
    });
    await waitFor(() =>
      expect(useWorkspaceStore.getState().workspaceId).toBe(TARGET),
    );
    expect(await screen.findByTestId("open-location")).toHaveTextContent(
      `source_workspace_id=${TARGET}`,
    );
    expect(await screen.findByRole("heading", { name: "Alpha" })).toBeVisible();
    expect(await screen.findByTestId("opened-sources")).toHaveTextContent(
      "source-A",
    );
    expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([
      "source-A",
    ]);
    expect(
      useWorkspaceStore.getState().workspaceSnapshots[PREVIOUS].currentNote
        .content,
    ).toBe("Keep unsaved Beta");
    expect(
      boundary.request.mock.calls.every(
        ([request]) => request.method === "GET",
      ),
    ).toBe(true);
  },
);

it("opens a direct server-only target without initializing another workspace", async () => {
  prepare(false);
  await mount(`/research-workspace?source_workspace_id=${TARGET}`);
  await waitFor(() =>
    expect(useWorkspaceStore.getState().workspaceId).toBe(TARGET),
  );
  expect(await screen.findByTestId("opened-sources")).toHaveTextContent(
    "source-A",
  );
  expect(
    useWorkspaceStore.getState().workspaceSnapshots[PREVIOUS].currentNote
      .content,
  ).toBe("Keep unsaved Beta");
});

it("keeps the current workspace and draft when the target is unavailable", async () => {
  prepare(false);
  boundary.request.mockRejectedValue(
    Object.assign(new Error("Unavailable"), { status: 404 }),
  );
  await mount(`/research-workspace?source_workspace_id=${TARGET}`);
  await screen.findByRole("alert");
  expect(useWorkspaceStore.getState().workspaceId).toBe(PREVIOUS);
  expect(useWorkspaceStore.getState().currentNote.content).toBe(
    "Keep unsaved Beta",
  );
  expect(
    useWorkspaceStore.getState().workspaceSnapshots[TARGET],
  ).toBeUndefined();
});

it.each(["account", "workspace", "unmount"] as const)(
  "retires held canonical target after %s changes",
  async (kind) => {
    prepare(false);
    let finish!: () => void;
    boundary.request.mockImplementation((request: { path: string }) =>
      request.path.endsWith("/context")
        ? new Promise((resolve) => {
            finish = () => resolve(canonical());
          })
        : reply(request),
    );
    const view = await mount(
      `/research-workspace?source_workspace_id=${TARGET}`,
    );
    await waitFor(() => expect(finish).toBeTypeOf("function"));
    if (kind === "account") boundary.controller.abort();
    else if (kind === "workspace")
      useWorkspaceStore
        .getState()
        .restoreServerWorkspace(snapshot("workspace-C", "Gamma"));
    else view.unmount();
    await act(async () => {
      finish();
      await Promise.resolve();
    });
    await waitFor(() => expect(boundary.release).toHaveBeenCalledOnce());
    expect(useWorkspaceStore.getState().workspaceId).toBe(
      kind === "workspace" ? "workspace-C" : PREVIOUS,
    );
    expect(
      useWorkspaceStore.getState().workspaceSnapshots[TARGET],
    ).toBeUndefined();
  },
);

it.each([
  "",
  "workspace-A&source_workspace_id=workspace-B",
  "%20",
  "x".repeat(129),
])(
  "rejects an invalid target without mounting or loading a workspace: %s",
  async (target) => {
    prepare(false);
    await mount("/research-workspace?source_workspace_id=" + target);
    await screen.findByRole("alert");
    expect(boundary.request).not.toHaveBeenCalled();
    expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
    expect(useWorkspaceStore.getState().currentNote.content).toBe(
      "Keep unsaved Beta",
    );
  },
);

it("opens the target through Retry after a temporary refusal, without losing the previous draft", async () => {
  prepare(false);
  boundary.request.mockRejectedValueOnce(
    Object.assign(new Error("Unavailable"), { status: 403 }),
  );
  await mount("/research-workspace?source_workspace_id=" + TARGET);
  await screen.findByRole("alert");
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByTestId("opened-sources")).toHaveTextContent(
    "source-A",
  );
  expect(
    useWorkspaceStore.getState().workspaceSnapshots[PREVIOUS].currentNote
      .content,
  ).toBe("Keep unsaved Beta");
});

it.each(["partial", "foreign", "archived", "deleted"])(
  "does not apply an incomplete or inaccessible canonical target: %s",
  async (kind) => {
    prepare(true);
    const value = canonical();
    if (kind === "partial")
      value.partial_errors = [{ code: "sources_unavailable" }];
    else if (kind === "foreign") value.workspace_id = "workspace-foreign";
    else value.workspace[kind as "archived" | "deleted"] = true;
    boundary.request.mockImplementation((request: { path: string }) =>
      request.path.endsWith("/context")
        ? Promise.resolve(value)
        : reply(request),
    );
    await mount("/research-workspace?source_workspace_id=" + TARGET);
    await screen.findByRole("alert");
    expect(useWorkspaceStore.getState().workspaceId).toBe(PREVIOUS);
    expect(useWorkspaceStore.getState().currentNote.content).toBe(
      "Keep unsaved Beta",
    );
    expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
  },
);

it.each(["cached", "active", "other-account"])(
  "uses canonical target fields instead of unbound %s data",
  async (kind) => {
    prepare(true);
    useWorkspaceStore.getState().switchWorkspace(TARGET);
    useWorkspaceStore.getState().setCurrentNote({
      title: "Old saved note",
      content: "Old private content",
      keywords: [],
      isDirty: false,
    });
    if (kind !== "active")
      useWorkspaceStore.getState().switchWorkspace(PREVIOUS);
    if (kind === "other-account") {
      boundary.userId = "2";
      boundary.scopeKey = "bob-open-scope";
    }
    const value = canonical();
    value.workspace.name = "Fresh canonical Alpha";
    value.sources.items = [];
    boundary.request.mockImplementation((request: { path: string }) =>
      request.path.endsWith("/context")
        ? Promise.resolve(value)
        : reply(request),
    );
    await mount("/research-workspace?source_workspace_id=" + TARGET);
    expect(
      await screen.findByRole("heading", { name: "Fresh canonical Alpha" }),
    ).toBeVisible();
    expect(useWorkspaceStore.getState().sources).toEqual([]);
    expect(useWorkspaceStore.getState().selectedSourceIds).toEqual([]);
    expect(useWorkspaceStore.getState().currentNote.content).not.toContain(
      "Old private content",
    );
  },
);

it.each(["cached", "active"])(
  "keeps an unbound dirty %s target intact instead of replacing its draft",
  async (kind) => {
    prepare(true);
    useWorkspaceStore.getState().switchWorkspace(TARGET);
    useWorkspaceStore.getState().setCurrentNote({
      title: "Alpha draft",
      content: "Keep unsaved Alpha",
      keywords: [],
      isDirty: true,
    });
    if (kind === "cached")
      useWorkspaceStore.getState().switchWorkspace(PREVIOUS);
    await mount("/research-workspace?source_workspace_id=" + TARGET);
    await screen.findByRole("alert");
    const state = useWorkspaceStore.getState();
    expect(state.workspaceId).toBe(kind === "active" ? TARGET : PREVIOUS);
    expect(
      (kind === "active"
        ? state.currentNote
        : state.workspaceSnapshots[TARGET].currentNote
      ).content,
    ).toBe("Keep unsaved Alpha");
    expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
  },
);

it("keeps an unverified cached Chat instead of replacing or displaying it", async () => {
  prepare(true);
  useWorkspaceStore.getState().saveWorkspaceChatSession(TARGET, {
    messages: [],
    history: [],
    historyId: null,
    serverChatId: "unverified-chat",
  });
  await mount("/research-workspace?source_workspace_id=" + TARGET);
  await screen.findByRole("alert");
  expect(useWorkspaceStore.getState().workspaceId).toBe(PREVIOUS);
  expect(
    useWorkspaceStore.getState().getWorkspaceChatSession(TARGET)?.serverChatId,
  ).toBe("unverified-chat");
  expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
});

it.each([true, false, "history-id"])(
  "validates receipt Chat ownership without discarding local work=%s",
  async (hasLocalMessage) => {
    prepare(true);
    const cachedChat = {
      messages:
        hasLocalMessage === true
          ? [
              {
                isBot: true,
                name: "AI",
                message: "Keep unacknowledged answer",
                sources: [],
              },
            ]
          : [],
      history: [],
      historyId: hasLocalMessage === "history-id" ? "keep-local-history" : null,
      serverChatId: "verified-chat",
    };
    useWorkspaceStore.getState().saveWorkspaceChatSession(TARGET, cachedChat);
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-A",
      JSON.stringify({
        serverWorkspaceId: TARGET,
        migrationId: "valid-migration",
        serverScopeKey: "alice-open-scope",
        contentRetained: false,
        deletedAt: "2026-10-01T00:00:00Z",
        chatReference: { referenceId: TARGET, serverChatId: "verified-chat" },
      }),
    );
    boundary.request.mockImplementation((request: { path: string }) =>
      request.path.startsWith("/api/v1/chats/")
        ? Promise.resolve({
            id: "verified-chat",
            scope_type: "workspace",
            workspace_id: TARGET,
            deleted: false,
          })
        : reply(request),
    );
    await mount("/research-workspace?source_workspace_id=" + TARGET);
    if (hasLocalMessage) {
      await screen.findByRole("alert");
      expect(useWorkspaceStore.getState().workspaceId).toBe(PREVIOUS);
      expect(
        useWorkspaceStore.getState().getWorkspaceChatSession(TARGET),
      ).toEqual(cachedChat);
      expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
    } else {
      expect(await screen.findByTestId("opened-sources")).toHaveTextContent(
        "source-A",
      );
      expect(
        useWorkspaceStore.getState().getWorkspaceChatSession(TARGET)
          ?.serverChatId,
      ).toBe("verified-chat");
    }
  },
);

it("keeps the existing Deep Research return route and active Chat intact", async () => {
  prepare(true);
  useWorkspaceStore.getState().switchWorkspace(TARGET);
  const session = {
    messages: [
      { isBot: true, name: "AI", message: "Keep current answer", sources: [] },
    ],
    history: [],
    historyId: "current-local-history",
    serverChatId: "current-chat",
  };
  useWorkspaceStore.getState().saveWorkspaceChatSession(TARGET, session);
  await mount(
    "/research-workspace?source_workspace_id=" +
      TARGET +
      "&research_run_id=run-123",
  );
  expect(await screen.findByRole("heading", { name: "Alpha" })).toBeVisible();
  expect(await screen.findByTestId("open-location")).toHaveTextContent(
    "research_run_id=run-123",
  );
  expect(boundary.request).not.toHaveBeenCalled();
  expect(useWorkspaceStore.getState().getWorkspaceChatSession(TARGET)).toEqual(
    session,
  );
});

it("does not treat an invalid Deep Research flag as an ordinary Open bypass", async () => {
  prepare(false);
  boundary.request.mockRejectedValue(
    Object.assign(new Error("Unavailable"), { status: 403 }),
  );
  await mount(
    "/research-workspace?source_workspace_id=" + TARGET + "&research_run_id=",
  );
  await screen.findByRole("alert");
  expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
  expect(useWorkspaceStore.getState().workspaceId).toBe(PREVIOUS);
});
