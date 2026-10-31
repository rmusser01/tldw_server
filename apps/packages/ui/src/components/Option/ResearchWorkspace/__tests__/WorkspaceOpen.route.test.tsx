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
  flushWorkspacePersistence,
  useWorkspaceStore,
} from "@/store/workspace";
import { tldwClient } from "@/services/tldw/TldwApiClient";
import { mapServerChatMessagesToPlaygroundMessages } from "@/hooks/chat/useServerChatLoader";
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
  workspace: { ...serverWorkspace },
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

it.each(["retained-hint", "pre-hint"])("opens a saved canonical Chat after real persistence rehydration through the requested route (%s)", async (hint) => {
  prepare(true);
  useWorkspaceStore.getState().switchWorkspace(TARGET);
  const serverMessages = [{
    id: "canonical-answer", role: "assistant" as const, sender: "AI",
    content: "Canonical cited answer", created_at: "2026-10-31T00:00:00Z", version: 1,
  }];
  const messages = mapServerChatMessagesToPlaygroundMessages({ serverMessages, assistantName: "AI", characterId: null });
  useWorkspaceStore.getState().saveWorkspaceChatSession(TARGET, {
    messages, history: [], historyId: null, serverChatId: "verified-chat",
  }, { scopeKey: boundary.scopeKey, isCurrent: () => !boundary.controller.signal.aborted });
  await flushWorkspacePersistence();
  await useWorkspaceStore.persist.rehydrate();
  expect(useWorkspaceStore.getState().getWorkspaceChatSession(TARGET)?.messages).toEqual(messages);
  if (hint === "pre-hint") {
    const key = Object.keys(localStorage).find(key => key.startsWith("tldw:research-workspace:chat-reference:"))!;
    localStorage.removeItem(key);
  }
  boundary.request.mockImplementation((request: { path: string }) =>
    request.path.startsWith("/api/v1/chats/verified-chat/messages")
      ? Promise.resolve({ messages: serverMessages, total: 1 })
      : request.path.startsWith("/api/v1/chats/verified-chat?")
        ? Promise.resolve({ id: "verified-chat", scope_type: "workspace", workspace_id: TARGET, deleted: false })
        : reply(request));
  const list = vi.spyOn(tldwClient, "listChatMessages");
  await mount("/research-workspace?source_workspace_id=" + TARGET);
  await waitFor(() => expect(list).toHaveBeenCalledWith("verified-chat",
    expect.objectContaining({ include_metadata: "true", include_images: "true", render_placeholders: "false" }),
    expect.objectContaining({ fresh: true,
      scope: { type: "workspace", workspaceId: TARGET },
      requestScope: expect.objectContaining({ userId: "1" }) })));
  expect(boundary.request.mock.calls.some(([request]) => {
    const url = new URL(request.path, "https://open.example.test");
    return url.pathname === "/api/v1/chats/verified-chat/messages" &&
      url.searchParams.get("scope_type") === "workspace" && url.searchParams.get("workspace_id") === TARGET;
  })).toBe(true);
  const loaded = await list.mock.results[0].value;
  expect(mapServerChatMessagesToPlaygroundMessages({ serverMessages: loaded, assistantName: "AI", characterId: null })).toEqual(messages);
  expect(await screen.findByTestId("opened-sources")).toHaveTextContent("source-A");
  expect(useWorkspaceStore.getState().getWorkspaceChatSession(TARGET)?.serverChatId).toBe("verified-chat");
});

it.each([
  "body", "role", "version", "metadata", "sources", "images", "variants",
  "unknown", "order", "duplicate", "history", "history-id",
  "concurrent-cache", "concurrent-reference", "concurrent-account", "outage",
  "absent-clear", "absent-new-reference", "absent-tombstone-clear",
])("retains local Workspace Chat work after %s differs from canonical data", async (change) => {
  prepare(true);
  useWorkspaceStore.getState().switchWorkspace(TARGET);
  const serverMessages = [
    { id: "canonical-user", role: "user" as const, content: "Saved question",
      created_at: "2026-10-31T00:00:00Z", version: 1 },
    { id: "canonical-answer", role: "assistant" as const, sender: "AI",
      content: "Saved answer", created_at: "2026-10-31T00:00:01Z", version: 1,
      metadata_extra: { audit: "canonical" } },
  ];
  const messages = mapServerChatMessagesToPlaygroundMessages({ serverMessages, assistantName: "AI", characterId: null });
  if (change === "body") messages[1].message = "Keep edited answer";
  if (change === "role") messages[1].isBot = false;
  if (change === "version") messages[1].serverMessageVersion = 2;
  if (change === "metadata") messages[1].metadataExtra = { audit: "Keep local metadata" };
  if (change === "sources") Object.assign(messages[1], { sources: [{ title: "Keep local citation", content: "Private" }] });
  if (change === "images") messages[1].images = ["data:image/png;base64,bG9jYWw="];
  if (change === "variants") Object.assign(messages[1], { variants: [{ message: "Keep alternate answer" }] });
  if (change === "unknown") Object.assign(messages[1], { id: "unknown-local", serverMessageId: undefined });
  if (change === "order") messages.reverse();
  if (change === "duplicate") messages[1].serverMessageId = messages[0].serverMessageId;
  const session = { messages, history: [], historyId: null, serverChatId: "verified-chat" };
  if (change === "history-id") Object.assign(session, { historyId: "unverified-local-mirror" });
  const save = useWorkspaceStore.getState().saveWorkspaceChatSession;
  save(TARGET, session, { scopeKey: boundary.scopeKey, isCurrent: () => !boundary.controller.signal.aborted });
  await flushWorkspacePersistence();
  await useWorkspaceStore.persist.rehydrate();
  let preserved = useWorkspaceStore.getState().getWorkspaceChatSession(TARGET);
  const hintKey = Object.keys(localStorage).find(key => key.startsWith("tldw:research-workspace:chat-reference:"))!;
  if (change.startsWith("absent-")) localStorage.removeItem(hintKey);
  if (change === "history") {
    preserved = { ...preserved!, history: [{ role: "system", content: "Keep local instruction" }] };
    save(TARGET, preserved);
  }
  boundary.request.mockImplementation((request: { path: string }) => {
    if (request.path.startsWith("/api/v1/chats/verified-chat/messages")) {
      if (change === "outage") throw new Error("Controlled canonical read failure");
      if (change.startsWith("absent-")) return Promise.resolve().then(() => {
        if (change === "absent-tombstone-clear") localStorage.setItem(
          "tldw:research-workspace:migration:tombstone:" + TARGET,
          JSON.stringify({ legacyWorkspaceId: TARGET, serverWorkspaceId: TARGET,
            migrationId: "new-acknowledged-migration", serverScopeKey: boundary.scopeKey,
            contentRetained: false, deletedAt: "2026-10-31T00:00:00Z",
            chatReference: { referenceId: TARGET, serverChatId: null } }));
        else localStorage.setItem(hintKey, JSON.stringify({ referenceId: TARGET,
          serverChatId: change === "absent-clear" ? null : "newer-selected-chat" }));
        return { messages: serverMessages, total: 2 };
      });
      if (change === "concurrent-cache") {
        preserved = { ...preserved!, messages: preserved!.messages.map((message, index) =>
          index === 1 ? { ...message, message: "Keep newer concurrent edit" } : message) };
        save(TARGET, preserved);
      }
      if (change === "concurrent-reference") {
        useWorkspaceStore.setState({ workspaceChatReferenceId: "newer-reference" });
        save(TARGET + "::newer-reference", preserved!);
      }
      if (change === "concurrent-account") boundary.controller.abort();
      return Promise.resolve({ messages: serverMessages, total: 2 });
    }
    if (request.path.startsWith("/api/v1/chats/verified-chat?"))
      return Promise.resolve({ id: "verified-chat", scope_type: "workspace", workspace_id: TARGET, deleted: false });
    return reply(request);
  });
  await mount("/research-workspace?source_workspace_id=" + TARGET);
  await screen.findByRole("alert");
  expect(screen.queryByTestId("opened-sources")).not.toBeInTheDocument();
  expect(useWorkspaceStore.getState().getWorkspaceChatSession(TARGET)).toEqual(preserved);
});

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
