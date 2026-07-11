import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SOURCE_LIST_VIEW_STATE } from "../SourcesPane/source-list-view";

const api = vi.hoisted(() => ({
  listWorkspaceSourceViews: vi.fn(),
  createWorkspaceSourceView: vi.fn(),
  updateWorkspaceSourceView: vi.fn(),
  deleteWorkspaceSourceView: vi.fn(),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: api }));

import { useSourceSavedViews } from "../SourcesPane/use-source-saved-views";

const wireState = (overrides: Record<string, unknown> = {}) => ({
  type_filters: [],
  status_filters: [],
  review_state_filters: [],
  lifecycle_state_filters: [],
  date_field: "added_at" as const,
  date_from: null,
  date_to: null,
  require_url: false,
  require_file_size: false,
  require_duration: false,
  require_page_count: false,
  file_size_min: null,
  file_size_max: null,
  duration_min: null,
  duration_max: null,
  page_count_min: null,
  page_count_max: null,
  sort: "manual" as const,
  ...overrides,
});

const validView = (overrides: Record<string, unknown> = {}) => ({
  id: "view-1",
  workspace_id: "ws-a",
  name: "Needs review",
  schema_version: 1,
  version: 2,
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z",
  state: wireState({ review_state_filters: ["needs_review"] }),
  valid: true as const,
  invalid_reason: null,
  ...overrides,
});

const invalidView = (overrides: Record<string, unknown> = {}) => ({
  id: "invalid-1",
  workspace_id: "ws-a",
  name: "Old view",
  schema_version: 2,
  version: 4,
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z",
  state: null,
  valid: false as const,
  invalid_reason: "unsupported_schema_version" as const,
  ...overrides,
});

const localState = (overrides: Record<string, unknown> = {}) => ({
  ...DEFAULT_SOURCE_LIST_VIEW_STATE,
  typeFilters: [...DEFAULT_SOURCE_LIST_VIEW_STATE.typeFilters],
  statusFilters: [...DEFAULT_SOURCE_LIST_VIEW_STATE.statusFilters],
  reviewStateFilters: [...DEFAULT_SOURCE_LIST_VIEW_STATE.reviewStateFilters],
  lifecycleStateFilters: [
    ...DEFAULT_SOURCE_LIST_VIEW_STATE.lifecycleStateFilters,
  ],
  ...overrides,
});

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
};

const setup = (
  workspaceId: string | null = "ws-a",
  state = localState(),
  onApplyState = vi.fn(),
) =>
  renderHook(
    ({ workspaceId, state }) =>
      useSourceSavedViews(workspaceId, state, onApplyState),
    { initialProps: { workspaceId, state } },
  );

describe("useSourceSavedViews", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [] });
  });

  it("is unavailable for null workspaces and never requests", () => {
    const { result } = setup(null);

    expect(result.current.available).toBe(false);
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(api.listWorkspaceSourceViews).not.toHaveBeenCalled();
  });

  it("loads workspace views and retries a nonblocking list failure", async () => {
    api.listWorkspaceSourceViews
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({ items: [validView()] });
    const { result } = setup();

    await waitFor(() => expect(result.current.listError?.retryable).toBe(true));
    expect(result.current.views).toEqual([]);
    await act(async () => result.current.retry());

    expect(result.current.views).toHaveLength(1);
    expect(result.current.listError).toBeNull();
    expect(api.listWorkspaceSourceViews).toHaveBeenNthCalledWith(1, "ws-a");
    expect(api.listWorkspaceSourceViews).toHaveBeenNthCalledWith(2, "ws-a");
  });

  it("applies a valid view and ignores an invalid view", async () => {
    const onApply = vi.fn();
    api.listWorkspaceSourceViews.mockResolvedValue({
      items: [validView(), invalidView()],
    });
    const { result } = setup("ws-a", localState({ expanded: true }), onApply);
    await waitFor(() => expect(result.current.views).toHaveLength(2));

    act(() => result.current.applyView(result.current.views[0]!));
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({
        expanded: true,
        reviewStateFilters: ["needs_review"],
      }),
    );
    expect(result.current.activeViewId).toBe("view-1");
    expect(result.current.activeSnapshot).toEqual(
      wireState({ review_state_filters: ["needs_review"] }),
    );

    act(() => result.current.applyView(result.current.views[1]!));
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(result.current.activeViewId).toBe("view-1");
  });

  it("canonicalizes a valid runtime response before applying and snapshotting", async () => {
    const onApply = vi.fn();
    api.listWorkspaceSourceViews.mockResolvedValue({
      items: [
        validView({
          state: wireState({ type_filters: ["website", "pdf", "website"] }),
        }),
      ],
    });
    const { result } = setup("ws-a", localState(), onApply);
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    act(() => result.current.applyView(result.current.views[0]!));

    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({ typeFilters: ["pdf", "website"] }),
    );
    expect(result.current.activeSnapshot?.type_filters).toEqual([
      "pdf",
      "website",
    ]);
  });

  it("creates from the canonical current state and announces success", async () => {
    const state = localState({ typeFilters: ["pdf"], sort: "name_asc" });
    api.createWorkspaceSourceView.mockResolvedValue(
      validView({
        name: "PDFs",
        state: wireState({ type_filters: ["pdf"], sort: "name_asc" }),
      }),
    );
    const { result, rerender } = setup("ws-a", state);
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalled(),
    );

    await act(async () => result.current.createView("  PDFs  "));

    expect(api.createWorkspaceSourceView).toHaveBeenCalledWith("ws-a", {
      name: "PDFs",
      schema_version: 1,
      state: wireState({ type_filters: ["pdf"], sort: "name_asc" }),
    });
    expect(result.current.activeViewId).toBe("view-1");
    expect(result.current.modified).toBe(false);
    expect(result.current.announcement).toBe("Saved view created.");

    rerender({ workspaceId: "ws-b", state });
    expect(result.current.announcement).toBeNull();
  });

  it("blocks invalid local state with field issues", async () => {
    const { result, rerender } = setup(
      "ws-a",
      localState({ fileSizeMin: 20, fileSizeMax: 10 }),
    );

    await act(async () => result.current.createView("Broken"));

    expect(api.createWorkspaceSourceView).not.toHaveBeenCalled();
    expect(result.current.serializationIssues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "fileSizeMax" }),
      ]),
    );

    rerender({ workspaceId: null, state: localState() });
    expect(result.current.serializationIssues).toEqual([]);
  });

  it("retains duplicate metadata and explicitly replaces with the server version", async () => {
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: {
        detail: {
          code: "source_view_name_exists",
          view_id: "view-1",
          version: 2,
        },
      },
    });
    api.updateWorkspaceSourceView.mockResolvedValue(validView({ version: 3 }));
    const { result } = setup("ws-a", localState({ typeFilters: ["pdf"] }));

    await act(async () => result.current.createView("PDFs"));
    expect(result.current.duplicateConflict).toEqual(
      expect.objectContaining({ viewId: "view-1", version: 2, name: "PDFs" }),
    );
    expect(api.updateWorkspaceSourceView).not.toHaveBeenCalled();

    await act(async () => result.current.confirmReplace());
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledWith(
      "ws-a",
      "view-1",
      {
        version: 2,
        name: "PDFs",
        schema_version: 1,
        state: wireState({ type_filters: ["pdf"] }),
      },
    );
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.activeViewId).toBe("view-1");
    expect(result.current.modified).toBe(false);
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("treats malformed duplicate detail as an ordinary retryable error", async () => {
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: { detail: { code: "source_view_name_exists", version: "2" } },
    });
    const { result } = setup();

    await act(async () => result.current.createView("Name"));

    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.mutationError?.retryable).toBe(true);
  });

  it.each([
    {
      code: "source_view_name_exists",
      view_id: "parent/view",
      version: 2,
    },
    { code: "source_view_limit_reached", limit: 99 },
  ])("treats malformed conflict detail as retryable: $code", async (detail) => {
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: { detail },
    });
    const { result } = setup();

    await act(async () => result.current.createView("Name"));

    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.limitState).toBeNull();
    expect(result.current.mutationError?.retryable).toBe(true);
  });

  it("exposes a nonretryable saved-view limit with deletion guidance", async () => {
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: { detail: { code: "source_view_limit_reached", limit: 100 } },
    });
    const { result, rerender } = setup();

    await act(async () => result.current.createView("Name"));

    expect(result.current.limitState).toEqual({
      limit: 100,
      retryable: false,
      guidance: "Delete an existing saved view before creating another.",
    });
    expect(result.current.mutationError).toBeNull();

    rerender({ workspaceId: "ws-b", state: localState() });
    expect(result.current.limitState).toBeNull();
  });

  it("refreshes rows on version conflict and requires an explicit retry", async () => {
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: {
        detail: {
          code: "source_view_name_exists",
          view_id: "view-1",
          version: 2,
        },
      },
    });
    api.updateWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: {
        detail: {
          code: "source_view_version_conflict",
          view_id: "view-1",
          current_version: 5,
        },
      },
    });
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ items: [validView({ version: 5 })] });
    const { result } = setup();

    await act(async () => result.current.createView("Name"));
    await act(async () => result.current.confirmReplace());

    expect(result.current.versionConflict).toEqual({
      viewId: "view-1",
      currentVersion: 5,
      retryable: true,
    });
    expect(result.current.views[0]?.version).toBe(5);
    expect(result.current.duplicateConflict?.version).toBe(5);
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(1);
  });

  it("retries a version conflict with the refreshed server version", async () => {
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: {
        detail: {
          code: "source_view_name_exists",
          view_id: "view-1",
          version: 2,
        },
      },
    });
    api.updateWorkspaceSourceView
      .mockRejectedValueOnce({
        status: 409,
        details: {
          detail: {
            code: "source_view_version_conflict",
            view_id: "view-1",
            current_version: 5,
          },
        },
      })
      .mockResolvedValueOnce(validView({ version: 6 }));
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ items: [validView({ version: 5 })] });
    const { result } = setup();

    await act(async () => result.current.createView("Name"));
    await act(async () => result.current.confirmReplace());
    await act(async () => result.current.retryVersionConflict());

    expect(api.updateWorkspaceSourceView).toHaveBeenLastCalledWith(
      "ws-a",
      "view-1",
      expect.objectContaining({ version: 5, name: "Name" }),
    );
    expect(result.current.versionConflict).toBeNull();
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("retries an ordinary mutation error but not a saved-view limit", async () => {
    api.createWorkspaceSourceView
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(validView({ name: "Retry me" }));
    const { result } = setup();

    await act(async () => result.current.createView("Retry me"));
    expect(result.current.mutationError?.retryable).toBe(true);
    expect(result.current.canRetryMutation).toBe(true);

    await act(async () => result.current.retryMutation());
    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(2);
    expect(result.current.mutationError).toBeNull();

    api.createWorkspaceSourceView.mockRejectedValueOnce({
      status: 409,
      details: { detail: { code: "source_view_limit_reached", limit: 100 } },
    });
    await act(async () => result.current.createView("At limit"));
    expect(result.current.limitState?.retryable).toBe(false);
    expect(result.current.canRetryMutation).toBe(false);
  });

  it("resets an invalid row to canonical V1 defaults and preserves expanded", async () => {
    const row = invalidView();
    const resetResponse = validView({
      id: row.id,
      name: row.name,
      version: 5,
      state: wireState(),
    });
    api.updateWorkspaceSourceView.mockResolvedValue(resetResponse);
    const onApply = vi.fn();
    const { result } = setup("ws-a", localState({ expanded: true }), onApply);

    await act(async () => result.current.resetView(row));

    expect(api.updateWorkspaceSourceView).toHaveBeenCalledWith("ws-a", row.id, {
      version: 4,
      schema_version: 1,
      state: wireState(),
    });
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({ expanded: true, sort: "manual" }),
    );
    expect(result.current.activeViewId).toBe(row.id);
    expect(result.current.modified).toBe(false);
    expect(result.current.announcement).toBe("Saved view reset.");
  });

  it("deletes the active row without changing current filters", async () => {
    const onApply = vi.fn();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.deleteWorkspaceSourceView.mockResolvedValue(undefined);
    const { result } = setup("ws-a", localState(), onApply);
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));

    await act(async () => result.current.deleteView(result.current.views[0]!));

    expect(api.deleteWorkspaceSourceView).toHaveBeenCalledWith(
      "ws-a",
      "view-1",
    );
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.activeSnapshot).toBeNull();
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(result.current.announcement).toBe("Saved view deleted.");
  });

  it("synchronously clears all exposed state when a workspace becomes null", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));

    rerender({ workspaceId: null, state: localState() });

    expect(result.current.available).toBe(false);
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.listError).toBeNull();
    expect(result.current.mutation).toBeNull();
    expect(result.current.announcement).toBeNull();
  });

  it("synchronously clears a pending mutation on identity changes", async () => {
    const pendingCreate = deferred<ReturnType<typeof validView>>();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.createWorkspaceSourceView.mockReturnValue(pendingCreate.promise);
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.createView("Pending");
    });
    expect(result.current.mutation).toBe("create");

    rerender({ workspaceId: "ws-b", state: localState() });
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.limitState).toBeNull();
    expect(result.current.listError).toBeNull();
    expect(result.current.mutationError).toBeNull();
    expect(result.current.mutation).toBeNull();
    expect(result.current.announcement).toBeNull();

    await act(async () => {
      pendingCreate.resolve(validView({ name: "Pending" }));
      await pending;
    });
  });

  it("synchronously clears populated conflict and mutation errors", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: {
        detail: {
          code: "source_view_name_exists",
          view_id: "view-1",
          version: 2,
        },
      },
    });
    api.updateWorkspaceSourceView.mockRejectedValue(new Error("offline"));
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));
    await act(async () => result.current.createView("Duplicate"));
    await act(async () => result.current.confirmReplace());
    expect(result.current.duplicateConflict).not.toBeNull();
    expect(result.current.mutationError).not.toBeNull();

    rerender({ workspaceId: null, state: localState() });
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.mutationError).toBeNull();
  });

  it("ignores deferred list and mutation completions after switching or nulling", async () => {
    const listA = deferred<{ items: ReturnType<typeof validView>[] }>();
    const createB = deferred<ReturnType<typeof validView>>();
    api.listWorkspaceSourceViews
      .mockReturnValueOnce(listA.promise)
      .mockResolvedValueOnce({ items: [] });
    api.createWorkspaceSourceView.mockReturnValue(createB.promise);
    const { result, rerender } = setup("ws-a");

    rerender({ workspaceId: "ws-b", state: localState() });
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledWith("ws-b"),
    );
    await act(async () => {
      const pending = result.current.createView("B view");
      rerender({ workspaceId: null, state: localState() });
      createB.resolve(validView({ workspace_id: "ws-b", name: "B view" }));
      await pending;
    });
    await act(async () => listA.resolve({ items: [validView()] }));

    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.announcement).toBeNull();
  });

  it("rejects an A response after A to B to A using monotonic generations", async () => {
    const oldA = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews
      .mockReturnValueOnce(oldA.promise)
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ items: [validView({ id: "new-a" })] });
    const { result, rerender } = setup("ws-a");

    rerender({ workspaceId: "ws-b", state: localState() });
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledWith("ws-b"),
    );
    rerender({ workspaceId: "ws-a", state: localState() });
    await waitFor(() => expect(result.current.views[0]?.id).toBe("new-a"));
    await act(async () =>
      oldA.resolve({ items: [validView({ id: "old-a" })] }),
    );

    expect(result.current.views.map((view) => view.id)).toEqual(["new-a"]);
    expect(result.current.generation).toBe(2);
  });

  it("rejects an A mutation completion after A to B to A", async () => {
    const oldCreateA = deferred<ReturnType<typeof validView>>();
    api.createWorkspaceSourceView.mockReturnValue(oldCreateA.promise);
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValueOnce({ items: [validView({ id: "fresh-a" })] });
    const { result, rerender } = setup("ws-a");
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledWith("ws-a"),
    );

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.createView("Old A");
    });
    expect(result.current.busy).toBe(true);
    rerender({ workspaceId: "ws-b", state: localState() });
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledWith("ws-b"),
    );
    rerender({ workspaceId: "ws-a", state: localState() });
    await waitFor(() => expect(result.current.views[0]?.id).toBe("fresh-a"));

    await act(async () => {
      oldCreateA.resolve(validView({ id: "old-create-a", name: "Old A" }));
      await pending;
    });

    expect(result.current.views.map((view) => view.id)).toEqual(["fresh-a"]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.announcement).toBeNull();
    expect(result.current.generation).toBe(2);
  });

  it("derives Modified from the active canonical signature and treats invalid local state as Modified", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    const { result, rerender } = setup(
      "ws-a",
      localState({ reviewStateFilters: ["needs_review"] }),
    );
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));
    expect(result.current.modified).toBe(false);

    rerender({
      workspaceId: "ws-a",
      state: localState({ typeFilters: ["pdf"] }),
    });
    expect(result.current.modified).toBe(true);

    rerender({
      workspaceId: "ws-a",
      state: localState({ fileSizeMin: Number.NaN }),
    });
    expect(result.current.currentSignature).toBeNull();
    expect(result.current.modified).toBe(true);
  });
});
