import React from "react";
import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
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
  workspaceExists = true,
) =>
  renderHook(
    ({ workspaceId, state }) =>
      useSourceSavedViews(
        workspaceId,
        workspaceExists,
        state,
        onApplyState,
      ),
    { initialProps: { workspaceId, state } },
  );

describe("useSourceSavedViews", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [] });
  });

  it("is unavailable for null workspaces and never requests", () => {
    const { result } = setup(null);

    expect(result.current.available).toBe(false);
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(api.listWorkspaceSourceViews).not.toHaveBeenCalled();
  });

  it("waits for matching server workspace readiness before listing saved views", async () => {
    const { result, rerender } = renderHook(
      ({ workspaceExists }) =>
        useSourceSavedViews(
          "workspace-new",
          workspaceExists,
          localState(),
          vi.fn(),
        ),
      { initialProps: { workspaceExists: false } },
    );

    expect(result.current.available).toBe(false);
    expect(api.listWorkspaceSourceViews).not.toHaveBeenCalled();

    rerender({ workspaceExists: true });

    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledWith(
        "workspace-new",
      ),
    );
    expect(result.current.available).toBe(true);
  });

  it("blocks saved-view mutations while the server workspace is not ready", async () => {
    const onApplyState = vi.fn();
    const { result } = setup("workspace-new", localState(), onApplyState, false);
    const view = validView({ workspace_id: "workspace-new" });

    await act(async () => {
      result.current.applyView(view);
      await result.current.createView("Blocked create");
      await result.current.replaceView(view);
      await result.current.resetView(view);
      await result.current.deleteView(view);
      await result.current.retry();
      await result.current.retryMutation();
      await result.current.retryVersionConflict();
    });

    expect(onApplyState).not.toHaveBeenCalled();
    expect(api.listWorkspaceSourceViews).not.toHaveBeenCalled();
    expect(api.createWorkspaceSourceView).not.toHaveBeenCalled();
    expect(api.updateWorkspaceSourceView).not.toHaveBeenCalled();
    expect(api.deleteWorkspaceSourceView).not.toHaveBeenCalled();
  });

  it("does not let an abandoned workspace render invalidate committed requests", async () => {
    const listA = deferred<{ items: ReturnType<typeof validView>[] }>();
    const never = new Promise<void>(() => undefined);
    const suspensionSpy = vi.spyOn(never, "then");
    api.listWorkspaceSourceViews.mockReturnValue(listA.promise);

    const Harness = ({ workspaceId, suspend }: { workspaceId: string; suspend: boolean }) => {
      const controller = useSourceSavedViews(
        workspaceId,
        true,
        localState(),
        vi.fn(),
      );
      if (workspaceId === "ws-b" && suspend) {
        throw never;
      }
      return (
        <div data-testid="committed-source-views">
          {controller.views.map((view) => view.id).join(",")}
        </div>
      );
    };

    const rendered = render(
      <React.Suspense fallback={<div>Loading pending workspace</div>}>
        <Harness workspaceId="ws-a" suspend={false} />
      </React.Suspense>,
    );
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledWith("ws-a"),
    );

    act(() => {
      React.startTransition(() => {
        rendered.rerender(
          <React.Suspense fallback={<div>Loading pending workspace</div>}>
            <Harness workspaceId="ws-b" suspend />
          </React.Suspense>,
        );
      });
    });
    await waitFor(() => expect(suspensionSpy).toHaveBeenCalled());

    await act(async () => listA.resolve({ items: [validView()] }));

    expect(screen.getByTestId("committed-source-views")).toHaveTextContent(
      "view-1",
    );
    expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1);
  });

  it("keeps committed generations monotonic under React StrictMode", async () => {
    const StrictModeWrapper = ({ children }: { children: React.ReactNode }) => (
      <React.StrictMode>{children}</React.StrictMode>
    );
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    const { result, rerender } = renderHook(
      ({ workspaceId }) =>
        useSourceSavedViews(workspaceId, true, localState(), vi.fn()),
      {
        initialProps: { workspaceId: "ws-a" },
        wrapper: StrictModeWrapper,
      },
    );
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    expect(result.current.generation).toBe(0);

    const listB = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews.mockReturnValue(listB.promise);
    rerender({ workspaceId: "ws-b" });
    expect(result.current.generation).toBe(1);
    expect(result.current.views).toEqual([]);
    await act(async () =>
      listB.resolve({
        items: [validView({ id: "view-b", workspace_id: "ws-b" })],
      }),
    );
    await waitFor(() => expect(result.current.views[0]?.id).toBe("view-b"));

    const listA = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews.mockReturnValue(listA.promise);
    rerender({ workspaceId: "ws-a" });
    expect(result.current.generation).toBe(2);
    expect(result.current.views).toEqual([]);
    await act(async () => listA.resolve({ items: [validView()] }));
    await waitFor(() => expect(result.current.views[0]?.id).toBe("view-1"));
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

  it("keeps local preset controls available when saved-view listing fails", async () => {
    api.listWorkspaceSourceViews.mockRejectedValue(new Error("offline"));
    const { result } = setup("ws-a", localState({ typeFilters: ["pdf"] }));

    await waitFor(() => expect(result.current.listError).not.toBeNull());

    expect(result.current.available).toBe(true);
    expect(result.current.views).toEqual([]);
    expect(result.current.busy).toBe(false);
    expect(result.current.mutation).toBeNull();
    expect(result.current.currentSignature).not.toBeNull();
  });

  it("reconciles all rows after a create supersedes the initial load", async () => {
    const initialLoad = deferred<{ items: ReturnType<typeof validView>[] }>();
    const reconciliation = deferred<{
      items: ReturnType<typeof validView>[];
    }>();
    const existing = validView({ id: "existing-view", name: "Existing" });
    const created = validView({ id: "created-view", name: "Created" });
    api.listWorkspaceSourceViews
      .mockReturnValueOnce(initialLoad.promise)
      .mockReturnValueOnce(reconciliation.promise);
    api.createWorkspaceSourceView.mockResolvedValue(created);
    const { result } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );

    await act(async () => result.current.createView("Created"));
    expect(result.current.views.map((view) => view.id)).toEqual([
      "created-view",
    ]);
    expect(result.current.activeViewId).toBe("created-view");
    expect(result.current.announcement).toBe("Saved view created.");
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(2),
    );

    await act(async () => initialLoad.resolve({ items: [existing] }));

    expect(result.current.views.map((view) => view.id)).toEqual([
      "created-view",
    ]);

    await act(async () =>
      reconciliation.resolve({ items: [existing, created] }),
    );

    expect(result.current.views.map((view) => view.id)).toEqual([
      "existing-view",
      "created-view",
    ]);
    expect(result.current.activeViewId).toBe("created-view");
    expect(result.current.announcement).toBe("Saved view created.");
  });

  it("makes same-generation retries latest-wins", async () => {
    api.listWorkspaceSourceViews.mockResolvedValueOnce({ items: [] });
    const { result } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );
    const firstRetry = deferred<{ items: ReturnType<typeof validView>[] }>();
    const secondRetry = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews
      .mockReturnValueOnce(firstRetry.promise)
      .mockReturnValueOnce(secondRetry.promise);

    let firstPending!: Promise<void>;
    let secondPending!: Promise<void>;
    act(() => {
      firstPending = result.current.retry();
      secondPending = result.current.retry();
    });
    await act(async () =>
      secondRetry.resolve({ items: [validView({ id: "newest-list" })] }),
    );
    await secondPending;
    await act(async () =>
      firstRetry.resolve({ items: [validView({ id: "older-list" })] }),
    );
    await firstPending;

    expect(result.current.views.map((view) => view.id)).toEqual([
      "newest-list",
    ]);
  });

  it("does not let an older retry overwrite a newer replace", async () => {
    api.listWorkspaceSourceViews.mockResolvedValueOnce({ items: [validView()] });
    const { result } = setup("ws-a", localState({ typeFilters: ["pdf"] }));
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    const retryLoad = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews.mockReturnValueOnce(retryLoad.promise);
    api.updateWorkspaceSourceView.mockResolvedValue(
      validView({
        version: 3,
        state: wireState({ type_filters: ["pdf"] }),
      }),
    );

    let retryPending!: Promise<void>;
    act(() => {
      retryPending = result.current.retry();
    });
    await act(async () => result.current.replaceView(result.current.views[0]!));
    expect(result.current.views[0]?.version).toBe(3);

    await act(async () =>
      retryLoad.resolve({ items: [validView({ id: "stale-retry" })] }),
    );
    await retryPending;

    expect(result.current.views[0]?.version).toBe(3);
    expect(result.current.views[0]?.state).toEqual(
      wireState({ type_filters: ["pdf"] }),
    );
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("does not let an older retry restore a deleted view", async () => {
    api.listWorkspaceSourceViews.mockResolvedValueOnce({ items: [validView()] });
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    const retryLoad = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews.mockReturnValueOnce(retryLoad.promise);
    api.deleteWorkspaceSourceView.mockResolvedValue(undefined);

    let retryPending!: Promise<void>;
    act(() => {
      retryPending = result.current.retry();
    });
    await act(async () => result.current.deleteView(result.current.views[0]!));
    expect(result.current.views).toEqual([]);

    await act(async () => retryLoad.resolve({ items: [validView()] }));
    await retryPending;

    expect(result.current.views).toEqual([]);
    expect(result.current.announcement).toBe("Saved view deleted.");
  });

  it("does not start a list retry while a replacement is in flight", async () => {
    const row = validView();
    const replacement = deferred<ReturnType<typeof validView>>();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [row] });
    api.updateWorkspaceSourceView.mockReturnValue(replacement.promise);
    const { result } = setup(
      "ws-a",
      localState({ typeFilters: ["pdf"] }),
    );
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.replaceView(result.current.views[0]!);
    });
    await act(async () => result.current.retry());

    expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1);

    await act(async () => {
      replacement.resolve(
        validView({
          version: 3,
          state: wireState({ type_filters: ["pdf"] }),
        }),
      );
      await pending;
    });
    expect(result.current.views[0]?.version).toBe(3);
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("ignores a duplicate mutation call before the busy state rerenders", async () => {
    const pendingCreate = deferred<ReturnType<typeof validView>>();
    api.createWorkspaceSourceView.mockReturnValue(pendingCreate.promise);
    const { result } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );

    let first!: Promise<void>;
    let duplicate!: Promise<void>;
    act(() => {
      first = result.current.createView("One request");
      duplicate = result.current.createView("One request");
    });

    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(1);
    await act(async () => {
      pendingCreate.resolve(validView({ name: "One request" }));
      await Promise.all([first, duplicate]);
    });
    expect(result.current.views[0]?.name).toBe("One request");
    expect(result.current.busy).toBe(false);
    expect(result.current.announcement).toBe("Saved view created.");
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

  it("accepts 120 Unicode code points and rejects 121", async () => {
    const astralCharacter = "\u{1F4C4}";
    const maximumName = astralCharacter.repeat(120);
    const tooLongName = astralCharacter.repeat(121);
    api.createWorkspaceSourceView.mockResolvedValue(
      validView({ name: maximumName, state: wireState() }),
    );
    const { result } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalled(),
    );

    await act(async () => result.current.createView(maximumName));

    expect(api.createWorkspaceSourceView).toHaveBeenCalledWith(
      "ws-a",
      expect.objectContaining({ name: maximumName }),
    );

    await act(async () => result.current.createView(tooLongName));

    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(1);
    expect(result.current.serializationIssues).toEqual(
      expect.arrayContaining([
        {
          field: "name",
          message: "Name must contain between 1 and 120 characters.",
        },
      ]),
    );
  });

  it("rejects a malformed valid create response atomically", async () => {
    api.createWorkspaceSourceView.mockResolvedValue(
      validView({ id: "malformed-create", state: { unknown_field: true } }),
    );
    const onApply = vi.fn();
    const { result } = setup("ws-a", localState(), onApply);
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalled(),
    );

    await act(async () => result.current.createView("Malformed"));

    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.activeSnapshot).toBeNull();
    expect(onApply).not.toHaveBeenCalled();
    expect(result.current.announcement).toBeNull();
    expect(result.current.busy).toBe(false);
    expect(result.current.mutation).toBeNull();
    expect(result.current.mutationError?.message).toBe(
      "The server returned an invalid saved view.",
    );
    expect(result.current.canRetryMutation).toBe(true);
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

  it("recomputes state issues after same-workspace local state correction", async () => {
    api.createWorkspaceSourceView.mockResolvedValue(
      validView({ name: "Corrected view", state: wireState() }),
    );
    const { result, rerender } = setup(
      "ws-a",
      localState({ fileSizeMin: 20, fileSizeMax: 10 }),
    );

    await act(async () => result.current.createView("Corrected view"));
    expect(result.current.serializationIssues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "fileSizeMax" }),
      ]),
    );

    rerender({ workspaceId: "ws-a", state: localState() });
    expect(result.current.serializationIssues).toEqual([]);

    await act(async () => result.current.createView("Corrected view"));
    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(1);
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

  it("dismisses duplicate replacement and lets the next save start fresh", async () => {
    api.createWorkspaceSourceView
      .mockRejectedValueOnce({
        status: 409,
        details: {
          detail: {
            code: "source_view_name_exists",
            view_id: "view-1",
            version: 2,
          },
        },
      })
      .mockResolvedValueOnce(validView({ id: "view-2", name: "Fresh name" }));
    const { result } = setup();

    await act(async () => result.current.createView("Duplicate"));
    expect(result.current.duplicateConflict).not.toBeNull();

    act(() => result.current.dismissDuplicateConflict());
    expect(result.current.duplicateConflict).toBeNull();

    await act(async () => result.current.createView("Fresh name"));
    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(2);
    expect(api.updateWorkspaceSourceView).not.toHaveBeenCalled();
    expect(result.current.activeViewId).toBe("view-2");
  });

  it("dismisses a failed duplicate replacement and invalidates its retry", async () => {
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
    const { result } = setup();

    await act(async () => result.current.createView("Duplicate"));
    await act(async () => result.current.confirmReplace());
    expect(result.current.canRetryMutation).toBe(true);

    act(() => result.current.dismissDuplicateConflict());
    await act(async () => result.current.retryMutation());

    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.mutationError).toBeNull();
    expect(result.current.canRetryMutation).toBe(false);
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(1);
  });

  it("ignores duplicate dismissal while its replacement is in flight", async () => {
    const replacement = deferred<ReturnType<typeof validView>>();
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
    api.updateWorkspaceSourceView.mockReturnValue(replacement.promise);
    const { result } = setup();

    await act(async () => result.current.createView("Duplicate"));
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.confirmReplace();
    });
    expect(result.current.busy).toBe(true);

    act(() => result.current.dismissDuplicateConflict());
    expect(result.current.duplicateConflict).not.toBeNull();

    await act(async () => {
      replacement.resolve(validView({ name: "Duplicate", version: 3 }));
      await pending;
    });

    expect(result.current.busy).toBe(false);
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("ignores mutation failure dismissal while a replacement is in flight", async () => {
    const replacement = deferred<ReturnType<typeof validView>>();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.updateWorkspaceSourceView.mockReturnValue(replacement.promise);
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.replaceView(result.current.views[0]!);
    });
    expect(result.current.busy).toBe(true);

    act(() => result.current.dismissMutationFailure());

    await act(async () => {
      replacement.resolve(validView({ version: 3 }));
      await pending;
    });

    expect(result.current.busy).toBe(false);
    expect(result.current.views[0]?.version).toBe(3);
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

  it("treats malformed version-conflict detail as an ordinary retryable error", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.updateWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: {
        detail: {
          code: "source_view_version_conflict",
          view_id: "view-1",
          current_version: "5",
        },
      },
    });
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    await act(async () => result.current.replaceView(result.current.views[0]!));

    expect(result.current.versionConflict).toBeNull();
    expect(result.current.mutationError?.retryable).toBe(true);
    expect(result.current.canRetryMutation).toBe(true);
    expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1);
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

    rerender({ workspaceId: null, state: localState() });
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
    const { result, rerender } = setup();

    await act(async () => result.current.createView("Name"));
    await act(async () => result.current.confirmReplace());

    expect(result.current.versionConflict).toEqual({
      viewId: "view-1",
      currentVersion: 5,
      retryable: true,
    });
    expect(result.current.views[0]?.version).toBe(5);
    expect(result.current.duplicateConflict?.version).toBe(5);
    expect(result.current.canRetryVersion).toBe(true);
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(1);

    rerender({ workspaceId: null, state: localState() });
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.versionConflict).toBeNull();
    expect(result.current.canRetryVersion).toBe(false);
  });

  it("does not retry a failed create after a replace supersedes it", async () => {
    api.createWorkspaceSourceView.mockRejectedValue(new Error("offline"));
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
      .mockResolvedValueOnce({ items: [validView()] })
      .mockResolvedValueOnce({ items: [validView({ version: 5 })] });
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    await act(async () => result.current.createView("Old create"));
    expect(result.current.canRetryMutation).toBe(true);
    expect(result.current.canRetryVersion).toBe(false);

    await act(async () => result.current.replaceView(result.current.views[0]!));
    expect(result.current.canRetryMutation).toBe(false);
    expect(result.current.canRetryVersion).toBe(true);

    await act(async () => result.current.retryMutation());

    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(1);
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
    expect(result.current.canRetryVersion).toBe(false);
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("dismisses an ordinary version conflict and invalidates its retry", async () => {
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [validView()] })
      .mockResolvedValueOnce({ items: [validView({ version: 5 })] });
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
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    await act(async () => result.current.replaceView(result.current.views[0]!));
    expect(result.current.canRetryVersion).toBe(true);

    act(() => result.current.dismissMutationFailure());
    await act(async () => result.current.retryVersionConflict());

    expect(result.current.versionConflict).toBeNull();
    expect(result.current.canRetryVersion).toBe(false);
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(1);
  });

  it.each(["create", "replace", "reset", "delete"] as const)(
    "dismisses a failed %s mutation and invalidates its retry",
    async (kind) => {
      api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
      api.createWorkspaceSourceView.mockRejectedValue(new Error("offline"));
      api.updateWorkspaceSourceView.mockRejectedValue(new Error("offline"));
      api.deleteWorkspaceSourceView.mockRejectedValue(new Error("offline"));
      const { result } = setup();
      await waitFor(() => expect(result.current.views).toHaveLength(1));

      await act(async () => {
        if (kind === "create") await result.current.createView("Failed save");
        if (kind === "replace") {
          await result.current.replaceView(result.current.views[0]!);
        }
        if (kind === "reset") await result.current.resetView(invalidView());
        if (kind === "delete") {
          await result.current.deleteView(result.current.views[0]!);
        }
      });
      expect(result.current.canRetryMutation).toBe(true);

      act(() => result.current.dismissMutationFailure());
      await act(async () => result.current.retryMutation());

      expect(result.current.mutationError).toBeNull();
      expect(result.current.canRetryMutation).toBe(false);
      expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(
        kind === "create" ? 1 : 0,
      );
      expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(
        kind === "replace" || kind === "reset" ? 1 : 0,
      );
      expect(api.deleteWorkspaceSourceView).toHaveBeenCalledTimes(
        kind === "delete" ? 1 : 0,
      );
    },
  );

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

  it("directly replaces a valid saved view with the current canonical state", async () => {
    const current = localState({ typeFilters: ["pdf"], sort: "name_desc" });
    const row = validView();
    const updated = validView({
      version: 3,
      state: wireState({ type_filters: ["pdf"], sort: "name_desc" }),
    });
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [row] });
    api.updateWorkspaceSourceView.mockResolvedValue(updated);
    const { result } = setup("ws-a", current);
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    await act(async () => result.current.replaceView(result.current.views[0]!));

    expect(api.updateWorkspaceSourceView).toHaveBeenCalledWith("ws-a", "view-1", {
      version: 2,
      schema_version: 1,
      state: wireState({ type_filters: ["pdf"], sort: "name_desc" }),
    });
    expect(result.current.activeViewId).toBe("view-1");
    expect(result.current.activeSnapshot).toEqual(updated.state);
    expect(result.current.modified).toBe(false);
    expect(result.current.announcement).toBe("Saved view replaced.");
  });

  it("rejects a malformed valid patch response atomically", async () => {
    const row = validView();
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [row] });
    api.updateWorkspaceSourceView.mockResolvedValue(
      validView({ version: 3, state: { sort: "unsupported" } }),
    );
    const onApply = vi.fn();
    const { result } = setup(
      "ws-a",
      localState({ typeFilters: ["pdf"] }),
      onApply,
    );
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));
    onApply.mockClear();
    const activeSnapshot = result.current.activeSnapshot;

    await act(async () => result.current.replaceView(result.current.views[0]!));

    expect(result.current.views).toEqual([row]);
    expect(result.current.activeViewId).toBe(row.id);
    expect(result.current.activeSnapshot).toEqual(activeSnapshot);
    expect(onApply).not.toHaveBeenCalled();
    expect(result.current.announcement).toBeNull();
    expect(result.current.busy).toBe(false);
    expect(result.current.mutation).toBeNull();
    expect(result.current.mutationError?.message).toBe(
      "The server returned an invalid saved view.",
    );
    expect(result.current.canRetryMutation).toBe(true);
  });

  it("preserves a concurrent server rename across an ordinary replace retry", async () => {
    const current = localState({ typeFilters: ["pdf"] });
    const renamed = validView({ name: "Renamed concurrently", version: 5 });
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [validView()] })
      .mockResolvedValueOnce({ items: [renamed] });
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
      .mockResolvedValueOnce(
        validView({
          name: "Renamed concurrently",
          version: 6,
          state: wireState({ type_filters: ["pdf"] }),
        }),
      );
    const { result } = setup("ws-a", current);
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    await act(async () => result.current.replaceView(result.current.views[0]!));
    expect(api.updateWorkspaceSourceView).toHaveBeenNthCalledWith(
      1,
      "ws-a",
      "view-1",
      {
        version: 2,
        schema_version: 1,
        state: wireState({ type_filters: ["pdf"] }),
      },
    );
    expect(result.current.views[0]?.name).toBe("Renamed concurrently");

    await act(async () => result.current.retryVersionConflict());

    expect(api.updateWorkspaceSourceView).toHaveBeenNthCalledWith(
      2,
      "ws-a",
      "view-1",
      {
        version: 5,
        schema_version: 1,
        state: wireState({ type_filters: ["pdf"] }),
      },
    );
    expect(result.current.views[0]?.name).toBe("Renamed concurrently");
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

  it.each(["replace", "reset"] as const)(
    "reconciles a missing row after a %s returns 404",
    async (kind) => {
      const row = kind === "replace" ? validView() : invalidView();
      api.listWorkspaceSourceViews
        .mockResolvedValueOnce({ items: [row] })
        .mockResolvedValueOnce({ items: [] });
      api.updateWorkspaceSourceView.mockRejectedValue({
        status: 404,
        message: "Saved view not found.",
      });
      const { result } = setup();
      await waitFor(() => expect(result.current.views).toHaveLength(1));

      await act(async () => {
        if (kind === "replace") {
          await result.current.replaceView(result.current.views[0]!);
        } else {
          await result.current.resetView(result.current.views[0]!);
        }
      });

      await waitFor(() =>
        expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(2),
      );
      expect(result.current.views).toEqual([]);
      expect(result.current.mutation).toBeNull();
      expect(result.current.mutationError).toBeNull();
      expect(result.current.canRetryMutation).toBe(false);
      expect(result.current.announcement).toBe("Saved view no longer exists.");
    },
  );

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

  it("treats a direct DELETE 404 as an already completed deletion", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.deleteWorkspaceSourceView.mockRejectedValue({
      status: 404,
      message: "Saved view not found.",
    });
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));

    await act(async () => result.current.deleteView(result.current.views[0]!));

    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.mutationError).toBeNull();
    expect(result.current.canRetryMutation).toBe(false);
    expect(result.current.announcement).toBe("Saved view deleted.");
  });

  it("reconciles other rows after DELETE 404 supersedes an incomplete list", async () => {
    const initialLoad = deferred<{ items: ReturnType<typeof validView>[] }>();
    const staleReconciliation = deferred<{
      items: ReturnType<typeof validView>[];
    }>();
    const created = validView({ id: "created-view", name: "Created" });
    const existing = validView({ id: "existing-view", name: "Existing" });
    api.listWorkspaceSourceViews
      .mockReturnValueOnce(initialLoad.promise)
      .mockReturnValueOnce(staleReconciliation.promise)
      .mockResolvedValueOnce({ items: [existing] });
    api.createWorkspaceSourceView.mockResolvedValue(created);
    api.deleteWorkspaceSourceView.mockRejectedValue({
      status: 404,
      message: "Saved view not found.",
    });
    const { result } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );

    await act(async () => result.current.createView("Created"));
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(2),
    );
    expect(result.current.views.map((view) => view.id)).toEqual([
      "created-view",
    ]);

    await act(async () => result.current.deleteView(created));

    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(3),
    );
    expect(result.current.views.map((view) => view.id)).toEqual([
      "existing-view",
    ]);
    expect(result.current.announcement).toBe("Saved view deleted.");
  });

  it("treats a DELETE retry 404 as success after a lost response", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.deleteWorkspaceSourceView
      .mockRejectedValueOnce(new Error("Connection lost after request."))
      .mockRejectedValueOnce({
        status: 404,
        message: "Saved view not found.",
      });
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    await act(async () => result.current.deleteView(result.current.views[0]!));
    expect(result.current.canRetryMutation).toBe(true);

    await act(async () => result.current.retryMutation());

    expect(api.deleteWorkspaceSourceView).toHaveBeenCalledTimes(2);
    expect(result.current.views).toEqual([]);
    expect(result.current.mutationError).toBeNull();
    expect(result.current.canRetryMutation).toBe(false);
    expect(result.current.announcement).toBe("Saved view deleted.");
  });

  it("clears limit and serialization transients after a successful delete", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.createWorkspaceSourceView.mockRejectedValue({
      status: 409,
      details: { detail: { code: "source_view_limit_reached", limit: 100 } },
    });
    api.deleteWorkspaceSourceView.mockResolvedValue(undefined);
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    await act(async () => result.current.createView("At limit"));
    rerender({
      workspaceId: "ws-a",
      state: localState({ fileSizeMin: Number.NaN }),
    });
    await act(async () => result.current.createView("Invalid"));
    expect(result.current.limitState).not.toBeNull();
    expect(result.current.serializationIssues).not.toEqual([]);

    await act(async () => result.current.deleteView(result.current.views[0]!));

    expect(result.current.limitState).toBeNull();
    expect(result.current.serializationIssues).toEqual([]);
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.versionConflict).toBeNull();
    expect(result.current.mutationError).toBeNull();
    expect(result.current.canRetryMutation).toBe(false);
  });

  it("clears version conflict state and retry after deleting its view", async () => {
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
    api.deleteWorkspaceSourceView.mockResolvedValue(undefined);
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    await act(async () => result.current.createView("Duplicate"));
    await act(async () => result.current.confirmReplace());
    expect(result.current.duplicateConflict).not.toBeNull();
    expect(result.current.versionConflict).not.toBeNull();

    await act(async () => result.current.deleteView(result.current.views[0]!));
    await act(async () => result.current.retryVersionConflict());

    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.versionConflict).toBeNull();
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(1);
  });

  it("clears ordinary conflict retries after a successful delete", async () => {
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
    api.deleteWorkspaceSourceView.mockResolvedValue(undefined);
    const { result } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    await act(async () => result.current.createView("Duplicate"));
    await act(async () => result.current.confirmReplace());
    expect(result.current.mutationError).not.toBeNull();
    expect(result.current.canRetryMutation).toBe(true);

    await act(async () => result.current.deleteView(result.current.views[0]!));
    await act(async () => result.current.retryMutation());

    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.mutationError).toBeNull();
    expect(result.current.canRetryMutation).toBe(false);
    expect(api.updateWorkspaceSourceView).toHaveBeenCalledTimes(1);
  });

  it("synchronously clears all exposed state when a workspace becomes null", async () => {
    api.listWorkspaceSourceViews.mockResolvedValue({ items: [validView()] });
    api.createWorkspaceSourceView.mockResolvedValue(
      validView({ name: "Snapshot", state: wireState() }),
    );
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));
    await act(async () => result.current.createView("Snapshot"));
    rerender({
      workspaceId: "ws-a",
      state: localState({ fileSizeMin: Number.NaN }),
    });
    await act(async () => result.current.createView("Invalid snapshot"));

    expect(result.current.activeSnapshot).not.toBeNull();
    expect(result.current.modified).toBe(true);
    expect(result.current.serializationIssues).not.toEqual([]);
    expect(result.current.announcement).toBe("Saved view created.");

    rerender({ workspaceId: null, state: localState() });

    expect(result.current.available).toBe(false);
    expect(result.current.views).toEqual([]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.activeSnapshot).toBeNull();
    expect(result.current.modified).toBe(false);
    expect(result.current.serializationIssues).toEqual([]);
    expect(result.current.duplicateConflict).toBeNull();
    expect(result.current.limitState).toBeNull();
    expect(result.current.versionConflict).toBeNull();
    expect(result.current.listError).toBeNull();
    expect(result.current.mutationError).toBeNull();
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

    rerender({ workspaceId: null, state: localState() });
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

  it("invalidates deferred list work and retry callbacks on unmount", async () => {
    const pendingList = deferred<{ items: ReturnType<typeof validView>[] }>();
    api.listWorkspaceSourceViews.mockReturnValue(pendingList.promise);
    const { result, unmount } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );
    const retry = result.current.retry;

    unmount();
    await act(async () => pendingList.reject(new Error("offline")));
    await act(async () => retry());

    expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1);
  });

  it("invalidates deferred mutation retry callbacks on unmount", async () => {
    const pendingCreate = deferred<ReturnType<typeof validView>>();
    api.createWorkspaceSourceView.mockReturnValue(pendingCreate.promise);
    const { result, unmount } = setup();
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.createView("Unmounted create");
    });
    const retryMutation = result.current.retryMutation;

    unmount();
    await act(async () => pendingCreate.reject(new Error("offline")));
    await pending;
    await act(async () => retryMutation());

    expect(api.createWorkspaceSourceView).toHaveBeenCalledTimes(1);
  });

  it("does not apply a deferred reset after unmount", async () => {
    const pendingReset = deferred<ReturnType<typeof validView>>();
    api.updateWorkspaceSourceView.mockReturnValue(pendingReset.promise);
    const onApply = vi.fn();
    const { result, unmount } = setup("ws-a", localState(), onApply);
    await waitFor(() =>
      expect(api.listWorkspaceSourceViews).toHaveBeenCalledTimes(1),
    );
    let pending!: Promise<void>;
    act(() => {
      pending = result.current.resetView(invalidView());
    });

    unmount();
    await act(async () => pendingReset.resolve(validView()));
    await pending;

    expect(onApply).not.toHaveBeenCalled();
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

  it("ignores a stale PATCH completion after a workspace switch", async () => {
    const oldPatch = deferred<ReturnType<typeof validView>>();
    api.updateWorkspaceSourceView.mockReturnValue(oldPatch.promise);
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [validView()] })
      .mockResolvedValueOnce({
        items: [validView({ id: "fresh-b", workspace_id: "ws-b" })],
      });
    const { result, rerender } = setup(
      "ws-a",
      localState({ typeFilters: ["pdf"] }),
    );
    await waitFor(() => expect(result.current.views).toHaveLength(1));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.replaceView(result.current.views[0]!);
    });
    expect(result.current.mutation).toBe("replace");
    rerender({ workspaceId: "ws-b", state: localState() });
    await waitFor(() => expect(result.current.views[0]?.id).toBe("fresh-b"));

    await act(async () => {
      oldPatch.resolve(validView({ id: "stale-patch" }));
      await pending;
    });

    expect(result.current.views.map((view) => view.id)).toEqual(["fresh-b"]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.mutation).toBeNull();
    expect(result.current.announcement).toBeNull();
  });

  it("ignores a stale DELETE completion after a workspace switch", async () => {
    const oldDelete = deferred<void>();
    api.deleteWorkspaceSourceView.mockReturnValue(oldDelete.promise);
    api.listWorkspaceSourceViews
      .mockResolvedValueOnce({ items: [validView()] })
      .mockResolvedValueOnce({
        items: [validView({ id: "fresh-b", workspace_id: "ws-b" })],
      });
    const { result, rerender } = setup();
    await waitFor(() => expect(result.current.views).toHaveLength(1));
    act(() => result.current.applyView(result.current.views[0]!));

    let pending!: Promise<void>;
    act(() => {
      pending = result.current.deleteView(result.current.views[0]!);
    });
    expect(result.current.mutation).toBe("delete");
    rerender({ workspaceId: "ws-b", state: localState() });
    await waitFor(() => expect(result.current.views[0]?.id).toBe("fresh-b"));

    await act(async () => {
      oldDelete.resolve(undefined);
      await pending;
    });

    expect(result.current.views.map((view) => view.id)).toEqual(["fresh-b"]);
    expect(result.current.activeViewId).toBeNull();
    expect(result.current.activeSnapshot).toBeNull();
    expect(result.current.mutation).toBeNull();
    expect(result.current.announcement).toBeNull();
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
