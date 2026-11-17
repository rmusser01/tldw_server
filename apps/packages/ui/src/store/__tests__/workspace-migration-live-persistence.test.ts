import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  flushWorkspacePersistence,
  useWorkspaceStore,
  workspacePersistWriterInternals,
} from "../workspace";
import {
  runResearchWorkspaceMigration,
  type ResearchWorkspaceMigrationApi,
} from "../workspace-migration";

beforeEach(async () => {
  await flushWorkspacePersistence();
  workspacePersistWriterInternals.cancelPendingWrite();
  localStorage.clear();
  useWorkspaceStore.setState(useWorkspaceStore.getInitialState(), true);
  await flushWorkspacePersistence();
});

afterEach(() => workspacePersistWriterInternals.cancelPendingWrite());

it("retains a real status update and all local payloads across receipt retries", async () => {
  const workspace = useWorkspaceStore
    .getState()
    .initializeWorkspace("Live migration");
  const source = useWorkspaceStore.getState().addSource({
    mediaId: 17,
    title: "Retain source",
    type: "text",
    status: "processing",
  });
  await flushWorkspacePersistence();
  const values = () =>
    Array.from(
      { length: localStorage.length },
      (_, i) => localStorage.key(i)!,
    ).filter(
      (key) =>
        key === "tldw-workspace" || key.startsWith("tldw-workspace:workspace:"),
    );
  const original = new Map(
    values().map((key) => [key, localStorage.getItem(key)]),
  );
  const api = {
    createWorkspaceMigration: vi.fn(
      async (
        body: Parameters<
          ResearchWorkspaceMigrationApi["createWorkspaceMigration"]
        >[0],
      ) => ({ id: body.id, status: "created", client_delete_eligible: false }),
    ),
    putWorkspaceMigrationChunk: vi.fn(async () => ({})),
    finalizeWorkspaceMigration: vi.fn(async (id: string) => {
      useWorkspaceStore
        .getState()
        .setSourceStatusByMediaId(17, "ready", "Ready", {
          text_extracted: true,
          metadata_ready: true,
          fts_ready: true,
          vector_ready: false,
          summary_ready: false,
          citation_ready: true,
          tool_accessible: true,
        });
      await flushWorkspacePersistence();
      return { id, status: "finalized", client_delete_eligible: false };
    }),
    getWorkspaceMigration: vi.fn(async (id: string) => ({
      id,
      status: "finalized",
      client_delete_eligible: false,
    })),
    ackWorkspaceMigrationClientDelete: vi.fn(async () => ({})),
  } satisfies ResearchWorkspaceMigrationApi;
  const remove = vi.fn();
  const write = vi.fn();
  const run = () =>
    runResearchWorkspaceMigration({
      targetWorkspaceId: workspace,
      targetWorkspaceName: "Live migration",
      discoveredLocalStorageKeys: values(),
      readLocalStorageValue: async (key) => localStorage.getItem(key),
      writeLocalStorageValue: write,
      deleteLocalStorageValue: remove,
      api,
    });
  const first = await run();
  expect(first.status).toBe("finalized_not_delete_eligible");
  expect(remove).not.toHaveBeenCalled();
  expect(write).not.toHaveBeenCalled();
  expect(api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
  expect(
    values().some((key) => localStorage.getItem(key) !== original.get(key)),
  ).toBe(true);
  expect(
    useWorkspaceStore.getState().sources.find((row) => row.id === source.id)
      ?.status,
  ).toBe("ready");
  const changed = new Map(
    values().map((key) => [key, localStorage.getItem(key)]),
  );
  api.finalizeWorkspaceMigration.mockImplementation(async (id: string) => ({
    id,
    status: "finalized",
    client_delete_eligible: false,
  }));
  const second = await run();
  expect(second.status).toBe("finalized_not_delete_eligible");
  expect(remove).not.toHaveBeenCalled();
  expect(write).not.toHaveBeenCalled();
  expect(api.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
  expect(changed.size).toBeGreaterThan(0);
  expect(
    [...changed].every(([key, value]) => localStorage.getItem(key) === value),
  ).toBe(true);
});
