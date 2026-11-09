import { beforeEach, expect, it, vi } from "vitest";
import {
  deleteResearchWorkspaceMigrationLocalValue,
  runResearchWorkspaceMigration,
} from "../workspace-migration";
import {
  classifyResearchWorkspaceLegacyStorageSurface,
  evaluateResearchWorkspaceLegacyDeletionEligibility,
} from "../research-workspace-legacy-storage-inventory";

const A = "workspace-a",
  B = "workspace-b";
const snapshotKey = (id: string) =>
  "tldw-workspace:workspace:" + id + ":snapshot";
const api = () => ({
  createWorkspaceMigration: vi.fn(async (body) => ({
    id: body.id,
    status: "created",
    client_delete_eligible: false,
  })),
  putWorkspaceMigrationChunk: vi.fn(async () => ({})),
  finalizeWorkspaceMigration: vi.fn(async (id: string) => ({
    id,
    status: "finalized",
    client_delete_eligible: true,
  })),
  getWorkspaceMigration: vi.fn(async (id: string) => ({
    id,
    status: "finalized",
    client_delete_eligible: true,
  })),
  ackWorkspaceMigrationClientDelete: vi.fn(async () => ({})),
});
const run = (values: Map<string, string>, network = api(), extra = {}) =>
  runResearchWorkspaceMigration({
    targetWorkspaceId: A,
    targetWorkspaceName: "Alpha",
    discoveredLocalStorageKeys: [...values.keys()],
    readLocalStorageValue: async (key) => values.get(key) ?? null,
    deleteLocalStorageValue: (key) => {
      values.delete(key);
    },
    writeLocalStorageValue: (key, value) => {
      values.set(key, value);
    },
    api: network,
    ...extra,
  });

beforeEach(() => localStorage.clear());

it("retains older deletion markers as non-authoritative metadata without writing new ones", async () => {
  const values = new Map([["tldw-workspace", JSON.stringify({ workspaceId: A })]]);
  const written: Array<[string, string]> = [];
  const result = await run(values, api(), {
    writeLocalStorageValue: (key: string, value: string) => { written.push([key, value]); }
  });
  expect(result.status).toBe("finalized_not_delete_eligible");
  expect(written).toEqual([]);
  expect(values.get("tldw-workspace")).toBe(JSON.stringify({ workspaceId: A }));
  const keys = [
    "tldw:research-workspace:migration:tombstone:" + A,
    "tldw:research-workspace:migration:tombstone-preflight:" + A
  ];
  for (const key of keys)
    expect(classifyResearchWorkspaceLegacyStorageSurface({ kind: "local_storage", key })).toMatchObject({
      classification: "metadata", deletionPolicy: "retain_local", authoritativeForMigration: false
    });
  expect(evaluateResearchWorkspaceLegacyDeletionEligibility({
    discoveredLocalStorageKeys: keys, manifestCoveredSurfaceIds: []
  }).retainedLocalSurfaces).toHaveLength(2);
});

it("keeps reserved-marker lookalikes and empty marker identities blocking", () => {
  for (const key of [
    "tldw:research-workspace:migration:tombstone-other:" + A,
    "tldw:research-workspace:migration:tombstone:",
    "tldw:research-workspace:migration:tombstone-preflight:",
  ]) {
    const result = evaluateResearchWorkspaceLegacyDeletionEligibility({
      discoveredLocalStorageKeys: [key],
      manifestCoveredSurfaceIds: [],
    });
    expect(result.eligible).toBe(false);
  }
});

it("does not claim or delete B split content during A migration", async () => {
  const untouched = JSON.stringify({
    workspaceId: B,
    currentNote: { content: "Keep Beta" },
    generatedArtifacts: [{ id: "beta-output" }],
  });
  const chat = JSON.stringify({
    messages: [{ message: "Keep Beta conversation" }],
  });
  const values = new Map([
    [snapshotKey(A), JSON.stringify({ workspaceId: A })],
    [snapshotKey(B), untouched],
    ["tldw-workspace:workspace:" + B + ":chat", chat],
  ]);
  const network = api();
  const result = await run(values, network);
  expect(result.status).toBe("blocked");
  expect(values.get(snapshotKey(B))).toBe(untouched);
  expect(values.get("tldw-workspace:workspace:" + B + ":chat")).toBe(chat);
  expect(result.deletedSurfaceIds).toEqual([]);
  expect(
    network.createWorkspaceMigration.mock.calls[0][0].manifest
      .covered_surface_ids,
  ).toEqual(["localStorage:" + snapshotKey(A)]);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it.each([
  {
    workspaceSnapshots: {
      [B]: { workspaceId: B, currentNote: { content: "Keep Beta" } },
    },
  },
  { savedWorkspaces: [{ id: B, name: "Beta" }] },
  { archivedWorkspaces: [{ id: B, name: "Beta" }] },
  { workspaceIds: [A, B] },
  {
    workspaceChatSessions: {
      [B + "::conversation"]: { messages: [{ message: "Keep Beta" }] },
    },
  },
  {
    generatedArtifacts: [
      {
        payloadRef: {
          offloadType: "workspace_artifact_payload_v1",
          key: "workspace:" + B + ":artifact:output",
          fields: ["content"],
          updatedAt: 1,
        },
      },
    ],
  },
])(
  "retains a mixed shared index despite its active target identity: %j",
  async (hidden) => {
    const raw = JSON.stringify({
      schema: "tldw-workspace-split-index",
      state: { workspaceId: A, ...hidden },
    });
    const values = new Map([["tldw-workspace", raw]]);
    const network = api(),
      result = await run(values, network);
    expect(result.status).toBe("blocked");
    expect(values.get("tldw-workspace")).toBe(raw);
    expect(result.deletedSurfaceIds).toEqual([]);
    expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
  },
);

it.each([
  "{}",
  "broken-json",
  JSON.stringify({ currentNote: { content: "Unbound note" } }),
])("retains an unbound or malformed shared payload: %s", async (raw) => {
  const values = new Map([["tldw-workspace", raw]]);
  const result = await run(values);
  expect(result.status).toBe("blocked");
  expect(values.get("tldw-workspace")).toBe(raw);
});

it("retains a shared IndexedDB collection containing another workspace", async () => {
  const untouched = {
    records: [
      { workspaceId: A, content: "Alpha" },
      { workspaceId: B, content: "Keep Beta" },
    ],
  };
  const deleteStore = vi.fn(),
    network = api();
  const result = await run(new Map(), network, {
    discoveredIndexedDbStores: [
      {
        databaseName: "tldw-workspace-storage",
        storeName: "workspace-artifact-payloads",
      },
    ],
    readIndexedDbStorePayload: async () => untouched,
    deleteIndexedDbStorePayload: deleteStore,
  });
  expect(result.status).toBe("blocked");
  expect(deleteStore).not.toHaveBeenCalled();
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("preserves a same-key replacement while server readback is held", async () => {
  const key = snapshotKey(A),
    values = new Map([
      [
        key,
        JSON.stringify({ workspaceId: A, currentNote: { content: "Planned" } }),
      ],
    ]);
  const network = api();
  let release!: () => void;
  network.getWorkspaceMigration.mockImplementation(
    (id) =>
      new Promise((resolve) => {
        release = () =>
          resolve({ id, status: "finalized", client_delete_eligible: true });
      }),
  );
  const work = run(values, network);
  await vi.waitFor(() => expect(release).toBeTypeOf("function"));
  const replacement = JSON.stringify({
    workspaceId: A,
    currentNote: { content: "Keep newer unsaved note" },
  });
  values.set(key, replacement);
  release();
  const result = await work;
  expect(result.status).not.toBe("deleted");
  expect(values.get(key)).toBe(replacement);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("does not mutate local data after the owner or workspace retires during readback", async () => {
  const key = snapshotKey(A),
    raw = JSON.stringify({
      workspaceId: A,
      currentNote: { content: "Keep Alpha" },
    }),
    values = new Map([[key, raw]]);
  const network = api();
  let current = true;
  network.getWorkspaceMigration.mockImplementation(async (id) => {
    current = false;
    return { id, status: "finalized", client_delete_eligible: true };
  });
  const result = await run(values, network, { isCurrent: () => current });
  expect(result.status).not.toBe("deleted");
  expect(values.get(key)).toBe(raw);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("retains a valid target-only payload after receipt matching", async () => {
  const key = snapshotKey(A),
    values = new Map([
      [
        key,
        JSON.stringify({ workspaceId: A, currentNote: { content: "Alpha" } }),
      ],
    ]);
  const network = api(),
    result = await run(values, network);
  expect(result.status).toBe("finalized_not_delete_eligible");
  expect(values.get(key)).toBe(JSON.stringify({ workspaceId: A, currentNote: { content: "Alpha" } }));
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("rejects contradictory shared-record ownership even when every row names the target", async () => {
  const raw = JSON.stringify({
    workspaceId: B,
    records: [{ workspaceId: A, content: "Alpha" }],
  });
  const values = new Map([["tldw-workspace", raw]]);
  const result = await run(values);
  expect(result.status).toBe("blocked");
  expect(values.get("tldw-workspace")).toBe(raw);
});

it.each(["changed", "retired"])(
  "preserves real localStorage at the final removal boundary when %s",
  (kind) => {
    const key = snapshotKey(A),
      original = JSON.stringify({
        workspaceId: A,
        currentNote: { content: "Planned" },
      });
    const current = kind === "changed" ? "New unsaved draft" : original;
    localStorage.setItem(key, current);
    expect(() =>
      deleteResearchWorkspaceMigrationLocalValue(
        key,
        original,
        () => kind !== "retired",
      ),
    ).toThrow(/Local workspace changed/);
    expect(localStorage.getItem(key)).toBe(current);
  },
);

it("removes only unchanged captured localStorage while its workspace is current", () => {
  const key = snapshotKey(A),
    raw = JSON.stringify({ workspaceId: A });
  localStorage.setItem(key, raw);
  deleteResearchWorkspaceMigrationLocalValue(key, raw, () => true);
  expect(localStorage.getItem(key)).toBeNull();
});

it("retains an orphan top-level draft despite a target saved-workspace listing", async () => {
  const raw = JSON.stringify({
    savedWorkspaces: [{ id: A }],
    currentNote: { content: "Unbound draft" },
  });
  const values = new Map([["tldw-workspace", raw]]);
  expect((await run(values)).status).toBe("blocked");
  expect(values.get("tldw-workspace")).toBe(raw);
});

it("rejects an IndexedDB record key contradicting its declared workspace", async () => {
  const network = api(),
    remove = vi.fn();
  const result = await run(new Map(), network, {
    discoveredIndexedDbStores: [
      {
        databaseName: "tldw-workspace-storage",
        storeName: "workspace-chat-sessions",
      },
    ],
    readIndexedDbStorePayload: async () => ({
      workspaceId: A,
      key: "workspace:" + B + ":chat",
      session: { messages: [{ message: "Keep Beta" }] },
    }),
    deleteIndexedDbStorePayload: remove,
  });
  expect(result.status).toBe("blocked");
  expect(remove).not.toHaveBeenCalled();
});

it("does not invoke destructive callbacks that could overwrite a queued draft", async () => {
  const key = snapshotKey(A),
    values = new Map([[key, JSON.stringify({ workspaceId: A })]]),
    network = api();
  const replacement = JSON.stringify({
    workspaceId: A,
    currentNote: { content: "Keep queued draft" },
  });
  const result = await run(values, network, {
    deleteLocalStorageValue: (name: string) => {
      values.delete(name);
      if (name === key) queueMicrotask(() => values.set(key, replacement));
    },
  });
  expect(result.status).not.toBe("deleted");
  expect(values.get(key)).toBe(JSON.stringify({ workspaceId: A }));
  expect(
    [...values.keys()].filter((name) =>
      name.startsWith("tldw:research-workspace:migration:tombstone:"),
    ),
  ).toEqual([]);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("retains a target-only IndexedDB collection after receipt matching", async () => {
  let payload: unknown = {
    workspaceId: A,
    key: "workspace:" + A + ":chat",
    session: { messages: [{ message: "Alpha" }] },
  };
  const network = api(),
    remove = vi.fn(async () => {
      payload = null;
    });
  const result = await run(new Map(), network, {
    discoveredIndexedDbStores: [
      {
        databaseName: "tldw-workspace-storage",
        storeName: "workspace-chat-sessions",
      },
    ],
    readIndexedDbStorePayload: async () => payload,
    deleteIndexedDbStorePayload: remove,
  });
  expect(result.status).toBe("finalized_not_delete_eligible");
  expect(remove).not.toHaveBeenCalled();
  expect(payload).toEqual({
    workspaceId: A, key: "workspace:" + A + ":chat",
    session: { messages: [{ message: "Alpha" }] }
  });
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("retains every chunk without invoking a partially failing deletion callback", async () => {
  const first = snapshotKey(A),
    second = "tldw-workspace:workspace:" + A + ":chat";
  const raw = JSON.stringify({
    messages: [{ message: "Keep remaining chat" }],
  });
  const values = new Map([
      [first, JSON.stringify({ workspaceId: A })],
      [second, raw],
    ]),
    network = api();
  const result = await run(values, network, {
    deleteLocalStorageValue: (key: string) => {
      if (key === second) throw new Error("Storage blocked");
      values.delete(key);
    },
  });
  expect(result.status).toBe("finalized_not_delete_eligible");
  expect(result.deletedSurfaceIds).toEqual([]);
  expect(values.get(first)).toBe(JSON.stringify({ workspaceId: A }));
  expect(values.get(second)).toBe(raw);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("retains unbound sibling content beside target-bound records", async () => {
  const raw = JSON.stringify({
    records: [{ workspaceId: A, content: "Alpha" }],
    currentNote: { content: "Keep unbound draft" },
  });
  const values = new Map([["tldw-workspace", raw]]),
    network = api();
  expect((await run(values, network)).status).toBe("blocked");
  expect(values.get("tldw-workspace")).toBe(raw);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("retains an explicitly target-bound records-only envelope", async () => {
  const values = new Map([
    [
      "tldw-workspace",
      JSON.stringify({
        records: [{ workspaceId: A, content: "Alpha" }],
        version: 1,
      }),
    ],
  ]);
  expect((await run(values)).status).toBe("finalized_not_delete_eligible");
  expect(JSON.parse(values.get("tldw-workspace")!).records).toEqual([{ workspaceId: A, content: "Alpha" }]);
});

it("does not invoke a callback that would remove data and then reject", async () => {
  const key = snapshotKey(A),
    values = new Map([[key, JSON.stringify({ workspaceId: A })]]),
    network = api();
  const result = await run(values, network, {
    deleteLocalStorageValue: async (name: string) => {
      values.delete(name);
      throw new Error("Delete receipt lost");
    },
  });
  expect(result.status).toBe("finalized_not_delete_eligible");
  expect(values.get(key)).toBe(JSON.stringify({ workspaceId: A }));
  expect(result.deletedSurfaceIds).toEqual([]);
  expect(result.message).not.toContain("before local deletion");
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
});

it("retains target notes, artifacts and chat when the server stores only their hashes", async () => {
  const key = snapshotKey(A);
  const payload = JSON.stringify({
    workspaceId: A,
    currentNote: { content: "Only recoverable note" },
    generatedArtifacts: [{ id: "artifact-1", content: "Only recoverable output" }],
    chatMessages: [{ content: "Only recoverable conversation" }]
  });
  const values = new Map([[key, payload]]);
  const network = api(); // Older servers can incorrectly advertise deletion eligibility.
  const result = await run(values, network);
  expect(values.get(key)).toBe(payload);
  expect(result.status).toBe("finalized_not_delete_eligible");
  expect(result.deletedSurfaceIds).toEqual([]);
  expect(network.ackWorkspaceMigrationClientDelete).not.toHaveBeenCalled();
  expect([...values.keys()]).toEqual([key]);
});
