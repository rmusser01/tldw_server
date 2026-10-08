import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { OfflineDraftEntry } from "../notes-manager-utils";

// Real Plasmo/safe-storage/local-registry routing; only the Chrome API is controlled.
const entries: Record<string, unknown> = {};
const get = vi.fn(async (keys?: string | string[] | null) => {
  if (keys == null) return { ...entries };
  const selected = typeof keys === "string" ? [keys] : keys;
  return Object.fromEntries(
    selected.filter((key) => key in entries).map((key) => [key, entries[key]]),
  );
});
const set = vi.fn(async (values: Record<string, unknown>) => {
  Object.assign(entries, values);
});
const remove = vi.fn(async (keys: string | string[]) => {
  for (const key of typeof keys === "string" ? [keys] : keys)
    delete entries[key];
});
const draft: OfflineDraftEntry = {
  key: "surface:knowledge-export:thread-a:8ed7ae96-cea4-4264-83ac-7ee6bdfe044e",
  noteId: null,
  baseVersion: null,
  title: "Original",
  content: "Immutable body",
  keywords: [],
  metadata: null,
  backlinkConversationId: null,
  backlinkMessageId: null,
  updatedAt: "2026-10-08T00:00:00.000Z",
  syncState: "queued",
  lastError: null,
  pendingWrite: {
    key: "immutable-key",
    body: {
      id: "8ed7ae96-cea4-4264-83ac-7ee6bdfe044e",
      content: "Immutable body",
    },
    expectedVersion: null,
  },
};
const storageKey = "tldw:notesOfflineDraftQueue:v1:alice:" + draft.key;
beforeEach(() => {
  window.localStorage.clear();
  for (const key of Object.keys(entries)) delete entries[key];
  vi.resetModules();
  get.mockClear();
  set.mockClear();
  remove.mockClear();
  vi.stubGlobal("browser", {
    storage: {
      local: { get, set, remove },
      onChanged: { addListener: vi.fn(), removeListener: vi.fn() },
    },
  });
});
afterEach(() => vi.unstubAllGlobals());

it("routes per-record pending writes through real extension serialization while retaining ordinary native-map compatibility", async () => {
  const owning = await import("../notes-manager-utils");
  window.localStorage.setItem(
    "tldw:notesOfflineDraftQueue:v1:alice",
    JSON.stringify({ "draft:new": { ...draft, key: "draft:new" } }),
  );
  await owning.retainSurfaceOfflineDraft("alice", draft);
  expect(typeof entries[storageKey]).toBe("string");
  expect(JSON.parse(entries[storageKey] as string).value).toEqual(draft);
  expect(window.localStorage.getItem(storageKey)).toBeNull();
  vi.resetModules();
  const second = await import("../notes-manager-utils");
  const other = {
    ...draft,
    key: "surface:quick-notes:workspace-a:65cd4527-a45e-4f5a-8e20-311965ad427d",
  };
  await second.retainSurfaceOfflineDraft("alice", other);
  owning.writeNotesEditorOfflineDraftQueue(
    "tldw:notesOfflineDraftQueue:v1:alice",
    {},
  );
  expect(await second.readSurfaceOfflineDraftQueue("alice")).toEqual({
    [draft.key]: draft,
    [other.key]: other,
  });
});

it("refuses a softly failed extension write and reads durable state outside registry tab pins", async () => {
  const owning = await import("../notes-manager-utils");
  const registry = await import("@/services/settings/local-bucket");
  const pinned = registry.createLocalRegistryBucket<OfflineDraftEntry>({
    prefix: "tldw:notesOfflineDraftQueue:v1:alice:",
    tabScoped: true,
  });
  set.mockRejectedValueOnce(new Error("Quota"));
  await pinned.set(draft.key, draft);
  expect((await pinned.get(draft.key))?.value).toEqual(draft);
  expect(await owning.readSurfaceOfflineDraftQueue("alice")).toEqual({});
  set.mockRejectedValueOnce(new Error("Quota"));
  await expect(
    owning.retainSurfaceOfflineDraft("alice", draft),
  ).rejects.toThrow("Could not retain");
  expect(entries[storageKey]).toBeUndefined();
});

it("refuses extension durable readback failure and preserves failed-retirement identity", async () => {
  const owning = await import("../notes-manager-utils");
  get
    .mockImplementationOnce(async () => ({}))
    .mockRejectedValueOnce(new Error("Read unavailable"));
  await expect(
    owning.retainSurfaceOfflineDraft("alice", draft),
  ).rejects.toThrow("Read unavailable");
  expect(JSON.parse(entries[storageKey] as string).value.pendingWrite).toEqual(
    draft.pendingWrite,
  );
  remove.mockRejectedValueOnce(new Error("Remove unavailable"));
  await expect(
    owning.retireSurfaceOfflineDraft("alice", draft.key, "immutable-key"),
  ).rejects.toThrow("Could not retire");
  expect(
    (await owning.readSurfaceOfflineDraftQueue("alice"))[draft.key]
      .pendingWrite,
  ).toEqual(draft.pendingWrite);
  await owning.retainSurfaceOfflineDraft("alice", draft);
  await owning.retireSurfaceOfflineDraft("alice", draft.key, "immutable-key");
  expect(await owning.readSurfaceOfflineDraftQueue("alice")).toEqual({});
});
