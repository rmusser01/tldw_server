vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
import { beforeEach, expect, it, vi } from "vitest";
import {
  readSurfaceOfflineDraftQueue,
  retainSurfaceOfflineDraft,
  retireSurfaceOfflineDraft,
  writeNotesEditorOfflineDraftQueue,
  type OfflineDraftEntry,
} from "../notes-manager-utils";

const storageKey = "tldw:notesOfflineDraftQueue:v1:alice";
const draft = (key: string, requestKey: string): OfflineDraftEntry => ({
  key,
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
    key: requestKey,
    body: {
      id: "8ed7ae96-cea4-4264-83ac-7ee6bdfe044e",
      content: "Immutable body",
    },
    expectedVersion: null,
  },
});
beforeEach(() => window.localStorage.clear());

it("merges surface operations with ordinary drafts and preserves them through a stale main editor snapshot", async () => {
  const ordinary = draft("draft:new", "ordinary");
  writeNotesEditorOfflineDraftQueue(storageKey, { "draft:new": ordinary });
  const quick = draft("surface:quick-notes:workspace-a", "quick");
  const exportDraft = draft("surface:knowledge-export:thread-a", "export");
  await retainSurfaceOfflineDraft("alice", quick);
  await retainSurfaceOfflineDraft("alice", exportDraft);
  writeNotesEditorOfflineDraftQueue(storageKey, {});
  expect(await readSurfaceOfflineDraftQueue("alice")).toEqual({
    [quick.key]: quick,
    [exportDraft.key]: exportDraft,
  });
  await retireSurfaceOfflineDraft("alice", quick.key, "quick");
  writeNotesEditorOfflineDraftQueue(storageKey, {});
  expect(await readSurfaceOfflineDraftQueue("alice")).toEqual({
    [exportDraft.key]: exportDraft,
  });
});

it("never replaces another unresolved operation at the same surface key", async () => {
  const original = draft("surface:knowledge-export:thread-a", "original");
  await retainSurfaceOfflineDraft("alice", original);
  await expect(
    retainSurfaceOfflineDraft("alice", draft(original.key, "different")),
  ).rejects.toThrow();
  expect((await readSurfaceOfflineDraftQueue("alice"))[original.key]).toEqual(
    original,
  );
});

it("retires only the acknowledged operation and keeps another owner's queue", async () => {
  const original = draft("surface:knowledge-export:thread-a", "original");
  await retainSurfaceOfflineDraft("alice", original);
  await retainSurfaceOfflineDraft("bob", original);
  await retireSurfaceOfflineDraft("alice", original.key, "different");
  expect((await readSurfaceOfflineDraftQueue("alice"))[original.key]).toEqual(
    original,
  );
  await retireSurfaceOfflineDraft("alice", original.key, "original");
  expect(await readSurfaceOfflineDraftQueue("alice")).toEqual({});
  expect((await readSurfaceOfflineDraftQueue("bob"))[original.key]).toEqual(
    original,
  );
});

it("keeps a separately retained operation when another module writes a stale ordinary queue", async () => {
  const staleOrdinary = window.localStorage.getItem(storageKey);
  vi.resetModules();
  const otherModule = await import("../notes-manager-utils");
  const pending = draft(
    "surface:knowledge-export:thread-a:0463b785-f3ba-4b66-a656-a77986e0fe9e",
    "original",
  );
  await otherModule.retainSurfaceOfflineDraft("alice", pending);
  const prototype = Object.getPrototypeOf(window.localStorage) as Storage;
  const original = prototype.getItem;
  let firstRead = true;
  const read = vi
    .spyOn(prototype, "getItem")
    .mockImplementation(function (key) {
      if (key === storageKey && firstRead) {
        firstRead = false;
        return staleOrdinary;
      }
      return original.call(this, key);
    });
  writeNotesEditorOfflineDraftQueue(storageKey, {});
  read.mockRestore();
  expect(
    (await otherModule.readSurfaceOfflineDraftQueue("alice"))[pending.key],
  ).toEqual(pending);
});
