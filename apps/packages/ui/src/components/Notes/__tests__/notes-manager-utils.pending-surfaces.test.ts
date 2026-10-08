vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  checkpointQuickNotesOfflineDraft,
  quickNotesDraftMatchesWrite,
  isQuickNotesRetainedDraft,
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

it("preserves a readable normalized legacy surface row while replacing ordinary drafts", async () => {
  const legacy = draft("surface:knowledge-export:legacy-thread", "legacy-key");
  const separate = draft(
    "surface:quick-notes:workspace-a:separate",
    "separate-key",
  );
  await retainSurfaceOfflineDraft("alice", separate);
  window.localStorage.setItem(
    storageKey,
    JSON.stringify({ [legacy.key]: legacy, invalid: null }),
  );
  const ordinary = draft("draft:new", "ordinary-key");
  writeNotesEditorOfflineDraftQueue(storageKey, { [ordinary.key]: ordinary });
  expect(JSON.parse(window.localStorage.getItem(storageKey)!)).toEqual({
    [ordinary.key]: ordinary,
    [legacy.key]: legacy,
  });
  expect(await readSurfaceOfflineDraftQueue("alice")).toEqual({
    [separate.key]: separate,
  });
});

describe("Quick Notes mutable draft and accepted receipt under the house record lock", () => {
  const id = "8ed7ae96-cea4-4264-83ac-7ee6bdfe044e";
  const quick = (): OfflineDraftEntry => ({
    ...draft(
      'surface:quick-notes:["workspace-a","workspace:a"]:draft-a',
      "frozen-key",
    ),
    metadata: {
      quickNotesAuthorityId: "verified-authority",
      quickNotesWorkspaceId: "workspace-a",
      quickNotesWorkspaceTag: "workspace:a",
    },
    pendingWrite: {
      authorityId: "verified-authority",
      key: "frozen-key",
      expectedVersion: null,
      body: {
        id,
        title: "Original",
        content: "Immutable body",
        keywords: ["workspace:a"],
      },
    },
  });
  beforeEach(() => {
    const tails = new Map<string, Promise<unknown>>();
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, {
        locks: {
          value: {
            request: (key: string, operation: () => unknown) => {
              const next = (tails.get(key) ?? Promise.resolve()).then(
                operation,
              );
              tails.set(
                key,
                next.catch(() => undefined),
              );
              return next;
            },
          },
        },
      }),
    );
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it.each(["checkpoint-first", "ack-first"])(
    "keeps later text across independent modules with %s and never resurrects the operation",
    async (order) => {
      const original = quick();
      await retainSurfaceOfflineDraft("alice", original);
      vi.resetModules();
      const other = await import("../notes-manager-utils");
      const { Storage } = await import("@plasmohq/storage");
      let release!: () => void;
      let entered!: () => void;
      const blocked = new Promise<void>((resolve) => {
        entered = resolve;
      });
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const get = Storage.prototype.get;
      let paused = false;
      vi.spyOn(Storage.prototype, "get").mockImplementation(async function (
        key: string,
      ) {
        const value = await get.call(this, key);
        if (!paused && key.endsWith(original.key)) {
          paused = true;
          entered();
          await gate;
        }
        return value;
      });
      let local = { ...original, isDirty: true };
      const checkpoint = () =>
        other.checkpointQuickNotesOfflineDraft(
          "alice",
          original.key,
          "frozen-key",
          () => local,
        );
      const ack = () =>
        other.checkpointQuickNotesOfflineDraft(
          "alice",
          original.key,
          "frozen-key",
          () => null,
          { id, version: 2 },
        );
      const first = order === "checkpoint-first" ? checkpoint() : ack();
      await blocked;
      const second = order === "checkpoint-first" ? ack() : checkpoint();
      local = { ...local, title: "Later title", content: "Later text" };
      release();
      await Promise.all([first, second]);
      const retained = (await readSurfaceOfflineDraftQueue("alice"))[
        original.key
      ];
      expect(retained).toMatchObject({
        title: "Later title",
        content: "Later text",
        noteId: id,
        baseVersion: 2,
        metadata: {
          quickNotesDirty: true,
          quickNotesAuthorityId: "verified-authority",
          quickNotesAcceptedKey: "frozen-key",
        },
      });
      expect(retained.pendingWrite).toBeUndefined();
      expect(isQuickNotesRetainedDraft(retained)).toBe(true);
    },
  );

  it("refuses a mutable checkpoint without the required lock and leaves its frozen operation intact", async () => {
    const original = quick();
    await retainSurfaceOfflineDraft("alice", original);
    vi.stubGlobal("navigator", {});
    await expect(
      checkpointQuickNotesOfflineDraft(
        "alice",
        original.key,
        "frozen-key",
        () => ({ ...original, content: "Later", isDirty: true }),
      ),
    ).rejects.toThrow("record lock");
    expect((await readSurfaceOfflineDraftQueue("alice"))[original.key]).toEqual(
      original,
    );
  });

  it("refuses quota loss and leaves the pending operation recoverable", async () => {
    const original = quick();
    await retainSurfaceOfflineDraft("alice", original);
    const prototype = Object.getPrototypeOf(window.localStorage);
    vi.spyOn(prototype, "setItem").mockImplementation(() => {
      throw new DOMException("Quota", "QuotaExceededError");
    });
    await expect(
      checkpointQuickNotesOfflineDraft(
        "alice",
        original.key,
        "frozen-key",
        () => ({ ...original, content: "Later", isDirty: true }),
      ),
    ).rejects.toThrow();
    expect((await readSurfaceOfflineDraftQueue("alice"))[original.key]).toEqual(
      original,
    );
  });

  it("validates draft-only ownership and keeps clean accepted rows out of pending-operation recovery", async () => {
    const original = quick();
    await retainSurfaceOfflineDraft("alice", original);
    const accepted = await checkpointQuickNotesOfflineDraft(
      "alice",
      original.key,
      "frozen-key",
      () => null,
      { id, version: 2 },
    );
    expect(accepted).toMatchObject({
      noteId: id,
      baseVersion: 2,
      metadata: {
        quickNotesDirty: false,
        quickNotesAuthorityId: "verified-authority",
      },
    });
    expect(accepted?.pendingWrite).toBeUndefined();
    expect(
      isQuickNotesRetainedDraft({
        ...accepted!,
        key: "surface:knowledge-export:thread-a",
      }),
    ).toBe(false);
    expect(isQuickNotesRetainedDraft({ ...accepted!, baseVersion: null })).toBe(
      false,
    );
    expect(
      isQuickNotesRetainedDraft({
        ...accepted!,
        metadata: { ...accepted!.metadata, quickNotesAuthorityId: undefined },
      }),
    ).toBe(false);
  });
  it.each(["checkpoint-first", "rejection-first"])(
    "preserves dirty canonical text at the rejected base with %s",
    async (order) => {
      const original = {
        ...quick(),
        noteId: id,
        baseVersion: 2,
        pendingWrite: {
          ...quick().pendingWrite!,
          expectedVersion: 2,
          body: {
            title: "Original",
            content: "Immutable body",
            keywords: ["workspace:a"],
          },
        },
      };
      await retainSurfaceOfflineDraft("alice", original);
      vi.resetModules();
      const other = await import("../notes-manager-utils");
      const local = {
        ...original,
        content: "Later rejected text",
        isDirty: true,
      };
      const checkpoint = () =>
        other.checkpointQuickNotesOfflineDraft(
          "alice",
          original.key,
          "frozen-key",
          () => local,
        );
      const reject = () =>
        retireSurfaceOfflineDraft("alice", original.key, "frozen-key");
      await Promise.all(
        order === "checkpoint-first"
          ? [checkpoint(), reject()]
          : [reject(), checkpoint()],
      );
      const retained = (await readSurfaceOfflineDraftQueue("alice"))[
        original.key
      ];
      expect(retained).toMatchObject({
        content: "Later rejected text",
        noteId: id,
        baseVersion: 2,
        metadata: {
          quickNotesRejectedKey: "frozen-key",
          quickNotesDirty: true,
          quickNotesAuthorityId: "verified-authority",
        },
      });
      expect(retained.pendingWrite).toBeUndefined();
      expect(retained.metadata?.quickNotesAcceptedKey).toBeUndefined();
      await retainSurfaceOfflineDraft("alice", original);
      expect(
        (await readSurfaceOfflineDraftQueue("alice"))[original.key],
      ).toEqual(retained);
    },
  );

  it("keeps accepted identity after readback failure without resurrecting its retired request", async () => {
    const original = quick();
    await retainSurfaceOfflineDraft("alice", original);
    const { Storage } = await import("@plasmohq/storage");
    const get = Storage.prototype.get;
    const failedRead = vi
      .spyOn(Storage.prototype, "get")
      .mockImplementation(async function (key: string) {
        const value = await get.call(this, key);
        if (
          key.endsWith(original.key) &&
          value?.value?.metadata?.quickNotesAcceptedKey
        )
          throw new Error("Read unavailable");
        return value;
      });
    await expect(
      checkpointQuickNotesOfflineDraft(
        "alice",
        original.key,
        "frozen-key",
        () => null,
        { id, version: 2 },
      ),
    ).rejects.toThrow("Read unavailable");
    failedRead.mockRestore();
    await retainSurfaceOfflineDraft("alice", original);
    const retained = (await readSurfaceOfflineDraftQueue("alice"))[
      original.key
    ];
    expect(retained).toMatchObject({
      noteId: id,
      baseVersion: 2,
      metadata: { quickNotesAcceptedKey: "frozen-key", quickNotesDirty: false },
    });
    expect(retained.pendingWrite).toBeUndefined();
  });

  it("compares keyword casing and pending provenance against the frozen body", () => {
    const original = quick();
    expect(
      quickNotesDraftMatchesWrite(
        { ...original, keywords: ["Case"] },
        { ...original.pendingWrite!.body, keywords: ["case", "workspace:a"] },
      ),
    ).toBe(false);
    expect(
      quickNotesDraftMatchesWrite(
        {
          ...original,
          metadata: {
            ...original.metadata,
            pendingKnowledgeProvenance: {
              origin: "knowledge_qa",
              question: "Later source choice",
            },
          },
        },
        original.pendingWrite!.body,
      ),
    ).toBe(false);
  });
});
