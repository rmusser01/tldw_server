import { afterEach, describe, expect, it, vi } from "vitest";

import { PageAssitDatabase } from "../..";

type PromptRecord = {
  id: string;
  title: string;
  content: string;
  is_system: boolean;
  createdAt: number;
};

const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

const installDelayedStorage = (initial: PromptRecord[]) => {
  let records = structuredClone(initial);
  const writes: Array<{
    value: PromptRecord[];
    gate: ReturnType<typeof deferred>;
  }> = [];
  vi.stubGlobal("chrome", {
    storage: {
      local: {
        get: (_key: string, callback: (result: { prompts: PromptRecord[] }) => void) =>
          callback({ prompts: structuredClone(records) }),
        set: ({ prompts }: { prompts: PromptRecord[] }) => {
          const gate = deferred();
          writes.push({ value: structuredClone(prompts), gate });
          return gate.promise.then(() => {
            records = structuredClone(prompts);
          });
        },
      },
    },
  });
  return {
    records: () => structuredClone(records),
    writes,
    release: async (index: number) => {
      writes[index].gate.resolve();
      await writes[index].gate.promise;
    },
  };
};

describe("Firefox prompt write ordering", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("finishes an add before an exact rollback delete reads and writes", async () => {
    const storage = installDelayedStorage([]);
    const db = new PageAssitDatabase();
    const added: PromptRecord = {
      id: "new-id",
      title: "New",
      content: "new",
      is_system: true,
      createdAt: 1,
    };
    const workflow = (async () => {
      await db.addPrompt(added);
      await db.deletePrompt(added.id);
    })();

    await vi.waitFor(() => expect(storage.writes).toHaveLength(1));
    await storage.release(0);
    await vi.waitFor(() => expect(storage.writes).toHaveLength(2));
    await storage.release(1);
    await workflow;

    expect(storage.records()).toEqual([]);
  });

  it("finishes an update before restoring the exact prior snapshot", async () => {
    const original: PromptRecord = {
      id: "saved-id",
      title: "Original",
      content: "  exact 🧪\n",
      is_system: true,
      createdAt: 1,
    };
    const storage = installDelayedStorage([original]);
    const db = new PageAssitDatabase();
    const workflow = (async () => {
      await db.updatePrompt(original.id, { content: "changed" });
      await db.restorePromptSnapshot(original);
    })();

    await vi.waitFor(() => expect(storage.writes).toHaveLength(1));
    await storage.release(0);
    await vi.waitFor(() => expect(storage.writes).toHaveLength(2));
    await storage.release(1);
    await workflow;

    expect(storage.records()).toEqual([original]);
  });
});
