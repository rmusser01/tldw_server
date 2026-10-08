import { afterEach, beforeEach, expect, it, vi } from "vitest";

// Two independent module contexts share the real installed Plasmo storage adapter.
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
beforeEach(() => {
  localStorage.clear();
  for (const key of Object.keys(entries)) delete entries[key];
  vi.resetModules();
  get.mockClear();
  set.mockClear();
  vi.stubGlobal("browser", {
    storage: {
      local: { get, set, remove: vi.fn() },
      onChanged: { addListener: vi.fn(), removeListener: vi.fn() },
    },
  });
});
afterEach(() => vi.unstubAllGlobals());
const record = (clipId: string) => ({
  ownerScope: "alice",
  workspaceId: "workspace",
  sourceId: `web-clipper:${clipId}`,
  body: {
    clip_id: clipId,
    clip_type: "article",
    source_url: "https://example.org",
    source_title: "Article",
    workspace: { workspace_id: "workspace" },
    content: { full_extract: "Immutable" },
  },
});

it("retains both capture checkpoints when independent contexts read before either write", async () => {
  const first = await import("../research-workspace-prefill");
  vi.resetModules();
  const second = await import("../research-workspace-prefill");
  let reads = 0;
  let release!: () => void;
  const bothRead = new Promise<void>((resolve) => {
    release = resolve;
  });
  const original = get.getMockImplementation()!;
  get.mockImplementation(async (keys) => {
    const snapshot = await original(keys);
    if (keys?.toString().includes("web-captures") && reads < 2) {
      if (++reads === 2) release();
      await bothRead;
    }
    return snapshot;
  });
  await Promise.all([
    first.saveResearchWebCapture(record("one")),
    second.saveResearchWebCapture(record("two")),
  ]);
  expect(
    (await first.readResearchWebCaptures("alice", "workspace"))
      .map((item) => item.body.clip_id)
      .sort(),
  ).toEqual(["one", "two"]);
});

it("rejects a soft extension write failure before accepting a checkpoint", async () => {
  const owning = await import("../research-workspace-prefill");
  set.mockRejectedValueOnce(new Error("Quota"));
  await expect(owning.saveResearchWebCapture(record("one"))).rejects.toThrow();
  expect(await owning.readResearchWebCaptures("alice", "workspace")).toEqual(
    [],
  );
});

it("reads legacy checkpoints and keeps retry bodies immutable across contexts and owners", async () => {
  const legacy = record("legacy");
  entries["__tldw_research_workspace_prefill:alice:web-captures"] =
    JSON.stringify({ legacy });
  const first = await import("../research-workspace-prefill");
  await first.saveResearchWebCapture(record("new"));
  vi.resetModules();
  const second = await import("../research-workspace-prefill");
  expect(
    (await second.readResearchWebCaptures("alice", "workspace"))
      .map((item) => item.body.clip_id)
      .sort(),
  ).toEqual(["legacy", "new"]);
  expect(await second.readResearchWebCaptures("bob", "workspace")).toEqual([]);
  expect(await second.readResearchWebCaptures("alice", "other")).toEqual([]);
  await expect(
    second.saveResearchWebCapture({
      ...legacy,
      body: { ...legacy.body, content: { full_extract: "Changed" } },
    }),
  ).rejects.toThrow("cannot change");
});

it("rejects a silently dropped extension write before accepting a checkpoint", async () => {
  const owning = await import("../research-workspace-prefill");
  set.mockResolvedValueOnce(undefined);
  await expect(owning.saveResearchWebCapture(record("one"))).rejects.toThrow();
  expect(await owning.readResearchWebCaptures("alice", "workspace")).toEqual(
    [],
  );
});
