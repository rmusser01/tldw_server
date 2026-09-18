import { beforeEach, describe, expect, it, vi } from "vitest";

const state = vi.hoisted(() => ({
  rows: [] as Array<Record<string, unknown>>,
}));
vi.mock("../schema", () => ({
  db: {
    messages: {
      where: () => ({
        equals: (id: string) => ({
          modify: async (change: (row: Record<string, unknown>) => void) => {
            const row = state.rows.find((row) => row.id === id);
            if (row) change(row);
          },
        }),
      }),
    },
  },
}));
vi.mock("../chat", () => ({ PageAssistDatabase: class {} }));

import { acknowledgeSavedUserMessage } from "../helpers";

describe("acknowledge an existing local user", () => {
  beforeEach(() => {
    state.rows = [
      {
        id: "user",
        history_id: "owned",
        role: "user",
        content: "Draft",
        createdAt: 12,
        images: [""],
        metadataExtra: { source: "mine" },
      },
    ];
  });

  it("updates only the acknowledgement, preserving content, position and metadata", async () => {
    const original = { ...state.rows[0] };
    await acknowledgeSavedUserMessage("owned", "user", "server-user", "Draft");
    expect(state.rows).toEqual([
      { ...original, serverMessageId: "server-user" },
    ]);
    await acknowledgeSavedUserMessage("owned", "user", "server-user", "Draft");
    expect(state.rows).toEqual([
      { ...original, serverMessageId: "server-user" },
    ]);
  });

  it.each(["different-history", "assistant", "missing"])(
    "cannot acknowledge %s",
    async (kind) => {
      if (kind === "assistant") state.rows[0].role = "assistant";
      const original = structuredClone(state.rows);
      await acknowledgeSavedUserMessage(
        kind === "different-history" ? "other" : "owned",
        kind === "missing" ? "missing" : "user",
        "server-user",
        "Draft",
      );
      expect(state.rows).toEqual(original);
    },
  );

  it("rejects a contradictory existing acknowledgement", async () => {
    state.rows[0].serverMessageId = "first-server-user";
    await expect(
      acknowledgeSavedUserMessage(
        "owned",
        "user",
        "other-server-user",
        "Draft",
      ),
    ).rejects.toThrow("changed");
    expect(state.rows[0].serverMessageId).toBe("first-server-user");
  });

  it("leaves a newer unsaved edit unacknowledged", async () => {
    state.rows[0].content = "Newer draft";
    await acknowledgeSavedUserMessage("owned", "user", "server-user", "Draft");
    expect(state.rows[0]).not.toHaveProperty("serverMessageId");
  });
});
