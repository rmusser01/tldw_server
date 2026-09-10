import { beforeEach, describe, expect, it, vi } from "vitest";

import { restorePromptSnapshot } from "../helpers";

const mocks = vi.hoisted(() => ({
  updateDexie: vi.fn(),
  updateFirefox: vi.fn(),
}));

vi.mock("../chat", () => ({
  PageAssistDatabase: class {
    restorePromptSnapshot(snapshot: unknown) {
      return mocks.updateDexie(snapshot);
    }
  },
}));

vi.mock("../..", () => ({
  deletePromptByIdFB: vi.fn(),
  getAllPromptsFB: vi.fn(),
  getPromptByIdFB: vi.fn(),
  savePromptFB: vi.fn(),
  restorePromptSnapshotFB: mocks.updateFirefox,
  updatePromptFB: mocks.updateFirefox,
}));

describe("restorePromptSnapshot", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("restores one exact prompt snapshot in both local stores", async () => {
    const snapshot = {
      id: "recipe-exact-id",
      title: "Original recipe",
      name: "Original recipe",
      content: "  exact 🧪\n",
      is_system: true,
      createdAt: 11,
      updatedAt: 12,
      syncStatus: "synced" as const,
      serverId: 44,
      structuredPromptDefinition: null,
    };

    await expect(restorePromptSnapshot(snapshot)).resolves.toBe(
      "recipe-exact-id",
    );
    expect(mocks.updateDexie).toHaveBeenCalledWith(snapshot);
    expect(mocks.updateFirefox).toHaveBeenCalledWith(snapshot);
    expect(mocks.updateDexie.mock.calls[0][0]).not.toBe(snapshot);
    expect(mocks.updateFirefox.mock.calls[0][0]).toBe(
      mocks.updateDexie.mock.calls[0][0],
    );
  });
});
