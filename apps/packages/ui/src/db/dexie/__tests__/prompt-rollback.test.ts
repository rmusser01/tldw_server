import { usePromptEditor } from "@/components/Option/Prompt/hooks/usePromptEditor";
import {
  clearRecipePersistenceScoped,
  forgetRecipePersistenceUnknown,
  markRecipePersistenceScoped,
  markRecipePersistenceUnknown,
  readRecipePersistenceUncertainty,
} from "@/services/recipe-persistence-uncertainty";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import React from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { permanentlyDeletePrompt, restorePromptSnapshot } from "../helpers";

vi.mock("antd", () => ({ notification: { success: vi.fn(), error: vi.fn() } }));

const mocks = vi.hoisted(() => ({
  updateDexie: vi.fn(),
  updateFirefox: vi.fn(),
  deleteDexie: vi.fn(),
  deleteFirefox: vi.fn(),
  getConfig: vi.fn(),
}));

vi.mock(
  "@/services/recipe-persistence-uncertainty",
  async (importOriginal) => ({
    ...(await importOriginal()),
    resolveRecipePersistenceOwnerView: async () => {
      const config = await mocks.getConfig().catch(() => null);
      return config
        ? {
            ownerId: "recipe-owner:sha256:" + "a".repeat(64),
            authorizationRevision: "revision",
          }
        : null;
    },
  }),
);

vi.mock("../chat", () => ({
  PageAssistDatabase: class {
    permanentlyDeletePrompt(id: string) {
      return mocks.deleteDexie(id);
    }
    restorePromptSnapshot(snapshot: unknown) {
      return mocks.updateDexie(snapshot);
    }
  },
}));

vi.mock("../..", () => ({
  deletePromptByIdFB: mocks.deleteFirefox,
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

describe("permanent prompt deletion uncertainty ownership", () => {
  const config = {
    serverUrl: "https://a.test",
    authMode: "single-user" as const,
    apiKey: "test-key",
  };
  const scope = "recipe-owner:sha256:" + "a".repeat(64);
  afterEach(async () => {
    for (const id of ["recipe-exact-id", "other-id"]) {
      await clearRecipePersistenceScoped(id, scope);
      await clearRecipePersistenceScoped(
        id,
        "recipe-owner:sha256:" + "b".repeat(64),
      );
    }
  });
  it("permanent deletion never clears exact-ID unknown quarantine or another owner", async () => {
    mocks.getConfig.mockResolvedValue(config);
    await markRecipePersistenceUnknown("recipe-exact-id");
    await markRecipePersistenceScoped("recipe-exact-id", scope);
    await permanentlyDeletePrompt("recipe-exact-id", scope);
    expect(
      await readRecipePersistenceUncertainty("recipe-exact-id", scope),
    ).toBe("unknown_owner");
    await forgetRecipePersistenceUnknown("recipe-exact-id");
    expect(
      await readRecipePersistenceUncertainty("recipe-exact-id", scope),
    ).toBe("clear");
  });
  it("failed local deletion retains the matching scoped marker", async () => {
    await markRecipePersistenceScoped("recipe-exact-id", scope);
    mocks.deleteFirefox.mockRejectedValueOnce(new Error("disk"));
    await expect(
      permanentlyDeletePrompt("recipe-exact-id", scope),
    ).rejects.toThrow("disk");
    expect(
      await readRecipePersistenceUncertainty("recipe-exact-id", scope),
    ).toBe("scoped");
  });
  it("clears only the current stable owner and deleted ID", async () => {
    mocks.getConfig.mockResolvedValue(config);
    await markRecipePersistenceScoped("recipe-exact-id", scope);
    await markRecipePersistenceScoped(
      "recipe-exact-id",
      "recipe-owner:sha256:" + "b".repeat(64),
    );
    await markRecipePersistenceScoped("other-id", scope);
    await permanentlyDeletePrompt("recipe-exact-id");
    expect(
      await readRecipePersistenceUncertainty(
        "recipe-exact-id",
        "recipe-owner:sha256:" + "b".repeat(64),
      ),
    ).toBe("scoped");
    expect(await readRecipePersistenceUncertainty("other-id", scope)).toBe(
      "scoped",
    );
    expect(
      await readRecipePersistenceUncertainty("recipe-exact-id", scope),
    ).toBe("clear");
  });
  it("cannot clear an owned marker without an available scope", async () => {
    mocks.getConfig.mockRejectedValue(new Error("no config"));
    await markRecipePersistenceScoped("recipe-exact-id", scope);
    await permanentlyDeletePrompt("recipe-exact-id");
    expect(
      await readRecipePersistenceUncertainty("recipe-exact-id", scope),
    ).toBe("scoped");
  });
  it("resolves canonical ownership when the library invokes a delete mutation", async () => {
    mocks.getConfig.mockResolvedValue(config);
    await markRecipePersistenceScoped("recipe-exact-id", scope);
    await markRecipePersistenceScoped(
      "recipe-exact-id",
      "recipe-owner:sha256:" + "b".repeat(64),
    );
    const queryClient = new QueryClient();
    const { result } = renderHook(
      () =>
        usePromptEditor({
          queryClient,
          isOnline: true,
          t: (key) => key,
          guardPrivateMode: () => false,
          getPromptTexts: () => ({ systemText: "", userText: "" }),
          getPromptKeywords: () => [],
          getPromptRecordById: () => undefined,
          confirmDanger: async () => true,
          syncPromptAfterLocalSave: async () => ({
            attempted: false,
            success: true,
          }),
          recipePersistenceAvailable: true,
        }),
      {
        wrapper: ({ children }) =>
          React.createElement(
            QueryClientProvider,
            { client: queryClient },
            React.createElement(MemoryRouter, null, children),
          ),
      },
    );
    act(() => result.current.permanentDeletePromptMutation("recipe-exact-id"));
    await waitFor(async () =>
      expect(
        await readRecipePersistenceUncertainty("recipe-exact-id", scope),
      ).toBe("clear"),
    );
    expect(
      await readRecipePersistenceUncertainty(
        "recipe-exact-id",
        "recipe-owner:sha256:" + "b".repeat(64),
      ),
    ).toBe("scoped");
  });
});
