import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { permanentlyDeletePrompt, restorePromptSnapshot } from "../helpers";
import {
  clearRecipePersistenceUncertainty,
  isRecipePersistenceUncertain,
  markRecipePersistenceUncertain,
} from "@/services/recipe-persistence-uncertainty";
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope";
import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { usePromptEditor } from "@/components/Option/Prompt/hooks/usePromptEditor";

vi.mock("antd", () => ({ notification: { success: vi.fn(), error: vi.fn() } }));

const mocks = vi.hoisted(() => ({
  updateDexie: vi.fn(),
  updateFirefox: vi.fn(),
  deleteDexie: vi.fn(),
  deleteFirefox: vi.fn(),
  getConfig: vi.fn(),
}));

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: mocks.getConfig },
}));

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
  const scope = buildChatSurfaceScopeKeyFromConfig(config);
  afterEach(() => {
    for (const id of ["recipe-exact-id", "other-id"]) {
      clearRecipePersistenceUncertainty(id, scope);
      clearRecipePersistenceUncertainty(id, "other-owner");
    }
  });
  it("clears only the current stable owner and deleted ID", async () => {
    mocks.getConfig.mockResolvedValue(config);
    markRecipePersistenceUncertain("recipe-exact-id", scope);
    markRecipePersistenceUncertain("recipe-exact-id", "other-owner");
    markRecipePersistenceUncertain("other-id", scope);
    await permanentlyDeletePrompt("recipe-exact-id");
    expect(isRecipePersistenceUncertain("recipe-exact-id", "other-owner")).toBe(
      true,
    );
    expect(isRecipePersistenceUncertain("other-id", scope)).toBe(true);
    expect(isRecipePersistenceUncertain("recipe-exact-id", scope)).toBe(false);
  });
  it("cannot clear an owned marker without an available scope", async () => {
    mocks.getConfig.mockRejectedValue(new Error("no config"));
    markRecipePersistenceUncertain("recipe-exact-id", scope);
    await permanentlyDeletePrompt("recipe-exact-id");
    expect(isRecipePersistenceUncertain("recipe-exact-id", scope)).toBe(true);
  });
  it("resolves canonical ownership when the library invokes a delete mutation", async () => {
    mocks.getConfig.mockResolvedValue(config);
    markRecipePersistenceUncertain("recipe-exact-id", scope);
    markRecipePersistenceUncertain("recipe-exact-id", "other-owner");
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
    await waitFor(() =>
      expect(isRecipePersistenceUncertain("recipe-exact-id", scope)).toBe(
        false,
      ),
    );
    expect(isRecipePersistenceUncertain("recipe-exact-id", "other-owner")).toBe(
      true,
    );
  });
});
