import { autoSyncPrompt, pullFromStudio } from "@/services/prompt-sync";
import {
  clearRecipePersistenceUncertainty,
  getRecipeAuthenticatedPrincipal,
  isRecipePersistenceUncertain,
  markRecipePersistenceUncertain,
} from "@/services/recipe-persistence-uncertainty";
import { tldwRequest } from "@/services/tldw/request-core";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PromptRecipeBuilder } from "../PromptRecipeBuilder";
import { CLEAR_TASK_RECIPE } from "../built-in-recipes";

const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  resolveConfig: vi.fn(),
  fetch: vi.fn(),
  extension: false,
  sendMessage: vi.fn(),
  rows: new Map<string, any>(),
  beforeRead: vi.fn(),
  markerFails: true,
}));
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return mocks.extension ? "test-extension" : undefined;
      },
      sendMessage: (...args: unknown[]) => mocks.sendMessage(...args),
    },
  },
}));
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({}) }));
vi.mock("@/services/tldw/direct-browser-config", () => ({
  resolveDirectBrowserConfig: () => mocks.resolveConfig(),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: async () => mocks.config },
}));
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
}));
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => ({
    defaultProjectId: 42,
    autoSyncWorkspacePrompts: true,
  }),
  setPromptStudioDefaults: vi.fn(),
}));
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }));
vi.mock("@/db/dexie/schema", () => ({
  db: {
    prompts: {
      get: async (id: string) => {
        await mocks.beforeRead();
        return mocks.rows.get(id);
      },
      update: async (id: string, fields: object) =>
        mocks.rows.set(id, { ...mocks.rows.get(id), ...fields }),
      add: async (row: { id: string }) => mocks.rows.set(row.id, row),
      where: (field: string) => ({
        equals: (value: unknown) => ({
          first: async () =>
            [...mocks.rows.values()].find((row) => row[field] === value),
        }),
      }),
    },
  },
}));
vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "dispatch-id",
  getAllPrompts: async () => structuredClone([...mocks.rows.values()]),
  savePrompt: async (fields: object) => {
    const row = {
      ...fields,
      id: "dispatch-id",
      createdAt: 1,
      syncStatus: "local",
    };
    mocks.rows.set(row.id, row);
    return row;
  },
  updatePrompt: async (row: { id: string }) => {
    mocks.rows.set(row.id, structuredClone(row));
    return row.id;
  },
  markPromptSyncError: async (id: string) => {
    if (mocks.markerFails) throw new Error("storage unavailable");
    mocks.rows.get(id).syncStatus = "error";
  },
  permanentlyDeletePrompt: async (id: string) => mocks.rows.delete(id),
  restorePromptSnapshot: async (row: { id: string }) =>
    mocks.rows.set(row.id, row),
}));
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }));
vi.mock("@/utils/is-private-mode", () => ({ isFireFoxPrivateMode: false }));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}));

const config = (serverUrl: string, sub: string, exp = 1) => ({
  serverUrl,
  authMode: "multi-user",
  accessToken: `header.${btoa(JSON.stringify({ sub, exp }))}.signature`,
});
const owner = config("https://a.test", "alice");
const switchedOwners = [
  config("https://b.test", "alice"),
  config("https://a.test", "bob"),
];
// Hand-checked owner fixtures use authoritative /auth/me IDs, not JWT subjects.
const ownerScope =
  "recipe-owner:sha256:5b2ffba615c995b4d48f10792412bf3107311878ebde8dc5f456aa15bd1e272e";
const switchedScopes = [
  "recipe-owner:sha256:ac377847397194455771f75d58d05b33df00a4790feea64143871b76ba4eadd0",
  "recipe-owner:sha256:a300c788423de96bdbd3ebd8d2327bb1ef101fba64fa1d58fbedf37c31e68bf4",
];
const promptRequests = () =>
  mocks.fetch.mock.calls.filter(([url]) =>
    new URL(String(url)).pathname.startsWith("/api/v1/prompt-studio/prompts/"),
  );
const promptMutations = () =>
  promptRequests().filter(([, init]) => init.method !== "GET");
const principalResponse = (init: RequestInit) => {
  const bearer = new Headers(init.headers).get("Authorization");
  if (bearer === `Bearer ${owner.accessToken}`)
    return jsonResponse({ id: "authoritative-alice" });
  if (bearer === `Bearer ${switchedOwners[1].accessToken}`)
    return jsonResponse({ id: "authoritative-bob" });
  return jsonResponse({ detail: "Not authenticated" }, 401);
};
const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};
const serverRecord = () => ({
  id: 101,
  project_id: 42,
  name: "Dispatch recipe",
  version_number: 1,
  updated_at: "2026-09-10T00:00:00Z",
  prompt_format: "structured",
  prompt_schema_version: 2,
  prompt_definition: CLEAR_TASK_RECIPE.definition,
});
const jsonResponse = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
const renderBuilder = (scope: string) =>
  render(
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }>
      <PromptRecipeBuilder
        target="system"
        persistenceScope={scope}
        onApply={() => {}}
        onBack={() => {}}
        capabilities={{
          availability: "available",
          single_text_recipe_v2: { supported: true },
          prompt_improvement_v1: { supported: true, limits: null },
          prompt_persistence: {
            create_authorized: true,
            update_authorized: true,
          },
        }}
      />
    </QueryClientProvider>,
  );
const selectSaved = async (user: ReturnType<typeof userEvent.setup>) => {
  await screen.findByRole("option", {
    name: /Dispatch recipe|Untitled recipe/,
  });
  await user.selectOptions(
    screen.getByRole("combobox", { name: "Recipe source" }),
    "saved:dispatch-id",
  );
};

describe("recipe dispatch ownership through real sync and transport", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mocks.rows.clear();
    mocks.config = owner;
    mocks.extension = false;
    mocks.markerFails = true;
    mocks.resolveConfig.mockImplementation(async () => mocks.config);
    mocks.fetch.mockImplementation(async (url, init) =>
      new URL(String(url)).pathname === "/api/v1/auth/me"
        ? principalResponse(init)
        : jsonResponse({
            success: true,
            data: { ...serverRecord(), prompt_schema_version: 999 },
          }),
    );
    vi.stubGlobal("fetch", mocks.fetch);
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE;
  });
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    for (const scope of [ownerScope, ...switchedScopes])
      clearRecipePersistenceUncertainty("dispatch-id", scope);
  });

  it.each(["create", "update", "pull"])(
    "preserves v1 %s with opaque credentials without clearing owned uncertainty",
    async (operation) => {
      const definition = {
        schema_version: 1,
        format: "structured",
        variables: [],
        blocks: [],
        assembly_config: {
          legacy_system_roles: ["system", "developer"],
          legacy_user_roles: ["user"],
          block_separator: "\n\n",
        },
      };
      mocks.config = { ...owner, accessToken: "opaque-v1-token" };
      mocks.rows.set("dispatch-id", {
        id: "dispatch-id",
        title: "V1 prompt",
        syncStatus: "local",
        promptFormat: "structured",
        promptSchemaVersion: 1,
        structuredPromptDefinition: definition,
        ...(operation === "update" ? { serverId: 101 } : {}),
      });
      markRecipePersistenceUncertain("dispatch-id", ownerScope);
      mocks.fetch.mockImplementation(async (url, init) =>
        new URL(String(url)).pathname === "/api/v1/auth/me"
          ? principalResponse(init)
          : jsonResponse({
              success: true,
              data: {
                ...serverRecord(),
                prompt_schema_version: 1,
                prompt_definition: definition,
              },
            }),
      );
      const result =
        operation === "pull"
          ? await pullFromStudio(101, "dispatch-id")
          : await autoSyncPrompt("dispatch-id", 42);
      expect(result).toMatchObject({ success: true, persistenceScope: null });
      expect(promptRequests()).toHaveLength(1);
      expect(promptMutations()).toHaveLength(operation === "pull" ? 0 : 1);
      expect(isRecipePersistenceUncertain("dispatch-id", ownerScope)).toBe(
        true,
      );
    },
  );

  it.each(["create", "update"])(
    "rejects a v2 %s with unknown identity before dispatch as a known failure",
    async (operation) => {
      mocks.config = { ...owner, accessToken: "opaque-v2-token" };
      mocks.rows.set("dispatch-id", {
        id: "dispatch-id",
        title: "V2 recipe",
        syncStatus: "local",
        promptFormat: "structured",
        promptSchemaVersion: 2,
        structuredPromptDefinition: CLEAR_TASK_RECIPE.definition,
        ...(operation === "update" ? { serverId: 101 } : {}),
      });
      expect(await autoSyncPrompt("dispatch-id", 42)).toMatchObject({
        success: false,
        failureKind: "validation",
        persistenceScope: null,
      });
      expect(promptRequests()).toHaveLength(0);
      expect(isRecipePersistenceUncertain("dispatch-id", ownerScope)).toBe(
        false,
      );
    },
  );

  for (const operation of ["create", "update"] as const) {
    for (const [index, nextOwner] of switchedOwners.entries()) {
      it(`${operation}: ${index === 0 ? "backend" : "principal"} switch fails before mutation until legacy sync supplies owner and local ID`, async () => {
        const user = userEvent.setup();
        if (operation === "update")
          mocks.rows.set("dispatch-id", {
            id: "dispatch-id",
            title: "Dispatch recipe",
            content: "",
            is_system: true,
            createdAt: 1,
            syncStatus: "synced",
            serverId: 101,
            promptFormat: "structured",
            promptSchemaVersion: 2,
            structuredPromptDefinition: structuredClone(
              CLEAR_TASK_RECIPE.definition,
            ),
          });
        const originalRow = structuredClone(mocks.rows.get("dispatch-id"));
        const view = renderBuilder(ownerScope);
        if (operation === "update") await selectSaved(user);
        const gate = deferred();
        mocks.beforeRead.mockImplementationOnce(() => gate.promise);
        await user.click(
          screen.getByRole("button", {
            name:
              operation === "create" ? "Save as new recipe" : "Update recipe",
          }),
        );
        await waitFor(() => expect(mocks.beforeRead).toHaveBeenCalled());
        mocks.config = nextOwner;
        gate.resolve();
        await screen.findByText(
          operation === "create"
            ? "Could not save the recipe. Try again."
            : "Could not update the recipe. Try again.",
        );
        const scope = switchedScopes[index];
        expect(promptRequests()).toHaveLength(0);
        const authRequest = mocks.fetch.mock.calls.find(
          ([url]) => new URL(String(url)).pathname === "/api/v1/auth/me",
        );
        expect(authRequest?.[0]).toBe(`${nextOwner.serverUrl}/api/v1/auth/me`);
        expect(new Headers(authRequest?.[1].headers).get("Authorization")).toBe(
          `Bearer ${nextOwner.accessToken}`,
        );
        expect(isRecipePersistenceUncertain("dispatch-id", scope)).toBe(false);
        expect(isRecipePersistenceUncertain("dispatch-id", ownerScope)).toBe(
          false,
        );
        expect(mocks.rows.get("dispatch-id")).toEqual(originalRow);
        view.unmount();
        const actualView = renderBuilder(scope);
        if (operation === "update") await selectSaved(user);
        expect(
          screen.getByRole("button", { name: "Save as new recipe" }),
        ).toBeEnabled();
        await user.click(
          screen.getByRole("button", {
            name:
              operation === "create" ? "Save as new recipe" : "Update recipe",
          }),
        );
        await screen.findByText(
          operation === "create"
            ? "Could not save the recipe. Try again."
            : "Could not update the recipe. Try again.",
        );
        expect(promptMutations()).toHaveLength(0);
        expect(isRecipePersistenceUncertain("dispatch-id", scope)).toBe(false);
        actualView.unmount();
        const staleView = renderBuilder(ownerScope);
        if (operation === "update") await selectSaved(user);
        expect(
          screen.getByRole("button", { name: "Save as new recipe" }),
        ).toBeEnabled();
        staleView.unmount();
      });
    }
  }

  it.each(["create", "update"])(
    "%s returns the transport owner when config changes during resolution",
    async (operation) => {
      mocks.rows.set("dispatch-id", {
        id: "dispatch-id",
        title: "Task",
        syncStatus: "local",
        ...(operation === "update" ? { serverId: 101 } : {}),
      });
      mocks.resolveConfig.mockImplementationOnce(async () => {
        mocks.config = switchedOwners[0];
        return mocks.config;
      });
      const result = await autoSyncPrompt("dispatch-id", 42);
      expect(result).toMatchObject({
        failureKind: "invalid_server_payload",
        persistenceScope: switchedScopes[0],
        localId: "dispatch-id",
      });
      expect(promptRequests()).toHaveLength(1);
      expect(promptMutations()).toHaveLength(1);
      expect(String(promptRequests()[0][0])).toMatch(
        /^https:\/\/b\.test\/api\/v1\/prompt-studio\/prompts\//,
      );
    },
  );

  it("returns the extension background dispatch owner, not the page config", async () => {
    mocks.extension = true;
    mocks.rows.set("dispatch-id", {
      id: "dispatch-id",
      title: "Task",
      syncStatus: "local",
    });
    mocks.sendMessage.mockImplementation(async ({ payload }) =>
      tldwRequest(payload, {
        getConfig: async () => switchedOwners[1],
        useRuntimeAuthOverride: false,
        getAuthenticatedPrincipal: getRecipeAuthenticatedPrincipal,
      }),
    );
    const result = await autoSyncPrompt("dispatch-id", 42);
    expect(result).toMatchObject({
      failureKind: "invalid_server_payload",
      persistenceScope: switchedScopes[1],
    });
    expect(
      new Headers(promptRequests()[0][1].headers).get("Authorization"),
    ).toBe(`Bearer ${switchedOwners[1].accessToken}`);
    expect(promptMutations()).toHaveLength(1);
  });

  it("does not fall back to direct prompt writes after ambiguous extension delivery", async () => {
    mocks.extension = true;
    mocks.rows.set("dispatch-id", {
      id: "dispatch-id",
      title: "Task",
      syncStatus: "local",
    });
    mocks.sendMessage.mockRejectedValue(new Error("message channel closed"));
    expect(await autoSyncPrompt("dispatch-id", 42)).toMatchObject({
      success: false,
      failureKind: "invalid_server_payload",
      persistenceScope: null,
    });
    expect(promptMutations()).toHaveLength(0);
    expect(mocks.fetch).not.toHaveBeenCalled();
  });
});
