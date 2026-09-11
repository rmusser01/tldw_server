import {
  autoSyncPrompt,
  pullFromStudio,
  pushToStudio,
} from "@/services/prompt-sync";
import {
  clearRecipePersistenceScoped,
  forgetRecipePersistenceUnknown,
  markRecipePersistenceScoped,
  markRecipePersistenceUnknown,
  readRecipePersistenceUncertainty,
  resolveRecipePersistenceOwnerView,
} from "@/services/recipe-persistence-uncertainty";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PromptRecipeBuilder } from "../PromptRecipeBuilder";
import { CLEAR_TASK_RECIPE } from "../built-in-recipes";

type BackgroundListener = (
  message: unknown,
  sender: { id: string },
  reply: (value: unknown) => void,
) => unknown;
const mocks = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  resolveConfig: vi.fn(),
  beforeConfig: vi.fn(),
  beforeDefaults: vi.fn(),
  beforePendingUpdate: vi.fn(),
  defaultProjectId: 42 as number | null,
  fetch: vi.fn(),
  extension: false,
  sendMessage: vi.fn(),
  rows: new Map<string, Record<string, unknown>>(),
  beforeRead: vi.fn(),
  markerFails: true,
  failedDurableErrorWrite: vi.fn(),
  runtimeKey: null as string | null,
  listeners: new Set<BackgroundListener>(),
}));
vi.mock("@/entries/shared/background-init", () => ({
  MODEL_WARM_ALARM_NAME: "warm",
  initBackground: async () => {},
}));
vi.mock("@/entries/shared/notification-subscription", () => ({
  startNotificationSubscription: async () => {},
}));
vi.mock("wxt/browser", () => {
  const event = () => ({ addListener: vi.fn() });
  return {
    browser: {
      runtime: {
        get id() {
          return mocks.extension ? "test-extension" : undefined;
        },
        getURL: (path: string) => "chrome-extension://test-extension" + path,
        sendMessage: (...args: unknown[]) => mocks.sendMessage(...args),
        onConnect: event(),
        onStartup: event(),
        onMessage: {
          addListener: (listener: BackgroundListener) =>
            mocks.listeners.add(listener),
        },
      },
      storage: {
        local: { get: async () => ({}), set: async () => {} },
        session: { get: async () => ({}), set: async () => {} },
        onChanged: event(),
      },
      alarms: {
        clear: async () => true,
        create: async () => {},
        onAlarm: event(),
      },
      tabs: {
        query: async () => [],
        create: vi.fn(),
        sendMessage: async () => {},
      },
      action: { onClicked: event() },
      contextMenus: { create: vi.fn(), removeAll: vi.fn(), onClicked: event() },
      i18n: { getMessage: (key: string) => key },
    },
  };
});
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serialize: (v: unknown) => v,
    deserialize: (v: unknown) => v,
  },
  createSafeStorage: () => ({
    get: async (key: string) => {
      if (key === "tldwConfig") {
        await mocks.beforeConfig();
        return mocks.config;
      }
      return undefined;
    },
    set: async () => {},
    remove: async () => {},
  }),
}));
vi.mock("@/services/tldw/direct-browser-config", () => ({
  resolveDirectBrowserConfig: () => mocks.resolveConfig(),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: async () => mocks.config },
}));
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => mocks.runtimeKey,
}));
vi.mock("@/services/prompt-studio-settings", () => ({
  getPromptStudioDefaults: async () => {
    await mocks.beforeDefaults();
    return {
      defaultProjectId: mocks.defaultProjectId,
      autoSyncWorkspacePrompts: true,
    };
  },
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
      update: async (
        id: string,
        fields: object | ((row: Record<string, unknown>) => void | boolean),
      ) => {
        if (
          typeof fields === "function" ||
          ("syncStatus" in fields && fields.syncStatus === "pending")
        ) {
          await mocks.beforePendingUpdate();
        }
        if (
          "syncStatus" in fields &&
          fields.syncStatus === "error" &&
          mocks.markerFails
        ) {
          mocks.failedDurableErrorWrite();
          throw new Error("durable sync-error storage unavailable");
        }
        if (!mocks.rows.has(id)) return 0;
        const row = { ...mocks.rows.get(id) };
        if (typeof fields === "function") {
          if (fields(row) === false) return 1;
        } else Object.assign(row, fields);
        mocks.rows.set(id, row);
        return 1;
      },
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
  if (
    bearer === `Bearer ${owner.accessToken}` ||
    bearer === `Bearer ${config("https://a.test", "alice", 2).accessToken}`
  )
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

const seed = (operation: string) => {
  if (operation !== "update") return;
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
    structuredPromptDefinition: structuredClone(CLEAR_TASK_RECIPE.definition),
  });
};
const clickWrite = async (
  user: ReturnType<typeof userEvent.setup>,
  operation: string,
) =>
  user.click(
    screen.getByRole("button", {
      name: operation === "create" ? "Save as new recipe" : "Update recipe",
    }),
  );
const startBackground = async () => {
  mocks.extension = true;
  mocks.listeners.clear();
  Object.defineProperty(globalThis, "defineBackground", {
    configurable: true,
    value: (v: unknown) => v,
  });
  const background = (await import("@/entries/background")).default;
  background.main();
  mocks.sendMessage.mockImplementation(
    (message: unknown) =>
      new Promise((resolve) => {
        const listener = [...mocks.listeners][0];
        if (!listener) throw new Error("Missing real background listener");
        listener(message, { id: "test-extension" }, resolve);
      }),
  );
};

describe("builder through real sync, Prompt Studio, apiSend and request-core", () => {
  it.each(["request", "reply"])(
    "background preserves known dispatch when the acknowledgement %s is lost",
    async (lost) => {
      await startBackground();
      seed("update");
      delete mocks.rows.get("dispatch-id").serverId;
      const deliver = mocks.sendMessage.getMockImplementation()!;
      mocks.sendMessage.mockImplementation(async (message) => {
        if (message.type === "tldw:recipe-uncertainty:acknowledge") {
          if (lost === "reply") await deliver(message);
          throw new Error("lost acknowledgement " + lost);
        }
        return deliver(message);
      });
      expect(
        await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope }),
      ).toMatchObject({
        success: false,
        recipeOwnership: {
          dispatch: { state: "dispatched", actualOwnerId: ownerScope },
        },
      });
      mocks.sendMessage.mockImplementation(deliver);
      mocks.config = switchedOwners[1];
      const nextOwner = (await resolveRecipePersistenceOwnerView())!.ownerId;
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", nextOwner),
      ).toBe(lost === "request" ? "unknown_owner" : "clear");
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: nextOwner });
      expect(promptMutations()).toHaveLength(lost === "request" ? 1 : 2);
    },
  );

  it.each(["success", "typed rejection"])(
    "%s cannot clear an undelivered background receipt acknowledgement",
    async (outcome) => {
      await startBackground();
      seed("update");
      const deliver = mocks.sendMessage.getMockImplementation()!;
      mocks.sendMessage.mockImplementation(async (message) => {
        if (message.type === "tldw:recipe-uncertainty:acknowledge")
          throw new Error("acknowledgement request unavailable");
        return deliver(message);
      });
      mocks.fetch.mockImplementation(async (url, init) => {
        if (new URL(String(url)).pathname === "/api/v1/auth/me")
          return principalResponse(init);
        return outcome === "success"
          ? jsonResponse({ success: true, data: serverRecord() })
          : jsonResponse(
              {
                detail: [
                  {
                    loc: ["body", "name"],
                    msg: "Field required",
                    type: "missing",
                  },
                ],
              },
              422,
            );
      });
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
      expect(
        await readRecipePersistenceUncertainty(
          "dispatch-id",
          switchedScopes[1],
        ),
      ).toBe("unknown_owner");
      await clearRecipePersistenceScoped("dispatch-id", ownerScope);
      expect(
        await readRecipePersistenceUncertainty(
          "dispatch-id",
          switchedScopes[1],
        ),
      ).toBe("unknown_owner");
      const requests = mocks.fetch.mock.calls.length;
      await forgetRecipePersistenceUnknown("dispatch-id");
      expect(
        await readRecipePersistenceUncertainty(
          "dispatch-id",
          switchedScopes[1],
        ),
      ).toBe("clear");
      expect(mocks.fetch).toHaveBeenCalledTimes(requests);
    },
  );

  it("live background quarantines an unacknowledged POST across owners when both recovery writes fail", async () => {
    await startBackground();
    seed("update");
    delete mocks.rows.get("dispatch-id").serverId;
    const deliver = mocks.sendMessage.getMockImplementation()!;
    let lostResponse = false;
    let lostUnknownMarker = false;
    mocks.sendMessage.mockImplementation(async (message) => {
      if (message.type === "tldw:request") {
        await deliver(message);
        lostResponse = true;
        throw new Error("response channel closed after POST");
      }
      if (message.type === "tldw:recipe-uncertainty:mark-unknown") {
        lostUnknownMarker = true;
        throw new Error("follow-up channel unavailable");
      }
      return deliver(message);
    });
    expect(
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope }),
    ).toMatchObject({
      success: false,
      recipeOwnership: { dispatch: { state: "unknown" } },
    });
    expect(lostResponse && lostUnknownMarker).toBe(true);
    expect(mocks.failedDurableErrorWrite).toHaveBeenCalledTimes(1);
    expect(mocks.rows.get("dispatch-id").syncStatus).not.toBe("error");
    expect(promptMutations()).toHaveLength(1);
    expect(promptMutations()[0][1].method).toBe("POST");
    mocks.sendMessage.mockImplementation(deliver);
    mocks.config = switchedOwners[1];
    const nextOwner = (await resolveRecipePersistenceOwnerView())!.ownerId;
    expect(nextOwner).not.toBe(ownerScope);
    expect
      .soft(await readRecipePersistenceUncertainty("dispatch-id", nextOwner))
      .toBe("unknown_owner");
    expect(
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: nextOwner }),
    ).toMatchObject({
      success: false,
      recipeWriteBlocked: true,
      recipeOwnership: { dispatch: { state: "not_dispatched" } },
    });
    expect(promptMutations()).toHaveLength(1);
  });

  it.each(["direct", "background"])(
    "%s received ambiguity stays scoped and does not block another owner after durable storage fails",
    async (adapter) => {
      if (adapter === "background") await startBackground();
      seed("update");
      delete mocks.rows.get("dispatch-id").serverId;
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("scoped");
      mocks.config = switchedOwners[1];
      const nextOwner = (await resolveRecipePersistenceOwnerView())!.ownerId;
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", nextOwner),
      ).toBe("clear");
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: nextOwner });
      expect(promptMutations()).toHaveLength(2);
    },
  );

  it.each([
    ["create", "scoped", false],
    ["update", "scoped", false],
    ["create", "unavailable", false],
    ["update", "unavailable", false],
    ["create", "scoped", true],
    ["update", "scoped", true],
    ["create", "unavailable", true],
    ["update", "unavailable", true],
  ] as const)(
    "background %s retains recovery after delayed %s authority (durable write fails: %s)",
    async (operation, authorityResult, markerFails) => {
      await startBackground();
      seed(operation);
      mocks.markerFails = markerFails;
      mocks.defaultProjectId = null;
      const user = userEvent.setup();
      const view = renderBuilder(ownerScope);
      if (operation === "update") await selectSaved(user);
      await user.selectOptions(
        screen.getByRole("combobox", { name: "Output format" }),
        "markdown",
      );
      const reached = deferred();
      const release = deferred();
      const deliver = mocks.sendMessage.getMockImplementation()!;
      let pauseNextRead = true;
      mocks.sendMessage.mockImplementation(async (message) => {
        if (pauseNextRead && message.type === "tldw:recipe-uncertainty:read") {
          pauseNextRead = false;
          reached.resolve();
          await release.promise;
          if (authorityResult === "unavailable")
            throw new Error("background connection lost");
        }
        return deliver(message);
      });
      await clickWrite(user, operation);
      await reached.promise;
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
      const recoveryRow = structuredClone(mocks.rows.get("dispatch-id"));
      expect(recoveryRow.syncStatus).toBe(
        markerFails ? (operation === "create" ? "local" : "synced") : "error",
      );
      release.resolve();
      await screen.findByText(
        /Could not (?:save|update) the recipe|server outcome could not be verified/i,
      );
      expect.soft(mocks.rows.get("dispatch-id")).toEqual(recoveryRow);
      expect
        .soft(screen.queryByText(/server outcome could not be verified/i))
        .toBeInTheDocument();
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("scoped");
      expect(promptMutations()).toHaveLength(1);
      view.unmount();
    },
  );

  it.each(["scoped", "unavailable"])(
    "manual background sync returns a fresh lock-aware result after delayed %s authority",
    async (authorityResult) => {
      await startBackground();
      seed("update");
      mocks.markerFails = false;
      const reached = deferred();
      const release = deferred();
      const deliver = mocks.sendMessage.getMockImplementation()!;
      let pauseNextRead = true;
      mocks.sendMessage.mockImplementation(async (message) => {
        if (pauseNextRead && message.type === "tldw:recipe-uncertainty:read") {
          pauseNextRead = false;
          reached.resolve();
          await release.promise;
          if (authorityResult === "unavailable")
            throw new Error("background connection lost");
        }
        return deliver(message);
      });
      const pending = pushToStudio("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      await reached.promise;
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
      release.resolve();
      expect(await pending).toMatchObject({
        success: false,
        recipeWriteBlocked: true,
        syncStatus: "error",
        failureKind: "validation",
        recipeOwnership: {
          dispatch: { state: "not_dispatched", actualOwnerId: null },
        },
      });
      expect(promptMutations()).toHaveLength(1);
    },
  );

  it.each(["direct", "background"])(
    "%s known-project transient fallback preserves concurrent durable ambiguity",
    async (adapter) => {
      if (adapter === "background") await startBackground();
      seed("update");
      mocks.markerFails = false;
      const reached = deferred();
      const release = deferred();
      mocks.beforeRead
        .mockResolvedValueOnce(undefined)
        .mockImplementationOnce(async () => {
          reached.resolve();
          await release.promise;
          throw new Error("transient local read failure");
        });
      const pending = autoSyncPrompt("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      await reached.promise;
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
      expect(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
      release.resolve();
      expect.soft(await pending).toMatchObject({
        success: false,
        failureKind: "validation",
        syncStatus: "error",
        recipeWriteBlocked: true,
        recipeOwnership: {
          dispatch: { state: "not_dispatched", actualOwnerId: null },
        },
      });
      expect.soft(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
      vi.resetModules();
      if (adapter === "background") await startBackground();
      const restartedSync = await import("@/services/prompt-sync");
      const restartedRegistry = await import(
        "@/services/recipe-persistence-uncertainty"
      );
      expect(
        await restartedRegistry.readRecipePersistenceUncertainty(
          "dispatch-id",
          ownerScope,
        ),
      ).toBe("clear");
      await restartedSync.autoSyncPrompt("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      expect(promptMutations()).toHaveLength(1);
    },
  );

  it.each([
    ["direct", "create"],
    ["background", "create"],
    ["direct", "update"],
    ["background", "update"],
  ])(
    "%s %s retains the exact local row when project-less sync reports concurrent durable ambiguity",
    async (adapter, operation) => {
      if (adapter === "background") await startBackground();
      seed(operation);
      mocks.defaultProjectId = null;
      mocks.markerFails = false;
      const user = userEvent.setup();
      const view = renderBuilder(ownerScope);
      if (operation === "update") await selectSaved(user);
      await user.selectOptions(
        screen.getByRole("combobox", { name: "Output format" }),
        "markdown",
      );
      const reached = deferred();
      const release = deferred();
      mocks.beforeDefaults
        .mockResolvedValueOnce(undefined)
        .mockImplementationOnce(async () => {
          reached.resolve();
          await release.promise;
        });
      await clickWrite(user, operation);
      await reached.promise;
      expect(mocks.fetch).not.toHaveBeenCalled();
      await pushToStudio("dispatch-id", 42, { expectedOwnerId: ownerScope });
      expect(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
      release.resolve();

      await screen.findByText(
        /Could not (?:save|update) the recipe|server outcome could not be verified/i,
      );
      expect.soft(mocks.rows.get("dispatch-id")).toMatchObject({
        id: "dispatch-id",
        syncStatus: "error",
        structuredPromptDefinition: {
          assembly_config: { render_format: "markdown" },
        },
      });
      expect(
        await screen.findByText(/server outcome could not be verified/i),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", {
          name: operation === "create" ? "Save as new recipe" : "Update recipe",
        }),
      ).toBeDisabled();
      expect(
        screen.queryByText(
          "Recipe saved locally and will sync when the server is available.",
        ),
      ).not.toBeInTheDocument();
      expect(promptMutations()).toHaveLength(1);
      view.unmount();

      vi.resetModules();
      if (adapter === "background") await startBackground();
      const restartedSync = await import("@/services/prompt-sync");
      const restartedRegistry = await import(
        "@/services/recipe-persistence-uncertainty"
      );
      expect(
        await restartedRegistry.readRecipePersistenceUncertainty(
          "dispatch-id",
          ownerScope,
        ),
      ).toBe("clear");
      expect(
        await restartedSync.autoSyncPrompt("dispatch-id", 42, {
          expectedOwnerId: ownerScope,
        }),
      ).toMatchObject({ success: false, syncStatus: "error" });
      expect(promptMutations()).toHaveLength(1);
    },
  );

  it.each([
    ["direct", "defaults"],
    ["background", "defaults"],
    ["direct", "local pending write"],
    ["background", "local pending write"],
  ])(
    "%s project-less completion paused at %s preserves concurrent ambiguity across restart",
    async (adapter, pauseAt) => {
      if (adapter === "background") await startBackground();
      seed("update");
      mocks.defaultProjectId = null;
      mocks.markerFails = false;
      const reached = deferred();
      const release = deferred();
      const pause = async () => {
        reached.resolve();
        await release.promise;
      };
      if (pauseAt === "defaults")
        mocks.beforeDefaults.mockImplementationOnce(pause);
      else mocks.beforePendingUpdate.mockImplementationOnce(pause);

      const pending = autoSyncPrompt("dispatch-id", undefined, {
        expectedOwnerId: ownerScope,
      });
      await reached.promise;
      expect(mocks.fetch).not.toHaveBeenCalled();
      const ambiguous = await pushToStudio("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      expect(ambiguous).toMatchObject({
        success: false,
        syncStatus: "error",
        recipeOwnership: {
          dispatch: { state: "dispatched", actualOwnerId: ownerScope },
        },
      });
      expect(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
      expect(promptMutations()).toHaveLength(1);

      release.resolve();
      const result = await pending;
      expect.soft(result).toMatchObject({
        success: false,
        failureKind: "validation",
        syncStatus: "error",
        recipeOwnership: { dispatch: { state: "not_dispatched" } },
      });
      expect.soft(mocks.rows.get("dispatch-id").syncStatus).toBe("error");
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("scoped");

      vi.resetModules();
      if (adapter === "background") await startBackground();
      const restartedSync = await import("@/services/prompt-sync");
      const restartedRegistry = await import(
        "@/services/recipe-persistence-uncertainty"
      );
      expect(
        await restartedRegistry.readRecipePersistenceUncertainty(
          "dispatch-id",
          ownerScope,
        ),
      ).toBe("clear");
      await restartedSync.autoSyncPrompt("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      expect(promptMutations()).toHaveLength(1);
    },
  );

  it.each([
    ["direct", "create"],
    ["direct", "update"],
    ["background", "create"],
    ["background", "update"],
  ])(
    "%s %s retains project-less local changes and shows pending recovery without fetching",
    async (adapter, operation) => {
      if (adapter === "background") await startBackground();
      seed(operation);
      mocks.defaultProjectId = null;
      const user = userEvent.setup();
      renderBuilder(ownerScope);
      if (operation === "update") await selectSaved(user);
      await user.selectOptions(
        screen.getByRole("combobox", { name: "Output format" }),
        "markdown",
      );

      await clickWrite(user, operation);
      await waitFor(() =>
        expect(
          screen.getByRole("button", {
            name:
              operation === "create" ? "Save as new recipe" : "Update recipe",
          }),
        ).toBeEnabled(),
      );

      expect(mocks.fetch).not.toHaveBeenCalled();
      expect(mocks.rows.get("dispatch-id")).toMatchObject({
        id: "dispatch-id",
        syncStatus: "pending",
        structuredPromptDefinition: {
          assembly_config: { render_format: "markdown" },
        },
      });
      expect(
        await screen.findByText(
          "Recipe saved locally and will sync when the server is available.",
        ),
      ).toBeInTheDocument();
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("clear");
    },
  );

  it.each(["direct", "background"])(
    "%s performs zero project GET/POST for project-less v2 under a stale owner",
    async (adapter) => {
      if (adapter === "background") await startBackground();
      seed("update");
      delete mocks.rows.get("dispatch-id").serverId;
      mocks.defaultProjectId = null;
      mocks.config = switchedOwners[0];
      const result = await autoSyncPrompt("dispatch-id", undefined, {
        expectedOwnerId: ownerScope,
      });
      expect(result).toMatchObject({
        success: false,
        syncStatus: "pending",
        recipeOwnership: { dispatch: { state: "not_dispatched" } },
      });
      expect(mocks.fetch).not.toHaveBeenCalled();
      expect(mocks.rows.get("dispatch-id").syncStatus).toBe("pending");
    },
  );

  it.each([
    ["direct", "scoped"],
    ["direct", "unknown"],
    ["background", "scoped"],
    ["background", "unknown"],
  ])(
    "%s atomically rejects %s uncertainty introduced after sync preflight",
    async (adapter, state) => {
      if (adapter === "background") await startBackground();
      seed("update");
      const reached = deferred();
      const release = deferred();
      const pause = async () => {
        reached.resolve();
        await release.promise;
        return mocks.config;
      };
      if (adapter === "direct")
        mocks.resolveConfig.mockImplementationOnce(pause);
      else mocks.beforeConfig.mockImplementationOnce(pause);
      const operation = autoSyncPrompt("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      await reached.promise;
      if (state === "scoped")
        await markRecipePersistenceScoped("dispatch-id", ownerScope);
      else await markRecipePersistenceUnknown("dispatch-id");
      release.resolve();
      expect(await operation).toMatchObject({
        success: false,
        recipeWriteBlocked: true,
        recipeOwnership: { dispatch: { state: "not_dispatched" } },
      });
      expect(promptMutations()).toHaveLength(0);
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe(state === "scoped" ? "scoped" : "unknown_owner");
    },
  );

  it.each(["direct", "background"])(
    "%s reserves one concurrent exact-ID mutation",
    async (adapter) => {
      if (adapter === "background") await startBackground();
      seed("update");
      const release = deferred();
      const reached = deferred();
      mocks.fetch.mockImplementation(async (url, init) => {
        if (new URL(String(url)).pathname === "/api/v1/auth/me")
          return principalResponse(init);
        reached.resolve();
        await release.promise;
        return jsonResponse({ malformed: true });
      });
      const first = autoSyncPrompt("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      const second = autoSyncPrompt("dispatch-id", 42, {
        expectedOwnerId: ownerScope,
      });
      await reached.promise;
      release.resolve();
      const results = await Promise.all([first, second]);
      expect(promptMutations()).toHaveLength(1);
      expect(
        results.filter(
          (result) =>
            result.recipeOwnership?.dispatch.state === "not_dispatched",
        ),
      ).toHaveLength(1);
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("scoped");
    },
  );

  it.each([
    ["direct", "create"],
    ["direct", "update"],
    ["background", "create"],
    ["background", "update"],
  ])(
    "%s %s clears a typed no-mutation rejection before exact rollback",
    async (adapter, operation) => {
      if (adapter === "background") await startBackground();
      seed(operation);
      const original = structuredClone(mocks.rows.get("dispatch-id"));
      mocks.fetch.mockImplementation(async (url, init) =>
        new URL(String(url)).pathname === "/api/v1/auth/me"
          ? principalResponse(init)
          : jsonResponse(
              {
                detail: [
                  {
                    loc: ["body", "name"],
                    msg: "Field required",
                    type: "missing",
                  },
                ],
              },
              422,
            ),
      );
      const user = userEvent.setup();
      renderBuilder(ownerScope);
      if (operation === "update") await selectSaved(user);
      await clickWrite(user, operation);
      await screen.findByText(
        operation === "create"
          ? "Could not save the recipe. Try again."
          : "Could not update the recipe. Try again.",
      );
      expect(mocks.rows.get("dispatch-id")).toEqual(original);
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("clear");
      expect(promptMutations()).toHaveLength(1);
    },
  );
  beforeEach(() => {
    vi.resetAllMocks();
    mocks.rows.clear();
    mocks.config = owner;
    mocks.extension = false;
    mocks.runtimeKey = null;
    mocks.markerFails = true;
    mocks.defaultProjectId = 42;
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
  afterEach(async () => {
    cleanup();
    for (const scope of [ownerScope, ...switchedScopes])
      await clearRecipePersistenceScoped("dispatch-id", scope);
    await forgetRecipePersistenceUnknown("dispatch-id");
    vi.unstubAllGlobals();
  });

  for (const adapter of ["direct", "background"]) {
    for (const operation of ["create", "update"]) {
      for (const change of ["backend", "principal", "auth-source"]) {
        it(
          adapter +
            " " +
            operation +
            ": " +
            change +
            " rejects stale A, dispatches B once, and keeps B locked after storage failure/reopen",
          async () => {
            if (change === "auth-source")
              mocks.config = {
                serverUrl: "https://a.test",
                authMode: "single-user",
                apiKey: "source-key",
                credentialSource: "manual",
                apiKeyPersistence: "device",
                apiKeyServerOrigin: "https://a.test",
              };
            if (adapter === "background") await startBackground();
            const originalOwner = (await resolveRecipePersistenceOwnerView())!
              .ownerId;
            seed(operation);
            const originalRow = structuredClone(mocks.rows.get("dispatch-id"));
            const user = userEvent.setup();
            const view = renderBuilder(originalOwner);
            if (operation === "update") await selectSaved(user);
            const gate = deferred();
            mocks.beforeRead.mockImplementationOnce(() => gate.promise);
            await clickWrite(user, operation);
            await waitFor(() => expect(mocks.beforeRead).toHaveBeenCalled());
            if (change === "auth-source") mocks.runtimeKey = "source-key";
            else mocks.config = switchedOwners[change === "backend" ? 0 : 1];
            gate.resolve();
            await screen.findByText(
              operation === "create"
                ? "Could not save the recipe. Try again."
                : "Could not update the recipe. Try again.",
            );
            expect(promptMutations()).toHaveLength(0);
            expect(mocks.rows.get("dispatch-id")).toEqual(originalRow);
            const actualOwner = (await resolveRecipePersistenceOwnerView())!
              .ownerId;
            expect(actualOwner).not.toBe(originalOwner);
            view.unmount();
            const retry = renderBuilder(actualOwner);
            if (operation === "update") await selectSaved(user);
            await clickWrite(user, operation);
            await screen.findByText(/server outcome.*not.*verified/i);
            expect(promptMutations()).toHaveLength(1);
            expect(
              await readRecipePersistenceUncertainty(
                "dispatch-id",
                actualOwner,
              ),
            ).toBe("scoped");
            expect(
              await readRecipePersistenceUncertainty(
                "dispatch-id",
                originalOwner,
              ),
            ).toBe("clear");
            expect(mocks.rows.has("dispatch-id")).toBe(true);
            expect(mocks.rows.get("dispatch-id")?.syncStatus).not.toBe("error");
            retry.unmount();
            const reopened = renderBuilder(actualOwner);
            await selectSaved(user);
            await waitFor(() =>
              expect(
                screen.getByRole("button", { name: "Save as new recipe" }),
              ).toBeDisabled(),
            );
            expect(
              screen.getByRole("button", { name: "Update recipe" }),
            ).toBeDisabled();
            await user.type(
              screen.getByRole("textbox", {
                name: "Current value for Task (not saved)",
              }),
              "local task",
            );
            expect(
              screen.getByRole("button", { name: "Apply to system prompt" }),
            ).toBeEnabled();
            expect(
              screen.queryByRole("button", {
                name: "Forget unresolved operation",
              }),
            ).toBeNull();
            await clickWrite(user, operation);
            expect(promptMutations()).toHaveLength(1);
            reopened.unmount();
            const unrelatedOwner = renderBuilder(originalOwner);
            await selectSaved(user);
            await waitFor(() =>
              expect(
                screen.getByRole("button", { name: "Save as new recipe" }),
              ).toBeEnabled(),
            );
            unrelatedOwner.unmount();
            await clearRecipePersistenceScoped("dispatch-id", actualOwner);
          },
        );
      }
    }
  }

  it.each(["create", "update"])(
    "allows same-principal token rotation for %s and clears only after reconciliation",
    async (operation) => {
      seed(operation);
      const user = userEvent.setup();
      renderBuilder(ownerScope);
      if (operation === "update") await selectSaved(user);
      mocks.config = config("https://a.test", "alice", 2);
      let observedMarker: unknown;
      mocks.fetch.mockImplementation(async (url, init) => {
        if (new URL(String(url)).pathname === "/api/v1/auth/me")
          return principalResponse(init);
        observedMarker = await readRecipePersistenceUncertainty(
          "dispatch-id",
          ownerScope,
        );
        return jsonResponse({ success: true, data: serverRecord() });
      });
      await clickWrite(user, operation);
      await waitFor(() =>
        expect(mocks.rows.get("dispatch-id").serverId).toBe(101),
      );
      await waitFor(() => expect(observedMarker).toBe("scoped"));
      await waitFor(async () =>
        expect(
          await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
        ).toBe("clear"),
      );
      expect(promptMutations()).toHaveLength(1);
    },
  );

  it.each(["create", "update"])(
    "quarantines an unknown background %s delivery and Forget sends no remote request",
    async (operation) => {
      await startBackground();
      seed(operation);
      const deliver = mocks.sendMessage.getMockImplementation()!;
      mocks.sendMessage.mockImplementation(async (message) => {
        const value = await deliver(message);
        if (message.type === "tldw:request")
          throw new Error("message channel closed");
        return value;
      });
      const user = userEvent.setup();
      const view = renderBuilder(ownerScope);
      if (operation === "update") await selectSaved(user);
      await clickWrite(user, operation);
      await screen.findByRole("button", {
        name: "Forget unresolved operation",
      });
      expect(
        await readRecipePersistenceUncertainty(
          "dispatch-id",
          switchedScopes[0],
        ),
      ).toBe("unknown_owner");
      expect(promptMutations()).toHaveLength(1);
      view.unmount();
      renderBuilder(switchedScopes[0]);
      await selectSaved(user);
      await user.click(
        await screen.findByRole("button", {
          name: "Forget unresolved operation",
        }),
      );
      expect(screen.getByText(/may already have saved/)).toBeTruthy();
      await user.click(screen.getByRole("button", { name: "Cancel" }));
      expect(
        await readRecipePersistenceUncertainty(
          "dispatch-id",
          switchedScopes[0],
        ),
      ).toBe("unknown_owner");
      await user.click(
        screen.getByRole("button", { name: "Forget unresolved operation" }),
      );
      await user.click(screen.getByRole("button", { name: "Confirm forget" }));
      await waitFor(async () =>
        expect(
          await readRecipePersistenceUncertainty(
            "dispatch-id",
            switchedScopes[0],
          ),
        ).toBe("clear"),
      );
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("scoped");
      expect(promptMutations()).toHaveLength(1);
      expect(mocks.rows.has("dispatch-id")).toBe(true);
    },
  );

  it.each(["create", "update", "pull"])(
    "keeps v1 %s compatible without requiring owner identity",
    async (operation) => {
      const definition = {
        schema_version: 1,
        format: "structured",
        variables: [],
        blocks: [],
        assembly_config: {
          legacy_system_roles: ["system", "developer"],
          legacy_user_roles: ["user"],
          block_separator: "\\n\\n",
        },
      };
      mocks.config = { ...owner, accessToken: "opaque-token" };
      mocks.rows.set("dispatch-id", {
        id: "dispatch-id",
        title: "V1",
        syncStatus: "local",
        promptFormat: "structured",
        promptSchemaVersion: 1,
        structuredPromptDefinition: definition,
        ...(operation === "update" ? { serverId: 101 } : {}),
      });
      await markRecipePersistenceScoped("dispatch-id", ownerScope);
      mocks.fetch.mockImplementation(async (url, init) =>
        new URL(String(url)).pathname === "/api/v1/auth/me"
          ? principalResponse(init)
          : jsonResponse({
              recipePersistence: {
                state: "not_dispatched",
                actualOwnerId: null,
              },
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
      expect(result).toMatchObject({
        success: true,
        recipeOwnership: {
          dispatch: { state: "dispatched", actualOwnerId: null },
        },
      });
      expect(
        await readRecipePersistenceUncertainty("dispatch-id", ownerScope),
      ).toBe("scoped");
      expect(promptMutations()).toHaveLength(operation === "pull" ? 0 : 1);
    },
  );
});
