import { beforeEach, afterEach, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({
  storage: new Map<string, unknown>(),
  set: vi.fn(),
}));
vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn(),
  bgStream: vi.fn(),
  bgUpload: vi.fn(),
}));
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => mocks.storage.get(key) ?? null,
    set: async (key: string, value: unknown) => {
      mocks.set(key, value);
      mocks.storage.set(key, value);
    },
    remove: async (key: string) => {
      mocks.storage.delete(key);
    },
  }),
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) =>
      typeof value === "string" ? JSON.parse(value) : value,
  },
}));
import { TldwApiClient } from "../TldwApiClient";
import { createServicePromptScopeChangedError } from "../service-prompt-scope-error";
const target = {
  serverUrl: "https://auth.example.test/base",
  authMode: "multi-user" as const,
};
beforeEach(() => {
  mocks.storage.clear();
  mocks.set.mockClear();
  mocks.storage.set("tldwConfig", target);
});
afterEach(() => vi.restoreAllMocks());
it.each(["target", "ABA"])(
  "rejects token publication after an awaited client read changes %s",
  async (change) => {
    const client = new TldwApiClient();
    vi.spyOn(client, "initialize").mockResolvedValue(undefined);
    let invalidated = false;
    vi.spyOn(client, "getConfig").mockImplementation(async () => {
      await Promise.resolve();
      invalidated = true;
      return change === "target"
        ? { ...target, serverUrl: "https://other.example.test" }
        : target;
    });
    await expect(
      client.updateConfig({ accessToken: "new-private-token" }, (current) => {
        if (invalidated || current.serverUrl !== target.serverUrl)
          throw createServicePromptScopeChangedError();
      }),
    ).rejects.toMatchObject({ status: 412 });
    expect(mocks.set).not.toHaveBeenCalled();
    expect(mocks.storage.get("tldwConfig")).toEqual(target);
  },
);
it("allows a current same-target guarded credential commit", async () => {
  const client = new TldwApiClient();
  vi.spyOn(client, "initialize").mockResolvedValue(undefined);
  vi.spyOn(client, "getConfig").mockResolvedValue(target);
  await client.updateConfig({ accessToken: "new-private-token" }, (current) => {
    expect(current.serverUrl).toBe(target.serverUrl);
  });
  expect(mocks.storage.get("tldwConfig")).toMatchObject({
    ...target,
    accessToken: "new-private-token",
  });
});

it("does not publish Alice after Bob replaces config during storage resolution", async () => {
  const client = new TldwApiClient();
  const internal = client as unknown as {
    storage: { set: (key: string, value: unknown) => Promise<void> };
    config: unknown;
  };
  // Model initialize's eventual reload, so a later repair cannot hide an earlier
  // stale public config event. Neither this probe nor the fixture makes network I/O.
  vi.spyOn(client, "initialize").mockImplementation(async () => {
    internal.config = mocks.storage.get("tldwConfig");
  });
  vi.spyOn(client, "getConfig").mockImplementation(
    async () =>
      mocks.storage.get("tldwConfig") as import("../TldwApiClient").TldwConfig,
  );
  const originalSet = internal.storage.set.bind(internal.storage);
  let release!: () => void;
  let entered!: () => void;
  const began = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let first = true;
  vi.spyOn(internal.storage, "set").mockImplementation(async (key, value) => {
    await originalSet(key, value);
    if (first) {
      first = false;
      entered();
      await held;
    }
  });
  let invalidated = false;
  const pending = client.updateConfig(
    { accessToken: "alice-new-token" },
    () => {
      if (invalidated) throw createServicePromptScopeChangedError();
    },
  );
  await began;
  const bob = {
    ...target,
    serverUrl: "https://auth.example.test/other",
    accessToken: "bob-token",
  };
  await client.updateConfig(bob);
  invalidated = true;
  const published: unknown[] = [];
  const listener = () => {
    published.push(internal.config);
  };
  window.addEventListener("tldw:config-updated", listener);
  try {
    release();
    await pending.catch(() => undefined);
    expect(mocks.storage.get("tldwConfig")).toEqual(bob);
    expect(internal.config).toEqual(bob);
    expect(published).not.toContainEqual({
      ...target,
      accessToken: "alice-new-token",
    });
  } finally {
    window.removeEventListener("tldw:config-updated", listener);
  }
});

it("review: rejects stale cached sign-in before transmitting credentials", async () => {
  const { tldwClient } = await import("../TldwApiClient");
  const { TldwAuthService } = await import("../TldwAuth");
  const { bgRequest } = await import("@/services/background-proxy");
  (tldwClient as unknown as { config: unknown }).config = target;
  mocks.storage.set("tldwConfig", {
    ...target,
    serverUrl: "https://auth.example.test/other",
  });
  vi.mocked(bgRequest).mockClear();
  vi.mocked(bgRequest).mockResolvedValue({
    access_token: "late-token",
    token_type: "bearer",
  });
  await new TldwAuthService()
    .login({ username: "alice", password: "synthetic" }, { target })
    .catch(() => undefined);
  expect(bgRequest).not.toHaveBeenCalled();
});

it("accepts its own successful storage account event without cancelling login", async () => {
  const { tldwClient } = await import("../TldwApiClient");
  const { TldwAuthService } = await import("../TldwAuth");
  const { bgRequest } = await import("@/services/background-proxy");
  (tldwClient as unknown as { config: unknown }).config = null;
  vi.mocked(bgRequest).mockResolvedValue({
    access_token: "own-new-token",
    token_type: "bearer",
  });
  mocks.set.mockImplementationOnce((key, value) =>
    window.dispatchEvent(
      new StorageEvent("storage", {
        key,
        oldValue: JSON.stringify(target),
        newValue: JSON.stringify(value),
      }),
    ),
  );
  await expect(
    new TldwAuthService().login(
      { username: "alice", password: "synthetic" },
      { target },
    ),
  ).resolves.toMatchObject({ access_token: "own-new-token" });
  expect(await tldwClient.getConfig()).toMatchObject({
    ...target,
    accessToken: "own-new-token",
  });
});

it("does not publish login after its view unmounts during the storage write", async () => {
  const { tldwClient } = await import("../TldwApiClient");
  const { TldwAuthService } = await import("../TldwAuth");
  const { bgRequest } = await import("@/services/background-proxy");
  (tldwClient as unknown as { config: unknown }).config = null;
  vi.mocked(bgRequest).mockResolvedValue({
    access_token: "own-new-token",
    token_type: "bearer",
  });
  const controller = new AbortController();
  mocks.set.mockImplementationOnce(() => controller.abort());
  const published = vi.fn();
  window.addEventListener("tldw:config-updated", published);
  try {
    await expect(
      new TldwAuthService().login(
        { username: "alice", password: "synthetic" },
        { target, signal: controller.signal },
      ),
    ).rejects.toMatchObject({ status: 412 });
    expect(published).not.toHaveBeenCalled();
  } finally {
    window.removeEventListener("tldw:config-updated", published);
  }
});

it("adopts a same-principal token refresh that finishes before the login write promise resolves", async () => {
  const client = new TldwApiClient();
  const jwt = (nonce: number) =>
    `header.${btoa(JSON.stringify({ sub: "42", nonce }))}.signature`;
  const written = { ...target, accessToken: jwt(1) };
  const refreshed = { ...target, accessToken: jwt(2) };
  const internal = client as unknown as {
    storage: { set: (key: string, value: unknown) => Promise<void> };
    config: unknown;
  };
  vi.spyOn(client, "initialize").mockImplementation(async () => {
    internal.config = mocks.storage.get("tldwConfig");
  });
  vi.spyOn(client, "getConfig").mockImplementation(
    async () =>
      mocks.storage.get("tldwConfig") as import("../TldwApiClient").TldwConfig,
  );
  vi.spyOn(internal.storage, "set").mockImplementation(async (key, _value) => {
    mocks.storage.set(key, refreshed);
  });
  const published: unknown[] = [];
  const listener = () => {
    published.push(internal.config);
  };
  window.addEventListener("tldw:config-updated", listener);
  try {
    await client.updateConfig(written, () => {});
    expect(internal.config).toEqual(refreshed);
    expect(published).not.toContainEqual(written);
  } finally {
    window.removeEventListener("tldw:config-updated", listener);
  }
});

it("does not publish an obsolete storage-read snapshot after Bob commits", async () => {
  const client = new TldwApiClient();
  const internal = client as unknown as {
    storage: {
      get: (key: string) => Promise<unknown>;
      set: (key: string, value: unknown) => Promise<void>;
    };
    config: unknown;
  };
  vi.spyOn(client, "initialize").mockImplementation(async () => {
    internal.config = mocks.storage.get("tldwConfig");
  });
  vi.spyOn(client, "getConfig").mockImplementation(
    async () =>
      mocks.storage.get("tldwConfig") as import("../TldwApiClient").TldwConfig,
  );
  let release!: () => void;
  let entered!: () => void;
  const began = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const originalGet = internal.storage.get.bind(internal.storage);
  let first = true;
  vi.spyOn(internal.storage, "get").mockImplementation(async (key) => {
    const value = await originalGet(key);
    if (first) {
      first = false;
      entered();
      await held;
    }
    return value;
  });
  let invalidated = false;
  const pending = client.updateConfig(
    { accessToken: "alice-new-token" },
    () => {
      if (invalidated) throw createServicePromptScopeChangedError();
    },
  );
  await began;
  const bob = {
    ...target,
    serverUrl: "https://auth.example.test/other",
    accessToken: "bob-token",
  };
  await client.updateConfig(bob);
  invalidated = true;
  const published: unknown[] = [];
  const listener = () => {
    published.push(internal.config);
  };
  window.addEventListener("tldw:config-updated", listener);
  try {
    release();
    await pending.catch(() => undefined);
    expect(mocks.storage.get("tldwConfig")).toEqual(bob);
    expect(internal.config).toEqual(bob);
    expect(published).not.toContainEqual({
      ...target,
      accessToken: "alice-new-token",
    });
  } finally {
    window.removeEventListener("tldw:config-updated", listener);
  }
});

it.each(["own", "refresh", "foreign", "ABA"] as const)(
  "handles a %s storage event during the guarded verification read",
  async (change) => {
    const client = new TldwApiClient();
    const jwt = (sub: string, nonce: number) =>
      `header.${btoa(JSON.stringify({ sub, nonce }))}.signature`;
    const alice = { ...target, accessToken: jwt("2", 1) };
    const refreshed = { ...target, accessToken: jwt("2", 2) };
    const bob = { ...target, accessToken: jwt("3", 1) };
    const internal = client as unknown as {
      storage: { get: (key: string) => Promise<unknown> };
      config: unknown;
    };
    vi.spyOn(client, "initialize").mockImplementation(async () => {
      internal.config = mocks.storage.get("tldwConfig");
    });
    vi.spyOn(client, "getConfig").mockResolvedValue(target);
    let release!: () => void;
    let entered!: () => void;
    const began = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    const originalGet = internal.storage.get.bind(internal.storage);
    vi.spyOn(internal.storage, "get").mockImplementation(async (key) => {
      const snapshot = await originalGet(key);
      entered();
      await held;
      return snapshot;
    });
    const pending = client.updateConfig(alice, () => {});
    // Attach rejection handling before releasing the read.
    const settled = pending.then(
      () => "accepted",
      (error) => error.status,
    );
    await began;
    const observed =
      change === "refresh" ? refreshed : change === "own" ? alice : bob;
    const emit = (previous: unknown, current: unknown) => {
      mocks.storage.set("tldwConfig", current);
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "tldwConfig",
          oldValue: JSON.stringify(previous),
          newValue: JSON.stringify(current),
        }),
      );
    };
    emit(change === "own" ? target : alice, observed);
    if (change === "ABA") emit(bob, alice);
    const published = vi.fn();
    window.addEventListener("tldw:config-updated", published);
    try {
      release();
      if (change === "foreign" || change === "ABA") {
        expect(await settled).toBe(412);
        expect(published).not.toHaveBeenCalled();
      } else {
        expect(await settled).toBe("accepted");
        expect(internal.config).toEqual(observed);
      }
      expect(mocks.storage.get("tldwConfig")).toEqual(
        change === "ABA" ? alice : observed,
      );
    } finally {
      window.removeEventListener("tldw:config-updated", published);
    }
  },
);

it("review: rejects an account ABA while the completed storage write promise is delayed", async () => {
  const { tldwClient } = await import("../TldwApiClient");
  const { TldwAuthService } = await import("../TldwAuth");
  const { bgRequest } = await import("@/services/background-proxy");
  const jwt = (sub: string) =>
    `header.${btoa(JSON.stringify({ sub }))}.signature`;
  (tldwClient as unknown as { config: unknown }).config = null;
  vi.mocked(bgRequest).mockResolvedValue({
    access_token: jwt("alice"),
    token_type: "bearer",
  });
  const internal = tldwClient as unknown as {
    storage: { set: (key: string, value: unknown) => Promise<void> };
  };
  const originalSet = internal.storage.set.bind(internal.storage);
  let release!: () => void;
  let entered!: () => void;
  const began = new Promise<void>((resolve) => {
    entered = resolve;
  });
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  vi.spyOn(internal.storage, "set").mockImplementationOnce(
    async (key, value) => {
      await originalSet(key, value);
      entered();
      await held;
    },
  );
  const pending = new TldwAuthService().login(
    { username: "alice", password: "synthetic" },
    { target },
  );
  const settled = pending.then(
    () => "accepted",
    (error) => error.status,
  );
  await began;
  const alice = mocks.storage.get("tldwConfig");
  const bob = { ...target, accessToken: jwt("bob") };
  for (const [previous, current] of [
    [alice, bob],
    [bob, alice],
  ]) {
    mocks.storage.set("tldwConfig", current);
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "tldwConfig",
        oldValue: JSON.stringify(previous),
        newValue: JSON.stringify(current),
      }),
    );
  }
  release();
  expect(await settled).toBe(412);
});

it("review: keeps a newer verified storage value over a delayed own event", async () => {
  const client = new TldwApiClient();
  const jwt = (nonce: number) =>
    `header.${btoa(JSON.stringify({ sub: "42", nonce }))}.signature`;
  const alice = { ...target, accessToken: jwt(1) };
  const refreshed = { ...target, accessToken: jwt(2) };
  const internal = client as unknown as {
    storage: { get: (key: string) => Promise<unknown> };
    config: unknown;
  };
  vi.spyOn(client, "initialize").mockResolvedValue(undefined);
  vi.spyOn(client, "getConfig").mockResolvedValue(target);
  vi.spyOn(internal.storage, "get").mockImplementationOnce(async () => {
    // The verification read captures refresh2; an older own-write event arrives
    // while the storage API is resolving that read.
    mocks.storage.set("tldwConfig", refreshed);
    const snapshot = refreshed;
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "tldwConfig",
        oldValue: JSON.stringify(target),
        newValue: JSON.stringify(alice),
      }),
    );
    return snapshot;
  });
  await client.updateConfig(alice, () => {});
  expect(internal.config).toEqual(refreshed);
});

it("review: rejects a verified foreign account despite a delayed own storage event", async () => {
  const client = new TldwApiClient();
  const jwt = (sub: string) =>
    `header.${btoa(JSON.stringify({ sub }))}.signature`;
  const alice = { ...target, accessToken: jwt("alice") };
  const bob = { ...target, accessToken: jwt("bob") };
  const internal = client as unknown as {
    storage: { get: (key: string) => Promise<unknown> };
    config: unknown;
  };
  const initSpy = vi.spyOn(client, "initialize").mockResolvedValue(undefined);
  const configSpy = vi.spyOn(client, "getConfig").mockResolvedValue(target);
  vi.spyOn(internal.storage, "get").mockImplementationOnce(async () => {
    // Bob has committed before this read captures its result. The notification
    // still queued from our older Alice write is delivered during that read.
    mocks.storage.set("tldwConfig", bob);
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "tldwConfig",
        oldValue: JSON.stringify(target),
        newValue: JSON.stringify(alice),
      }),
    );
    return bob;
  });
  const outcome = await client
    .updateConfig(alice, () => {})
    .then(
      () => "accepted",
      (error) => error.status,
    );
  initSpy.mockRestore();
  configSpy.mockRestore();
  const visible = await client.getConfig();
  expect({ outcome, visible, stored: mocks.storage.get("tldwConfig") }).toEqual(
    { outcome: 412, visible: bob, stored: bob },
  );
});
