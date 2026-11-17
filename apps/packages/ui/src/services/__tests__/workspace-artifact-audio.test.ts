import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GeneratedArtifact } from "@/types/workspace";
import { bindWorkspaceArtifactOrigin } from "@/store/workspace-artifact-codec";
import { isServicePromptRequestPath } from "@/services/tldw/service-prompt-scope-error";
import {
  getStoredWorkspaceAudioRef,
  readStoredWorkspaceAudio,
} from "../workspace-artifact-audio";

const io = vi.hoisted(() => ({
  request: vi.fn(),
  snapshot: vi.fn(),
  state: {
    workspaceId: "owned",
    generatedArtifacts: [] as GeneratedArtifact[],
  },
  storeListeners: new Set<() => void>(),
  accountListeners: new Set<(invalidated: boolean) => void>(),
}));
vi.mock("@/services/background-proxy", () => ({
  bgRequest: io.request,
  sanitizeResponseData: (value: unknown) => value,
}));
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: io.snapshot,
}));
vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: {
    getState: () => io.state,
    subscribe: (listener: () => void) => {
      io.storeListeners.add(listener);
      return () => io.storeListeners.delete(listener);
    },
  },
}));
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (listener: (invalidated: boolean) => void) => {
    io.accountListeners.add(listener);
    return () => io.accountListeners.delete(listener);
  },
}));
let artifact: GeneratedArtifact;
let scope: AbortController;
let release: ReturnType<typeof vi.fn>;
const response = () => ({
  ok: true,
  data: Uint8Array.from([1, 2, 3]).buffer,
  headers: { "Content-Type": "audio/wav" },
});
const read = (signal?: AbortSignal) =>
  readStoredWorkspaceAudio(artifact, {
    workspaceId: "owned",
    isCurrent: () => true,
    signal,
  });
beforeEach(() => {
  vi.useRealTimers();
  io.request.mockReset();
  io.snapshot.mockReset();
  io.storeListeners.clear();
  io.accountListeners.clear();
  artifact = bindWorkspaceArtifactOrigin(
    {
      id: "saved-audio",
      workspaceArtifactId: "saved-audio",
      workspaceArtifactVersion: 1,
      type: "audio_overview",
      title: "Rowan audio",
      status: "completed",
      content: "Selene.",
      createdAt: new Date(),
      audioFormat: "wav",
      exportRefs: [
        {
          sourceType: "storage_file",
          fileId: 47,
          format: "wav",
          status: "ready",
        },
      ],
    },
    "server:alice",
    "owned",
  );
  io.state = { workspaceId: "owned", generatedArtifacts: [artifact] };
  scope = new AbortController();
  release = vi.fn();
  io.snapshot.mockResolvedValue({
    scopeKey: "server:alice",
    requestScope: {
      config: { serverUrl: "https://owned.test", authMode: "multi-user" },
      userId: "alice",
    },
    scopeSignal: scope.signal,
    scopeInvalidatedSignal: scope.signal,
    release,
  });
  io.request.mockResolvedValue(response());
});
afterEach(() => {
  vi.useRealTimers();
});

describe("owned stored Workspace audio", () => {
  it("reads only the typed file with captured account authority and audio media type", async () => {
    const blob = await read();
    expect(blob.type).toBe("audio/wav");
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(
      Uint8Array.from([1, 2, 3]),
    );
    expect(io.request).toHaveBeenCalledTimes(1);
    expect(io.request.mock.calls[0][0]).toMatchObject({
      path: "/api/v1/storage/files/47/download",
      method: "GET",
      responseType: "arrayBuffer",
      returnResponse: true,
      servicePromptConfig: {
        serverUrl: "https://owned.test",
        expectedUserId: "alice",
      },
      headers: { "X-TLDW-Expected-User-ID": "alice" },
    });
    expect(
      isServicePromptRequestPath("/api/v1/storage/files/47/download", "GET"),
    ).toBe(true);
    expect(release).toHaveBeenCalledTimes(1);
    expect(io.storeListeners.size + io.accountListeners.size).toBe(0);
  });

  it.each([
    { fileId: 0 },
    { fileId: "47" },
    { fileId: Number.MAX_SAFE_INTEGER + 1 },
    { status: "pending" },
    { sourceType: "media_output" },
    { format: "mp3" },
  ])(
    "refuses ambiguous or invalid storage references before requesting bytes: %j",
    async (change) => {
      artifact.exportRefs = [{ ...artifact.exportRefs![0], ...change }];
      await expect(read()).rejects.toThrow();
      expect(io.request).not.toHaveBeenCalled();
    },
  );

  it("refuses cached authority and duplicate storage references", async () => {
    artifact = JSON.parse(JSON.stringify(artifact));
    io.state.generatedArtifacts = [artifact];
    await expect(read()).rejects.toThrow();
    expect(io.request).not.toHaveBeenCalled();
    artifact.exportRefs!.push({ ...artifact.exportRefs![0] });
    expect(() => getStoredWorkspaceAudioRef(artifact)).toThrow();
  });

  it.each([
    "account",
    "workspace-ABA",
    "replacement",
    "unmount",
    "timeout",
  ] as const)("retires a held read permanently on %s", async (kind) => {
    if (kind === "timeout") vi.useFakeTimers();
    let resolve!: (value: unknown) => void;
    io.request.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const external = new AbortController();
    const pending = read(external.signal);
    pending.catch(() => {});
    await vi.waitFor(() => expect(io.request).toHaveBeenCalledTimes(1));
    const requestSignal = io.request.mock.calls[0][0]
      .abortSignal as AbortSignal;
    if (kind === "account")
      for (const listener of [...io.accountListeners]) listener(true);
    if (kind === "workspace-ABA") {
      io.state.workspaceId = "other";
      io.storeListeners.forEach((listener) => listener());
      io.state.workspaceId = "owned";
      io.storeListeners.forEach((listener) => listener());
    }
    if (kind === "replacement") {
      io.state.generatedArtifacts = [
        {
          ...artifact,
          exportRefs: [
            {
              sourceType: "storage_file",
              fileId: 48,
              format: "wav",
              status: "ready",
            },
          ],
        },
      ];
      io.storeListeners.forEach((listener) => listener());
    }
    if (kind === "unmount") external.abort();
    if (kind === "timeout") await vi.advanceTimersByTimeAsync(15000);
    expect(requestSignal.aborted).toBe(true);
    resolve(response());
    await expect(pending).rejects.toBeDefined();
    expect(release).toHaveBeenCalledTimes(1);
    expect(io.storeListeners.size + io.accountListeners.size).toBe(0);
  });

  it.each([
    { ok: false, status: 403 },
    { ok: false, status: 404 },
    { data: new ArrayBuffer(0) },
    { headers: { "Content-Type": "text/plain" } },
  ])("refuses unavailable or non-audio assets: %j", async (change) => {
    io.request.mockResolvedValue({ ...response(), ...change });
    await expect(read()).rejects.toThrow(/audio/i);
    expect(io.request).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["/api/v1/storage/files/47/download", "POST"],
    ["/api/v1/storage/files/0/download", "GET"],
    ["/api/v1/storage/files/47%2f48/download", "GET"],
    ["/api/v1/storage/files/47", "GET"],
  ])("does not admit unrelated file actions %s %s", (path, method) => {
    expect(isServicePromptRequestPath(path, method)).toBe(false);
  });
  it.each(["scope", "body"] as const)(
    "refuses a late %s before an overdue timer callback",
    async (stage) => {
      const ownedScope = await io.snapshot();
      io.snapshot.mockClear();
      const now = vi.spyOn(performance, "now").mockReturnValue(0);
      let finish!: (value: unknown) => void;
      const held = new Promise((resolve) => {
        finish = resolve;
      });
      if (stage === "scope") io.snapshot.mockReturnValueOnce(held);
      else io.request.mockReturnValueOnce(held);
      const pending = read();
      pending.catch(() => {});
      try {
        await vi.waitFor(() =>
          expect(
            stage === "scope" ? io.snapshot : io.request,
          ).toHaveBeenCalledTimes(1),
        );
        now.mockReturnValue(15001);
        finish(stage === "scope" ? ownedScope : response());
        await expect(pending).rejects.toMatchObject({ name: "TimeoutError" });
        if (stage === "scope") expect(io.request).not.toHaveBeenCalled();
        expect(io.storeListeners.size + io.accountListeners.size).toBe(0);
        expect(release).toHaveBeenCalledTimes(1);
      } finally {
        now.mockRestore();
      }
    },
  );
});
