import type { MessageInstance } from "antd/es/message/interface";
import type { TFunction } from "i18next";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GeneratedArtifact } from "@/types/workspace";
import { useArtifactExport } from "../useArtifactExport";

const io = vi.hoisted(() => ({
  read: vi.fn(),
  output: vi.fn(),
  create: vi.fn(),
  revoke: vi.fn(),
  state: {
    workspaceId: "owned",
    generatedArtifacts: [] as GeneratedArtifact[],
  },
  accountListeners: new Set<(value: boolean) => void>(),
  storeListeners: new Set<() => void>(),
  files: [] as { filename: string; href: string }[],
}));
vi.mock("@/services/workspace-artifact-audio", () => ({
  readStoredWorkspaceAudio: io.read,
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { downloadOutput: io.output },
}));
vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: {
    getState: () => io.state,
    subscribe: (listener: () => void) => {
      io.storeListeners.add(listener);
      return () => io.storeListeners.delete(listener);
    },
  },
  hasResearchWorkspaceMigrationTombstone: () => true,
}));
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (listener: (value: boolean) => void) => {
    io.accountListeners.add(listener);
    return () => io.accountListeners.delete(listener);
  },
}));
const StandardURL = URL;
const messages = { error: vi.fn(), success: vi.fn() };
let artifact: GeneratedArtifact;
const bytes = () =>
  new Blob([Uint8Array.from([1, 2, 3])], { type: "audio/wav" });
beforeEach(() => {
  io.read.mockReset();
  io.output.mockReset();
  io.create.mockReset();
  io.revoke.mockReset();
  messages.error.mockReset();
  messages.success.mockReset();
  io.accountListeners.clear();
  io.storeListeners.clear();
  io.files = [];
  artifact = {
    id: "saved-audio",
    workspaceArtifactId: "saved-audio",
    serverId: "saved-audio",
    type: "audio_overview",
    title: "Rowan audio",
    status: "completed",
    content: "Retained script",
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
  };
  io.state = { workspaceId: "owned", generatedArtifacts: [artifact] };
  io.read.mockResolvedValue(bytes());
  io.output.mockResolvedValue(
    new Blob(["wrong resource"], { type: "text/plain" }),
  );
  io.create.mockReturnValue("blob:audio-download");
  class TestURL extends StandardURL {
    static createObjectURL = io.create;
    static revokeObjectURL = io.revoke;
  }
  vi.stubGlobal("URL", TestURL);
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
    this: HTMLAnchorElement,
  ) {
    io.files.push({ filename: this.download, href: this.href });
  });
});
afterEach(() => {
  vi.unstubAllGlobals();
});
const setup = () =>
  renderHook(() =>
    useArtifactExport({
      workspaceId: io.state.workspaceId,
      generatedArtifacts: io.state.generatedArtifacts,
      messageApi: messages as unknown as MessageInstance,
      isMobile: false,
      removeArtifact: vi.fn(),
      restoreArtifact: vi.fn(),
      captureToCurrentNote: vi.fn(),
      t: ((_key: string, fallback: unknown) =>
        fallback) as unknown as TFunction,
    }),
  );

describe("stored Studio audio download", () => {
  it("downloads the stored file without treating a canonical ID as a media output", async () => {
    const hook = setup();
    await hook.result.current.handleDownloadArtifact(artifact);
    expect(io.read).toHaveBeenCalledTimes(1);
    expect(io.read.mock.calls[0][0]).toBe(artifact);
    expect(io.read.mock.calls[0][1]).toMatchObject({ workspaceId: "owned" });
    expect(io.output).not.toHaveBeenCalled();
    expect(io.files).toEqual([
      { filename: "Rowan audio.wav", href: "blob:audio-download" },
    ]);
    expect(io.revoke).toHaveBeenCalledExactlyOnceWith("blob:audio-download");
  });
  it("does not fall through to script or another resource for unavailable stored audio", async () => {
    io.read.mockRejectedValueOnce(new Error("Saved audio missing"));
    const hook = setup();
    await hook.result.current.handleDownloadArtifact(artifact);
    expect(io.read).toHaveBeenCalledTimes(1);
    expect(io.output).not.toHaveBeenCalled();
    expect(io.files).toHaveLength(0);
    expect(messages.error).toHaveBeenCalledTimes(1);
  });
  it.each([
    "account",
    "workspace",
    "workspace-ABA",
    "replacement",
    "unmount",
  ] as const)("does not download after %s retirement", async (reason) => {
    let finish!: (value: Blob) => void;
    io.read.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const hook = setup();
    const pending = hook.result.current.handleDownloadArtifact(artifact);
    await waitFor(() => expect(io.read).toHaveBeenCalledTimes(1));
    const signal = io.read.mock.calls[0][1].signal;
    if (reason === "account")
      io.accountListeners.forEach((listener) => listener(true));
    if (reason === "workspace") {
      io.state.workspaceId = "other";
      hook.rerender();
    }
    if (reason === "workspace-ABA") {
      io.state.workspaceId = "other";
      io.storeListeners.forEach((listener) => listener());
      io.state.workspaceId = "owned";
      io.storeListeners.forEach((listener) => listener());
    }
    if (reason === "replacement") {
      io.state.generatedArtifacts = [{ ...artifact }];
      io.storeListeners.forEach((listener) => listener());
    }
    if (reason === "unmount") hook.unmount();
    expect(signal.aborted).toBe(true);
    await act(async () => {
      finish(bytes());
      await pending;
    });
    expect(io.files).toHaveLength(0);
    expect(io.create).not.toHaveBeenCalled();
    expect(io.output).not.toHaveBeenCalled();
  });
  it("exports browser speech as a named text script", async () => {
    artifact.audioFormat = "browser";
    artifact.exportRefs = undefined;
    const hook = setup();
    await hook.result.current.handleDownloadArtifact(artifact);
    expect(io.read).not.toHaveBeenCalled();
    expect(io.output).not.toHaveBeenCalled();
    expect(io.files[0].filename).toBe("Rowan audio.txt");
    expect(io.create.mock.calls[0][0].type).toBe("text/plain");
  });
  it("retains the existing local blob download without inventing file authority", async () => {
    artifact.audioUrl = "blob:legacy-audio";
    artifact.exportRefs = undefined;
    const hook = setup();
    await hook.result.current.handleDownloadArtifact(artifact);
    expect(io.read).not.toHaveBeenCalled();
    expect(io.output).not.toHaveBeenCalled();
    expect(io.files).toEqual([
      { filename: "Rowan audio.wav", href: "blob:legacy-audio" },
    ]);
  });
  it("releases the private URL if the browser rejects the download click", async () => {
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementationOnce(
      () => {
        throw new Error("Controlled download failure");
      },
    );
    const hook = setup();
    await hook.result.current.handleDownloadArtifact(artifact);
    expect(io.create).toHaveBeenCalledTimes(1);
    expect(io.revoke).toHaveBeenCalledExactlyOnceWith("blob:audio-download");
    expect(io.output).not.toHaveBeenCalled();
    expect(messages.error).toHaveBeenCalledTimes(1);
  });
  it("refuses an export after its absolute end before timers run", async () => {
    let finish!: (value: Blob) => void;
    io.read.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        }),
    );
    const now = vi.spyOn(performance, "now").mockReturnValue(0);
    const hook = setup();
    const pending = hook.result.current.handleDownloadArtifact(artifact);
    try {
      await waitFor(() => expect(io.read).toHaveBeenCalledTimes(1));
      now.mockReturnValue(15001);
      await act(async () => {
        finish(bytes());
        await pending;
      });
      expect(io.create).not.toHaveBeenCalled();
      expect(messages.error).toHaveBeenCalledTimes(1);
      expect(io.files).toHaveLength(0);
      expect(io.accountListeners.size + io.storeListeners.size).toBe(0);
    } finally {
      now.mockRestore();
    }
  });
  it.each([false, true])(
    "reports a timed-out download only while still owned: retired=%s",
    async (retired) => {
      let fail!: (error: unknown) => void;
      io.read.mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            fail = reject;
          }),
      );
      const now = vi.spyOn(performance, "now").mockReturnValue(0);
      const hook = setup();
      const pending = hook.result.current.handleDownloadArtifact(artifact);
      try {
        await waitFor(() => expect(io.read).toHaveBeenCalledTimes(1));
        if (retired) io.accountListeners.forEach((listener) => listener(true));
        now.mockReturnValue(15001);
        await act(async () => {
          fail(new DOMException("Controlled deadline", "TimeoutError"));
          await pending;
        });
        expect(messages.error).toHaveBeenCalledTimes(retired ? 0 : 1);
        expect(io.create).not.toHaveBeenCalled();
        expect(io.accountListeners.size + io.storeListeners.size).toBe(0);
      } finally {
        now.mockRestore();
      }
    },
  );
});
