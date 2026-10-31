import React from "react";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { GeneratedArtifact } from "@/types/workspace";
import StoredAudioArtifactViewer from "../StoredAudioArtifactViewer";

const io = vi.hoisted(() => ({
  read: vi.fn(),
  create: vi.fn(),
  revoke: vi.fn(),
  state: {
    workspaceId: "owned",
    generatedArtifacts: [] as GeneratedArtifact[],
  },
  storeListeners: new Set<() => void>(),
  accountListeners: new Set<(invalidated: boolean) => void>(),
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
vi.mock("@/services/workspace-artifact-audio", () => ({
  readStoredWorkspaceAudio: io.read,
}));
const StandardURL = URL;
let artifact: GeneratedArtifact;
const blob = () =>
  new Blob([Uint8Array.from([1, 2, 3])], { type: "audio/wav" });
beforeEach(() => {
  io.read.mockReset();
  io.create.mockReset();
  io.revoke.mockReset();
  artifact = {
    id: "owned-audio",
    type: "audio_overview",
    title: "Rowan audio",
    status: "completed",
    content: "Selene tours start Friday at 18:00.",
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
  io.storeListeners.clear();
  io.accountListeners.clear();
  io.read.mockResolvedValue(blob());
  io.create.mockReturnValue("blob:owned-stored-audio");
  class TestURL extends StandardURL {
    static createObjectURL = io.create;
    static revokeObjectURL = io.revoke;
  }
  vi.stubGlobal("URL", TestURL);
});
afterEach(() => {
  vi.unstubAllGlobals();
});

describe("stored Studio audio viewer", () => {
  it.each(["account", "workspace", "replacement", "file-reference"] as const)(
    "revokes already loaded audio immediately on %s retirement",
    async (reason) => {
      render(
        <StoredAudioArtifactViewer
          artifact={artifact}
          workspaceId="owned"
          isCurrent={() => true}
        />,
      );
      await screen.findByLabelText("Saved Audio Summary");
      act(() => {
        if (reason === "account")
          io.accountListeners.forEach((listener) => listener(true));
        else {
          if (reason === "workspace") io.state.workspaceId = "other";
          if (reason === "replacement")
            io.state.generatedArtifacts = [{ ...artifact }];
          if (reason === "file-reference") artifact.exportRefs![0].fileId = 48;
          io.storeListeners.forEach((listener) => listener());
        }
      });
      expect(screen.queryByLabelText("Saved Audio Summary")).toBeNull();
      expect(screen.queryByText(artifact.content!)).toBeNull();
      expect(io.revoke).toHaveBeenCalledExactlyOnceWith(
        "blob:owned-stored-audio",
      );
      expect(io.read).toHaveBeenCalledTimes(1);
    },
  );

  it("loads owned audio and revokes its URL on close", async () => {
    const view = render(
      <StoredAudioArtifactViewer
        artifact={artifact}
        workspaceId="owned"
        isCurrent={() => true}
      />,
    );
    expect(await screen.findByLabelText("Saved Audio Summary")).toHaveAttribute(
      "src",
      "blob:owned-stored-audio",
    );
    expect(io.read.mock.calls[0][1]).toMatchObject({
      workspaceId: "owned",
      signal: expect.any(AbortSignal),
    });
    const signal = io.read.mock.calls[0][1].signal;
    view.unmount();
    expect(signal.aborted).toBe(true);
    expect(io.revoke).toHaveBeenCalledExactlyOnceWith(
      "blob:owned-stored-audio",
    );
  });
  it.each(["close", "retired"] as const)(
    "does not publish a late read after %s",
    async (reason) => {
      let finish!: (value: Blob) => void;
      let current = true;
      io.read.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      );
      const view = render(
        <StoredAudioArtifactViewer
          artifact={artifact}
          workspaceId="owned"
          isCurrent={() => current}
        />,
      );
      await waitFor(() => expect(io.read).toHaveBeenCalledTimes(1));
      if (reason === "close") view.unmount();
      else current = false;
      await act(async () => {
        finish(blob());
        await Promise.resolve();
      });
      expect(io.create).not.toHaveBeenCalled();
      if (reason === "retired") view.unmount();
    },
  );
  it("retains the script and permits one explicit loading Retry", async () => {
    io.read.mockRejectedValueOnce(new Error("Unavailable audio"));
    const view = render(
      <StoredAudioArtifactViewer
        artifact={artifact}
        workspaceId="owned"
        isCurrent={() => true}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("Saved audio");
    expect(screen.getByText(artifact.content!)).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Retry loading audio" }),
    );
    expect(
      await screen.findByLabelText("Saved Audio Summary"),
    ).toBeInTheDocument();
    expect(io.read).toHaveBeenCalledTimes(2);
    view.unmount();
    expect(io.revoke).toHaveBeenCalledTimes(1);
  });
  it("shows a recoverable error if the browser cannot create an audio URL", async () => {
    io.create.mockImplementationOnce(() => {
      throw new Error("Controlled URL failure");
    });
    render(
      <StoredAudioArtifactViewer
        artifact={artifact}
        workspaceId="owned"
        isCurrent={() => true}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("Saved audio");
    expect(screen.queryByLabelText("Saved Audio Summary")).toBeNull();
    expect(io.revoke).not.toHaveBeenCalled();
  });
  it("keeps a legacy script and recovery after an unrelated store notification", async () => {
    artifact.exportRefs = undefined;
    io.read.mockRejectedValueOnce(
      new Error("This output has no saved audio file"),
    );
    render(
      <StoredAudioArtifactViewer
        artifact={artifact}
        workspaceId="owned"
        isCurrent={() => true}
      />,
    );
    await screen.findByRole("alert");
    act(() => {
      io.storeListeners.forEach((listener) => listener());
    });
    expect(screen.getByText(artifact.content!)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Retry loading audio" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("alert")).not.toHaveTextContent("account changed");
    expect(io.create).not.toHaveBeenCalled();
  });
});
