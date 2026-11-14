// @vitest-environment jsdom
import React from "react";
import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import type { ComparisonResult } from "@/hooks/useComparisonTranscribe";
import type { SttHistoryEntry } from "../HistoryPanel";
import { Storage } from "../../../../../../../tldw-frontend/extension/shims/plasmo-storage";
const scope = vi.hoisted(() => ({
  user: "alice" as string | null,
  server: "http://server-a",
  pending: null as Promise<void> | null,
}));
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
vi.mock(
  "@plasmohq/storage/hook",
  async () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook"),
);
vi.mock("wxt/browser", () => ({
  browser: {
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } },
  },
}));
vi.mock("@/services/service-prompts", () => ({
  isServicePromptScopeUnresolvedError: () => false,
  resolveServicePromptScope: async () => {
    await scope.pending;
    if (!scope.user)
      throw Object.assign(new Error("No active principal"), { status: 401 });
    return {
      config: {
        serverUrl: scope.server,
        authMode: "multi-user",
        authSource: "cookie-session",
      },
      userId: scope.user,
    };
  },
}));
const boundary = async (user: string | null, server = "http://server-a") => {
  await act(async () => {
    scope.user = user;
    scope.server = server;
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"));
  });
};
beforeEach(() => {
  localStorage.clear();
  scope.pending = null; loaded.mockReset(); note.mockReset(); note.mockResolvedValue({}); notices.mockClear();
  scope.user = "alice";
  scope.server = "http://server-a";
});


let history: {entries: SttHistoryEntry[]; onRecompare: (entry: SttHistoryEntry) => Promise<void>};
let comparison: {blob: Blob | null; onComparisonComplete: (results: ComparisonResult[]) => Promise<void>; onSaveToNotes: (text: string, model: string) => Promise<void>};
let recording: {onBlobReady: (blob: Blob | null, duration: number) => void; restoredRecording?: unknown};
vi.mock("react-i18next", () => ({ useTranslation: () => ({t: (_key: string, fallback: string) => fallback}), Trans: () => null }));
vi.mock("@/hooks/useTranscriptionModelsCatalog", () => ({ useTranscriptionModelsCatalog: () => ({ serverModels: [], modelOptions: [], serverModelsLoading: false, serverModelsError: null, retryServerModels: vi.fn() }) }));
const { notices } = vi.hoisted(() => ({notices: vi.fn()}));
vi.mock("@/hooks/useAntdNotification", () => ({ useAntdNotification: () => ({ success: notices, error: notices, info: notices }) }));
const { note } = vi.hoisted(() => ({note: vi.fn().mockResolvedValue({})}));
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { createNote: note } }));
vi.mock("@/components/Common/PageShell", () => ({ PageShell: ({children}: {children: React.ReactNode}) => <>{children}</> }));
vi.mock("../RecordingStrip", () => ({ RecordingStrip: (props: typeof recording) => { recording=props; return null; } }));
vi.mock("../ComparisonPanel", () => ({ ComparisonPanel: (props: typeof comparison) => { comparison=props; return null; } }));
vi.mock("../HistoryPanel", () => ({ HistoryPanel: (props: typeof history) => { history=props; return null; } }));
vi.mock("@/components/Option/Audio/AudioReadinessStrip", () => ({ AudioReadinessStrip: () => null }));
vi.mock("@/components/Option/Audio/AudioPresetControls", () => ({ AudioPresetControls: () => null }));
const { saved, loaded } = vi.hoisted(() => ({ saved: vi.fn().mockResolvedValue("recording-alice"), loaded: vi.fn() }));
vi.mock("@/db/dexie/stt-recordings", () => ({ saveSttRecording: saved, getSttRecording: loaded, deleteSttRecording: vi.fn() }));
import { SttPlaygroundPage } from "../SttPlaygroundPage";
const result: ComparisonResult = { id: "unit-result", model: "parakeet", text: "Alice private transcript", status: "done", latencyMs: 1, wordCount: 3 };

it("B05 history restore never hands a delayed Alice recording to Bob", async () => {
  let resolve!: (recording: { blob: Blob; durationMs: number }) => void;
  loaded.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  const view = render(<SttPlaygroundPage />);
  await act(async () => {});
  await act(async () => { recording.onBlobReady(new Blob(["Alice source"]), 1000); });
  await act(async () => { await comparison.onComparisonComplete([result]); });
  await waitFor(() => expect(history.entries).toHaveLength(1));
  let pending!: Promise<void>;
  act(() => { pending = history.onRecompare(history.entries[0]); });
  await boundary("bob");
  await act(async () => { resolve({ blob: new Blob(["Alice private recording"]), durationMs: 1000 }); await pending; });
  expect(recording.restoredRecording).toBeNull();
  expect(comparison.blob).toBeNull();
  view.unmount();
});

it("B05 history does not adopt unowned legacy transcripts", async () => {
  const storage = new Storage();
  const legacy = [{ id: "legacy", recordingId: "unassigned", results: [result] }];
  await storage.set("sttComparisonHistory", legacy);
  const view=render(<SttPlaygroundPage />);
  await act(async () => {});
  expect(history.entries).toEqual([]);
  expect(await storage.get("sttComparisonHistory")).toEqual(legacy);
  view.unmount();
});

it("B05 history and pending recording stay with their account across a boundary", async () => {
  const view=render(<SttPlaygroundPage />);
  await act(async () => {});
  await act(async () => { recording.onBlobReady(new Blob(["private audio"]), 1000); });
  await act(async () => { await comparison.onComparisonComplete([result]); });
  await waitFor(() => expect(history.entries[0]?.results[0].text).toBe("Alice private transcript"));
  const staleComplete=comparison.onComparisonComplete;
  await boundary("bob");
  expect(history.entries).toEqual([]);
  expect(comparison.blob).toBeNull();
  await act(async () => { await staleComplete([result]); });
  expect(history.entries).toEqual([]);
  await boundary("alice");
  await waitFor(() => expect(history.entries).toHaveLength(1));
  expect(history.entries[0].results[0].text).toBe("Alice private transcript");
  view.unmount();
});

it("B05 Notes handoff pins its current verified account", async () => {
  const view=render(<SttPlaygroundPage />);
  await act(async () => {});
  await act(async () => { await comparison.onSaveToNotes("Reviewed Alice transcript", "parakeet"); });
  expect(note.mock.calls[0][2]).toMatchObject({requestScope:{userId:"alice"}});
  view.unmount();
});

it("B05 Notes handoff stops if the account changes during scope resolution", async () => {
  const view=render(<SttPlaygroundPage />);
  await act(async () => {});
  let release!: () => void;
  scope.pending=new Promise<void>(resolve => {release=resolve;});
  let pending!: Promise<void>;
  await act(async () => { pending=comparison.onSaveToNotes("Alice private draft", "parakeet"); });
  await boundary("bob");
  await act(async () => { release(); await pending; });
  expect(note).not.toHaveBeenCalled();
  view.unmount();
});

it("B05 Notes completion after an account boundary is cancelled and silent", async () => {
  const view=render(<SttPlaygroundPage />);
  await act(async () => {});
  let reject!: (error: Error) => void;
  note.mockImplementationOnce(() => new Promise((_resolve, fail) => {reject=fail;}));
  let pending!: Promise<void>;
  await act(async () => { pending=comparison.onSaveToNotes("Alice reviewed text", "parakeet"); });
  await waitFor(() => expect(note).toHaveBeenCalledTimes(1));
  const signal=note.mock.calls[0][2].signal;
  await boundary("bob");
  expect(signal.aborted).toBe(true);
  await act(async () => { reject(new Error("Retired Alice request")); await pending; });
  expect(notices).not.toHaveBeenCalled();
  view.unmount();
});
