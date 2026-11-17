// @vitest-environment jsdom
import React from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HistorySelectionProvider, useHistorySelectionContext } from "@/hooks/chat/useHistorySelection";
import { useStoreMessageOption } from "@/store/option";
import { ModelPlayground } from "../index";
import * as historyStore from "@/db/dexie/history-selection";
import * as captureService from "@/services/chat-history-selection";
import * as forkOperations from "@/db/dexie/fork-operations";

const fixture = vi.hoisted(() => ({
  current: true,
  capture: null as ReturnType<typeof import("@/hooks/chat/actor-scene-source").captureActorSceneSend>,
  controller: null as import("@/hooks/chat/useHistorySelection").HistorySelectionController | null,
  ancestor: null as import("@/hooks/chat/useHistorySelection").HistorySelectionController | null,
}));

// Account input is controlled; the real controller and scene-capture guards remain active.
vi.mock("@/hooks/useChatDraftOwner", () => {
  const account = {
    requestScope: { config: { serverUrl: "https://model.test", authMode: "single-user" as const }, userId: 341 },
    isCurrent: () => fixture.current,
    retryVerification: vi.fn(),
  };
  return { useChatDraftOwner: () => account };
});
// Only passive display/layout dependencies are stubbed; the leaf exercises the actual Send authority gate.
vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => ({ messages: [], streaming: false }) }));
vi.mock("@/hooks/useSmartScroll", () => ({ useSmartScroll: () => ({
  containerRef: { current: null }, isAutoScrollToBottom: true, autoScrollToBottom: vi.fn(),
}) }));
vi.mock("@plasmohq/storage/hook", () => ({ useStorage: () => [false, vi.fn()] }));
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key }) }));
vi.mock("../ParametersSidebar", () => ({ ParametersSidebar: () => null }));
vi.mock("../DebugPanel", () => ({ DebugPanel: () => null }));
vi.mock("../ModelPlaygroundChat", async () => {
  const { useHistorySelectionContext } = await import("@/hooks/chat/useHistorySelection");
  const { captureActorSceneSend } = await import("@/hooks/chat/actor-scene-source");
  return { ModelPlaygroundChat: () => {
    const controller = useHistorySelectionContext();
    return <button onClick={() => {
      fixture.controller = controller;
      fixture.capture = captureActorSceneSend(controller);
    }}>Prepare model question</button>;
  } };
});

const Ancestor = ({ children }: { children: React.ReactNode }) => {
  fixture.ancestor = useHistorySelectionContext();
  return <>{children}</>;
};
const prepare = (ancestor = false) => {
  const content = ancestor
    ? <HistorySelectionProvider storageKey="model-test-ancestor"><Ancestor><ModelPlayground /></Ancestor></HistorySelectionProvider>
    : <ModelPlayground />;
  render(<MemoryRouter>{content}</MemoryRouter>);
  fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
};

beforeEach(() => {
  fixture.current = true;
  fixture.capture = fixture.controller = fixture.ancestor = null;
  useStoreMessageOption.getState().startNewChatDraft();
  useStoreMessageOption.setState({ historyId: null, serverChatId: null });
});

describe("Model Playground conversation authority", () => {
  it("initializes a missing draft and preserves it on rerender", () => {
    useStoreMessageOption.setState({ chatDraftAddress: null });
    const view = render(<MemoryRouter><ModelPlayground /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
    expect(fixture.capture?.isCurrent()).toBe(true);
    const address = useStoreMessageOption.getState().chatDraftAddress;
    expect(address?.kind).toBe("fresh");
    view.rerender(<MemoryRouter><ModelPlayground /></MemoryRouter>);
    expect(useStoreMessageOption.getState().chatDraftAddress).toBe(address);
    expect(fixture.capture?.isCurrent()).toBe(true);
  });

  it.each(["serverChatId", "historyId"] as const)("keeps a selected %s on Model until its scoped handoff is ready", (field) => {
    useStoreMessageOption.setState({ [field]: "saved-conversation" });
    const Location = () => <output aria-label="Loading route">{useLocation().pathname}</output>;
    render(<MemoryRouter><ModelPlayground /><Location /></MemoryRouter>);
    const handoff = screen.getByRole("button", { name: "Simple Chat" });
    expect(handoff).toBeDisabled();
    fireEvent.click(handoff);
    expect(screen.getByLabelText("Loading route").textContent).toBe("/");
  });


  it.each(["live loading", "newer selection", "qualified refresh"] as const)("waits for a qualified capture through %s before handoff", async (scenario) => {
    sessionStorage.removeItem("tldw-h1-playground-reference");
    const profile = vi.spyOn(historyStore, "ensureLocalProfileId").mockResolvedValue("model-profile");
    const forks = vi.spyOn(forkOperations, "findForkCandidate").mockResolvedValue(null);
    const readBookmark = vi.spyOn(historyStore, "loadHistoryBookmark").mockResolvedValue(null);
    const saveBookmark = vi.spyOn(historyStore, "saveHistoryBookmark").mockResolvedValue(undefined);
    const receipt = (view: import("@/types/history-selection").HistoryCaptureRequestV1["view"]) => ({
      status: "captured" as const,
      purpose: "send" as const,
      storage_context_digest: "model-storage",
      snapshot: {
        version: 1 as const,
        owner_key: "native-owner",
        conversation_id: view.conversation_id,
        fences: { conversation: "1", history: "1", settings: "1" },
        nodes: [],
        source_digest: "model-source",
        interpretation_status: { kind: "parent_graph_v1" as const },
        storage_context_digest: "model-storage",
      },
      view: { ...view, owner_key: "native-owner" },
      rows: [],
      selected_content: [],
    });
    const capture = vi.spyOn(captureService, "captureHistorySnapshot")
      .mockImplementation(async (_owner, view) => receipt(view));
    try {
      useStoreMessageOption.setState({ serverChatId: "saved-conversation" });
      const Location = () => <output aria-label="Qualified route">{useLocation().pathname + useLocation().search}</output>;
      render(<MemoryRouter><ModelPlayground /><Location /></MemoryRouter>);
      fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
      const controller = fixture.controller!;
      const handoff = screen.getByRole("button", { name: "Simple Chat" });
      expect(handoff).toBeDisabled();
      await act(async () => {
        expect(await controller.open({
          kind: "native",
          owner_key: "native-owner",
          conversation_id: "saved-conversation",
          scope: { type: "global" },
          request_scope: { userId: 341, config: { serverUrl: "https://model.test", authMode: "single-user" } },
          validate_lease: () => true,
        })).toBe(true);
      });
      expect(controller.getCurrent().error).toBeNull();
      expect(controller.getCurrent().status).toBe("ready");
      expect(controller.getReference()?.conversation_id).toBe("saved-conversation");
      expect(useStoreMessageOption.getState().serverChatId).toBe("saved-conversation");
      expect(screen.getByRole("button", { name: "Simple Chat" })).toBeEnabled();
      fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
      const liveController = fixture.controller!;
      if (scenario === "live loading") {
        const read = vi.spyOn(liveController, "getCurrent").mockReturnValue({
          ...liveController.getCurrent(), settingsQualified: false, status: "loading",
        });
        const prepare = vi.spyOn(liveController, "prepareExpansionPath");
        try {
          fireEvent.click(screen.getByRole("button", { name: "Simple Chat" }));
          await act(async () => {});
          expect(prepare).not.toHaveBeenCalled();
          expect(screen.getByLabelText("Qualified route").textContent).toBe("/");
        } finally { prepare.mockRestore(); read.mockRestore(); }
        return;
      }
      if (scenario === "newer selection") {
        let complete!: (path: string) => void;
        const prepare = vi.spyOn(liveController, "prepareExpansionPath")
          .mockReturnValue(new Promise<string>((resolve) => { complete = resolve; }));
        try {
          fireEvent.click(screen.getByRole("button", { name: "Simple Chat" }));
          act(() => useStoreMessageOption.setState({ serverChatId: "newer-conversation" }));
          await act(async () => { complete("/chat?historySelection=older-conversation"); });
          expect(prepare).toHaveBeenCalledOnce();
          expect(screen.getByLabelText("Qualified route").textContent).toBe("/");
        } finally { prepare.mockRestore(); }
        return;
      }
      let complete!: () => void;
      capture.mockImplementationOnce((_owner, view) => new Promise((resolve) => {
        complete = () => resolve(receipt(view));
      }));
      let refresh!: Promise<boolean>;
      act(() => { refresh = controller.refresh(); });
      expect(controller.getReference()).not.toBeNull();
      expect(handoff).toBeDisabled();
      await act(async () => { complete(); expect(await refresh).toBe(true); });
      expect(handoff).toBeEnabled();
      fireEvent.click(handoff);
      await waitFor(() => expect(screen.getByLabelText("Qualified route").textContent).toMatch(/^\/chat\?historySelection=/));
      const route = new URL(screen.getByLabelText("Qualified route").textContent!, "https://model.test");
      expect(JSON.parse(route.searchParams.get("historySelection")!)).toMatchObject({
        conversation_id: "saved-conversation", owner_kind: "native",
      });
    } finally {
      capture.mockRestore(); saveBookmark.mockRestore(); readBookmark.mockRestore(); forks.mockRestore(); profile.mockRestore();
      sessionStorage.removeItem("tldw-h1-playground-reference");
    }
  });

  it("hands Simple Chat to the canonical Chat route", async () => {
    const Location = () => <output aria-label="Current route">{useLocation().pathname}</output>;
    render(<MemoryRouter><ModelPlayground /><Location /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Simple Chat" }));
    await waitFor(() => expect(screen.getByLabelText("Current route")).toHaveTextContent(/^\/chat$/));
  });

  it.each([
    ["/chat?historySelection=prepared-scoped-reference", "/chat?historySelection=prepared-scoped-reference"],
    [null, "/"],
  ])("uses the prepared conversation handoff %s", async (prepared, expected) => {
    const Location = () => <output aria-label="Prepared route">{useLocation().pathname + useLocation().search}</output>;
    render(<MemoryRouter><ModelPlayground /><Location /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
    const preparePath = vi.spyOn(fixture.controller!, "prepareExpansionPath").mockResolvedValue(prepared);
    fireEvent.click(screen.getByRole("button", { name: "Simple Chat" }));
    await waitFor(() => expect(preparePath).toHaveBeenCalledOnce());
    expect(screen.getByLabelText("Prepared route").textContent).toBe(expected);
  });

  it("keeps a retired handoff from navigating after preparation", async () => {
    const Location = () => <output aria-label="Retired route">{useLocation().pathname + useLocation().search}</output>;
    render(<MemoryRouter><ModelPlayground /><Location /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
    let resolve!: (path: string) => void;
    const preparePath = vi.spyOn(fixture.controller!, "prepareExpansionPath")
      .mockReturnValue(new Promise<string>((done) => { resolve = done; }));
    fireEvent.click(screen.getByRole("button", { name: "Simple Chat" }));
    act(() => fixture.controller!.reset());
    await act(async () => { resolve("/chat?historySelection=retired-reference"); });
    expect(preparePath).toHaveBeenCalledOnce();
    expect(screen.getByLabelText("Retired route").textContent).toBe("/");
  });

  it("admits a standalone question under its verified account", () => {
    prepare();
    expect(fixture.capture?.isCurrent()).toBe(true);
    expect(fixture.controller?.draftAccount?.requestScope?.userId).toBe(341);
  });

  it("keeps the ancestor controller when one already owns the surface", () => {
    prepare(true);
    expect(fixture.capture?.isCurrent()).toBe(true);
    expect(fixture.controller).toBe(fixture.ancestor);
  });

  it("retires a prepared question when the account changes", () => {
    prepare();
    fixture.current = false;
    expect(fixture.capture?.isCurrent()).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Prepare model question" }));
    expect(fixture.capture).toBeNull();
  });
});
