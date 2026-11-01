// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { HistorySelectionProvider, useHistorySelectionContext } from "@/hooks/chat/useHistorySelection";
import { useStoreMessageOption } from "@/store/option";
import { ModelPlayground } from "../index";

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

  it("hands Simple Chat to the canonical Chat route", () => {
    const Location = () => <output aria-label="Current route">{useLocation().pathname}</output>;
    render(<MemoryRouter><ModelPlayground /><Location /></MemoryRouter>);
    fireEvent.click(screen.getByRole("button", { name: "Simple Chat" }));
    expect(screen.getByLabelText("Current route")).toHaveTextContent(/^\/chat$/);
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
