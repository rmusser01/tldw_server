// @vitest-environment jsdom
import React from "react";
import type {} from "@testing-library/jest-dom/vitest";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils";
import { ModelPlaygroundChat } from "../ModelPlaygroundChat";

const turn = vi.hoisted(() => ({
  submit: vi.fn(),
  streaming: false,
  compare: false,
  current: true,
  hasMessage: false,
  hasContext: true,
}));

// The tested owner is ModelPlaygroundChat's own textarea state. Its chat
// provider is held at the callback boundary while the component stays real.
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    messages: turn.hasMessage ? [{ id: "existing", message: "saved", isBot: false }] : [],
    streaming: turn.streaming,
    isProcessing: false,
    regenerateLastMessage: vi.fn(),
    editMessage: vi.fn(),
    deleteMessage: vi.fn(),
    onSubmit: turn.submit,
    stopStreamingRequest: vi.fn(),
    selectedModel: "ready-model",
    setSelectedModel: vi.fn(),
    compareMode: turn.compare,
    setCompareMode: vi.fn(),
    compareSelectedModels: turn.compare ? ["ready-model"] : [],
    setCompareSelectedModels: vi.fn(),
    compareMaxModels: 3,
    clearChat: vi.fn(),
    ttsEnabled: false,
    historyId: null,
  }),
}));
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => turn.hasContext ? { fence: () => () => turn.current } : null,
}));
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: () => null,
}));
vi.mock("@/hooks/useChatModelCatalog", () => ({
  useChatModelCatalog: () => ({ data: [] }),
  EMPTY_CHAT_MODELS: [],
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}));
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() }),
}));
vi.mock("@/components/Option/Playground/PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => null,
}));

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
};

describe("Model Playground local draft ownership", () => {
  beforeEach(() => {
    turn.submit.mockReset();
    turn.streaming = false;
    turn.compare = false;
    turn.current = true;
    turn.hasMessage = false;
    turn.hasContext = true;
  });

  it("names the primary model and conversation controls", () => {
    render(<ModelPlaygroundChat />);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Clear chat" })).toBeDisabled();
    expect(screen.getByRole("switch", { name: "Compare Mode" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Select model" })).toBeInTheDocument();
  });

  it("names Stop while a reply is streaming", () => {
    turn.streaming = true;
    render(<ModelPlaygroundChat />);
    expect(screen.getByRole("button", { name: "Stop" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Send" })).not.toBeInTheDocument();
  });

  it("names Compare model addition and removal", () => {
    turn.compare = true;
    render(<ModelPlaygroundChat />);
    expect(screen.getByRole("button", { name: "Remove ready-model" })).toBeEnabled();
    expect(screen.getByRole("combobox", { name: "Add model" })).toBeInTheDocument();
  });

  it.each([
    { label: "submitted", result: { status: "submitted" } as ChatSubmitResult },
    {
      label: "failed",
      result: {
        status: "failed",
        errorMessage: "Provider unavailable",
        restoreDraft: true,
      } as ChatSubmitResult,
    },
    {
      label: "skipped",
      result: {
        status: "skipped",
        reason: "Request scope changed",
      } as ChatSubmitResult,
    },
  ])("keeps text typed after a held $label turn", async ({ result }) => {
    const response = deferred<ChatSubmitResult>();
    turn.submit.mockReturnValue(response.promise);
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    try {
      fireEvent.change(input, { target: { value: "first" } });
      fireEvent.click(input.parentElement!.querySelector("button")!);
      await waitFor(() =>
        expect(turn.submit).toHaveBeenCalledWith({
          message: "first",
          image: "",
        }),
      );
      expect(input).toHaveValue("");
      fireEvent.change(input, { target: { value: "second" } });
      await act(async () => {
        response.resolve(result);
        await response.promise;
      });
      expect(input).toHaveValue("second");
      expect(turn.submit).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => {
        response.resolve(result);
        await response.promise;
      });
    }
  });
  it.each([
    { label: "proven refusal", result: { status: "failed", errorMessage: "Disabled", restoreDraft: true }, restores: true },
    { label: "unknown outcome", result: { status: "failed", errorMessage: "Transport lost" }, restores: false },
    { label: "stopped turn", result: { status: "skipped", reason: "Request cancelled" }, restores: false },
  ] as const)("restores only the $label draft", async ({ result, restores }) => {
    turn.submit.mockResolvedValue(result);
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "  exact refused question  " } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await act(async () => { await Promise.resolve(); });
    expect(turn.submit).toHaveBeenCalledWith({ message: "exact refused question", image: "" });
    expect(input).toHaveValue(restores ? "  exact refused question  " : "");
  });

  it.each(["typed then cleared", "source retired"] as const)("does not restore after $0", async change => {
    const response = deferred<ChatSubmitResult>();
    turn.submit.mockReturnValue(response.promise);
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(turn.submit).toHaveBeenCalledOnce());
    if (change === "typed then cleared") {
      fireEvent.change(input, { target: { value: "newer" } });
      fireEvent.change(input, { target: { value: "" } });
    } else turn.current = false;
    await act(async () => { response.resolve({ status: "failed", errorMessage: "Disabled", restoreDraft: true }); await response.promise; });
    expect(input).toHaveValue("");
  });

  it("does not restore an older refusal into a newer pending submission", async () => {
    const first = deferred<ChatSubmitResult>(), second = deferred<ChatSubmitResult>();
    turn.submit.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    fireEvent.change(input, { target: { value: "second" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(turn.submit).toHaveBeenCalledTimes(2));
    await act(async () => { first.resolve({ status: "failed", errorMessage: "Disabled", restoreDraft: true }); await first.promise; });
    expect(input).toHaveValue("");
    await act(async () => { second.resolve({ status: "submitted" }); await second.promise; });
    expect(input).toHaveValue("");
  });

  it.each([true, false])("restores only the current post-creation owner (%s)", async current => {
    turn.submit.mockImplementation(async () => {
      turn.current = false;
      return { status: "failed", errorMessage: "Disabled", restoreDraft: true,
        restoreDraftIsCurrent: () => current };
    });
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await act(async () => { await Promise.resolve(); });
    expect(input).toHaveValue(current ? "first" : "");
  });

  it("does not restore after Clear chat invalidates the pending draft", async () => {
    turn.hasMessage = true;
    const response = deferred<ChatSubmitResult>();
    turn.submit.mockReturnValue(response.promise);
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "first" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear chat" }));
    await act(async () => { response.resolve({ status: "failed", errorMessage: "Disabled", restoreDraft: true,
      restoreDraftIsCurrent: () => true }); await response.promise; });
    expect(input).toHaveValue("");
  });

  it("restores a safe refusal without history provider", async () => {
    turn.hasContext = false;
    turn.submit.mockResolvedValue({ status: "failed", errorMessage: "Disabled", restoreDraft: true });
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "retained" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await act(async () => { await Promise.resolve(); });
    expect(turn.submit).toHaveBeenCalledOnce();
    expect(input).toHaveValue("retained");
  });

  it("does not restore a mixed Compare outcome as an ordinary Send", async () => {
    turn.compare = true;
    turn.submit.mockResolvedValue({ status: "failed", errorMessage: "One provider refused", restoreDraft: true });
    render(<ModelPlaygroundChat />);
    const input = screen.getByPlaceholderText("Type your message...");
    fireEvent.change(input, { target: { value: "compare question" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await act(async () => { await Promise.resolve(); });
    expect(turn.submit).toHaveBeenCalledOnce();
    expect(input).toHaveValue("");
  });

});
