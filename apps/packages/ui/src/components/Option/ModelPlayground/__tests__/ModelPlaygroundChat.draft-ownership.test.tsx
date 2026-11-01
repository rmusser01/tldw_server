// @vitest-environment jsdom
import React from "react";
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
}));

// The tested owner is ModelPlaygroundChat's own textarea state. Its chat
// provider is held at the callback boundary while the component stays real.
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    messages: [],
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
});
