// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundRuntimeInspector } from "../PlaygroundRuntimeInspector";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback,
  }),
}));

const renderInspector = (
  overrides: Partial<React.ComponentProps<typeof PlaygroundRuntimeInspector>> = {},
) => {
  const props = {
    streaming: false,
    selectedProvider: "openai",
    selectedModel: "gpt-4.1-mini",
    providerRouteLabel: "openai:gpt-4.1-mini",
    runtimeStatus: "ready" as const,
    runtimeStatusDetail: null,
    messageCount: 2,
    threadSearchOpen: false,
    assistantSummary: {
      mode: "character" as const,
      name: "Mira Vale",
      detail: "Character selected",
    },
    onOpenModelSettings: vi.fn(),
    onOpenAssistantSelect: vi.fn(),
    onOpenSceneDirector: vi.fn(),
    onOpenMcpSettings: vi.fn(),
    toolChoice: "auto" as const,
    onToolChoiceChange: vi.fn(),
    canStopStreaming: false,
    onStopStreaming: vi.fn(),
    canRegenerate: true,
    onRegenerate: vi.fn(),
    ...overrides,
  };

  render(<PlaygroundRuntimeInspector {...props} />);

  return props;
};

describe("PlaygroundRuntimeInspector first-slice controls", () => {
  it("renders provider and model as separate fields", () => {
    renderInspector();

    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText("openai")).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    expect(screen.getByText("gpt-4.1-mini")).toBeInTheDocument();
    expect(screen.getByText("Route openai:gpt-4.1-mini")).toBeInTheDocument();
  });

  it("derives provider and model display from provider-qualified selected model when provider is missing", () => {
    renderInspector({
      selectedProvider: null,
      selectedModel: "anthropic:claude-sonnet-4",
      providerRouteLabel: null,
    });

    expect(screen.getByText("anthropic")).toBeInTheDocument();
    expect(screen.getByText("claude-sonnet-4")).toBeInTheDocument();
    expect(
      screen.getByText("Route anthropic:claude-sonnet-4"),
    ).toBeInTheDocument();
  });

  it("opens model/chat settings and assistant selection through supplied callbacks", () => {
    const props = renderInspector();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Model & Chat settings" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Select character or persona" }),
    );

    expect(props.onOpenModelSettings).toHaveBeenCalledTimes(1);
    expect(props.onOpenAssistantSelect).toHaveBeenCalledTimes(1);
    expect(props.onOpenSceneDirector).not.toHaveBeenCalled();
  });

  it("keeps Scene Director secondary to the primary assistant selector", () => {
    const props = renderInspector();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Scene Director" }),
    );

    expect(props.onOpenSceneDirector).toHaveBeenCalledTimes(1);
    expect(props.onOpenAssistantSelect).not.toHaveBeenCalled();
  });

  it("changes MCP tool choice and opens MCP settings directly", () => {
    const props = renderInspector({
      toolSummary: {
        state: "available",
        label: "MCP tools",
        detail: "3 chat tools enabled",
      },
    });

    expect(
      screen.getByRole("button", { name: "MCP tool choice Auto" }),
    ).toHaveAttribute("aria-pressed", "true");

    fireEvent.click(
      screen.getByRole("button", { name: "MCP tool choice Required" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Configure MCP tools" }));

    expect(props.onToolChoiceChange).toHaveBeenCalledWith("required");
    expect(props.onOpenMcpSettings).toHaveBeenCalledTimes(1);
  });

  it("calls the existing stop handler while streaming", () => {
    const props = renderInspector({
      streaming: true,
      runtimeStatus: "streaming",
      canStopStreaming: true,
      canRegenerate: false,
    });

    fireEvent.click(screen.getByRole("button", { name: "Stop generation" }));

    expect(props.onStopStreaming).toHaveBeenCalledTimes(1);
    expect(
      screen.queryByRole("button", { name: "Regenerate last response" }),
    ).toBeNull();
  });

  it("calls the existing regenerate handler when ready", () => {
    const props = renderInspector({
      streaming: false,
      runtimeStatus: "ready",
      canStopStreaming: false,
      canRegenerate: true,
    });

    fireEvent.click(
      screen.getByRole("button", { name: "Regenerate last response" }),
    );

    expect(props.onRegenerate).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Stop generation" })).toBeNull();
  });

  it("hides stop and regenerate controls when the shared handlers are unavailable", () => {
    renderInspector({
      canStopStreaming: false,
      canRegenerate: false,
      onStopStreaming: undefined,
      onRegenerate: undefined,
    });

    expect(
      screen.queryByRole("button", { name: "Stop generation" }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Regenerate last response" }),
    ).toBeNull();
  });

  it("renders streaming, degraded, and error details through explicit status props", () => {
    const { rerender } = render(
      <PlaygroundRuntimeInspector
        streaming
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        providerRouteLabel="openai:gpt-4.1-mini"
        runtimeStatus="streaming"
        runtimeStatusDetail={null}
        messageCount={2}
        threadSearchOpen={false}
        assistantSummary={{ mode: "none", name: null, detail: null }}
        onOpenModelSettings={vi.fn()}
        onOpenAssistantSelect={vi.fn()}
        canStopStreaming
        onStopStreaming={vi.fn()}
      />,
    );

    expect(screen.getByText("Streaming")).toBeInTheDocument();

    rerender(
      <PlaygroundRuntimeInspector
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        providerRouteLabel="openai:gpt-4.1-mini"
        runtimeStatus="degraded"
        runtimeStatusDetail="Provider metadata degraded"
        messageCount={2}
        threadSearchOpen={false}
        assistantSummary={{ mode: "none", name: null, detail: null }}
        onOpenModelSettings={vi.fn()}
        onOpenAssistantSelect={vi.fn()}
      />,
    );

    expect(screen.getByText("Degraded")).toBeInTheDocument();
    expect(screen.getByText("Provider metadata degraded")).toBeInTheDocument();

    rerender(
      <PlaygroundRuntimeInspector
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        providerRouteLabel="openai:gpt-4.1-mini"
        runtimeStatus="error"
        runtimeStatusDetail="Provider failed"
        messageCount={2}
        threadSearchOpen={false}
        assistantSummary={{ mode: "none", name: null, detail: null }}
        onOpenModelSettings={vi.fn()}
        onOpenAssistantSelect={vi.fn()}
      />,
    );

    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("Provider failed")).toBeInTheDocument();
  });
});
