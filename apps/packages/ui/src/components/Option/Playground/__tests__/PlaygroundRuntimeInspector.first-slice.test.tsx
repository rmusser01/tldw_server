// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
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
    onClearAssistant: vi.fn(),
    onInspectAssistant: vi.fn(),
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

  it("keeps cockpit rail sections in runtime configuration order", () => {
    renderInspector({
      settingSummaries: [{ label: "Temperature", value: "0.7" }],
      toolSummary: {
        state: "available",
        label: "MCP tools",
        detail: "3 chat tools enabled",
      },
    });

    expect(
      screen.getAllByRole("heading", { level: 2 }).map((heading) => heading.textContent),
    ).toEqual([
      "Runtime",
      "Model route",
      "Assistant",
      "MCP tools",
      "Run controls",
    ]);
  });

  it("keeps first-time runtime controls discoverable when nothing is configured", () => {
    renderInspector({
      selectedProvider: null,
      selectedModel: null,
      providerRouteLabel: null,
      messageCount: 0,
      assistantSummary: {
        mode: "none",
        name: null,
        detail: null,
      },
      canRegenerate: false,
      toolSummary: {
        state: "unavailable",
        label: "MCP unavailable",
        detail: "MCP tools unavailable",
      },
    });

    expect(screen.getByText("No provider selected")).toBeInTheDocument();
    expect(screen.getByText("No model selected")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open model settings" }),
    ).toBeInTheDocument();

    const assistant = screen.getByRole("region", { name: "Assistant" });
    expect(within(assistant).getByText("No assistant selected")).toBeInTheDocument();
    expect(
      within(assistant).getByText("No persona or character will shape replies."),
    ).toBeInTheDocument();
    expect(
      within(assistant).getByRole("button", {
        name: "Select character or persona",
      }),
    ).toBeInTheDocument();

    const tools = screen.getByRole("region", { name: "MCP tools" });
    expect(within(tools).getByText("MCP unavailable")).toBeInTheDocument();
    expect(within(tools).getByText("MCP tools unavailable")).toBeInTheDocument();
    expect(
      within(tools).getByRole("button", { name: "Configure MCP tools" }),
    ).toBeInTheDocument();
    expect(
      within(tools).queryByRole("button", { name: "MCP tool choice Auto" }),
    ).toBeNull();

    const runControls = screen.getByRole("region", { name: "Run controls" });
    expect(within(runControls).getByText("0 messages")).toBeInTheDocument();
    expect(within(runControls).getByText("Search closed")).toBeInTheDocument();
    expect(within(runControls).getByText("No turn is running.")).toBeInTheDocument();
    expect(
      within(runControls).getByText(
        "Regenerate becomes available after an assistant response.",
      ),
    ).toBeInTheDocument();
  });

  it("lets each right rail section collapse without replacing the side rail", () => {
    renderInspector({
      settingSummaries: [{ label: "Temperature", value: "0.7" }],
      toolSummary: {
        state: "available",
        label: "MCP tools",
        detail: "3 chat tools enabled",
      },
    });

    const rail = screen.getByTestId("playground-runtime-inspector");
    const collapseRuntime = within(rail).getByRole("button", {
      name: "Collapse Runtime",
    });
    const runtimePanelId = collapseRuntime.getAttribute("aria-controls");
    const runtimePanel = runtimePanelId
      ? document.getElementById(runtimePanelId)
      : null;

    expect(collapseRuntime).toHaveAttribute("aria-expanded", "true");
    expect(runtimePanel).not.toBeNull();
    expect(runtimePanel).toHaveAttribute("aria-hidden", "false");
    expect(runtimePanel).not.toHaveClass("hidden");

    fireEvent.click(collapseRuntime);

    expect(rail).toBeInTheDocument();
    expect(collapseRuntime).toHaveAttribute("aria-expanded", "false");
    expect(runtimePanel).toHaveAttribute("aria-hidden", "true");
    expect(runtimePanel).toHaveClass("hidden");
    expect(
      within(rail).getByRole("button", { name: "Expand Runtime" }),
    ).toBeInTheDocument();

    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse Model route" }),
    );
    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse Assistant" }),
    );
    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse MCP tools" }),
    );
    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse Run controls" }),
    );

    expect(screen.getByRole("heading", { name: "Runtime" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Model route" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Assistant" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "MCP tools" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Run controls" })).toBeInTheDocument();
  });

  it("preserves existing right rail actions after regrouping", () => {
    const props = renderInspector({
      streaming: true,
      runtimeStatus: "streaming",
      canStopStreaming: true,
      canRegenerate: false,
      settingSummaries: [{ label: "Temperature", value: "0.7" }],
      toolSummary: {
        state: "available",
        label: "MCP tools",
        detail: "3 chat tools enabled",
      },
    });

    fireEvent.click(
      screen.getByRole("button", { name: "Open model settings" }),
    );

    const assistant = screen.getByRole("region", { name: "Assistant" });
    fireEvent.click(
      within(assistant).getByRole("button", {
        name: "Select character or persona",
      }),
    );
    fireEvent.click(
      within(assistant).getByRole("button", { name: "Manage assistant" }),
    );
    fireEvent.click(
      within(assistant).getByRole("button", { name: "Open Scene Director" }),
    );
    fireEvent.click(
      within(assistant).getByRole("button", { name: "Clear assistant" }),
    );

    const tools = screen.getByRole("region", { name: "MCP tools" });
    fireEvent.click(
      within(tools).getByRole("button", { name: "MCP tool choice Required" }),
    );
    fireEvent.click(
      within(tools).getByRole("button", { name: "Configure MCP tools" }),
    );

    const runControls = screen.getByRole("region", { name: "Run controls" });
    fireEvent.click(
      within(runControls).getByRole("button", { name: "Stop generation" }),
    );

    expect(props.onOpenModelSettings).toHaveBeenCalledTimes(1);
    expect(props.onOpenAssistantSelect).toHaveBeenCalledTimes(1);
    expect(props.onInspectAssistant).toHaveBeenCalledTimes(1);
    expect(props.onOpenSceneDirector).toHaveBeenCalledTimes(1);
    expect(props.onClearAssistant).toHaveBeenCalledTimes(1);
    expect(props.onToolChoiceChange).toHaveBeenCalledWith("required");
    expect(props.onOpenMcpSettings).toHaveBeenCalledTimes(1);
    expect(props.onStopStreaming).toHaveBeenCalledTimes(1);
    expect(props.onRegenerate).not.toHaveBeenCalled();
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
      screen.getByRole("button", { name: "Open model settings" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Select character or persona" }),
    );

    expect(props.onOpenModelSettings).toHaveBeenCalledTimes(1);
    expect(props.onOpenAssistantSelect).toHaveBeenCalledTimes(1);
    expect(props.onOpenSceneDirector).not.toHaveBeenCalled();
  });

  it("distinguishes inherited defaults from provider:model overrides", () => {
    renderInspector({
      settingSummaries: [
        {
          label: "Temperature",
          value: "0.7",
          source: "default",
        },
        {
          label: "Context",
          value: "8192",
          source: "override",
        },
      ],
    });

    expect(screen.getByText("Temperature")).toBeInTheDocument();
    expect(screen.getByText("0.7")).toBeInTheDocument();
    expect(screen.getByText("Inherited")).toBeInTheDocument();
    expect(screen.getByText("Context")).toBeInTheDocument();
    expect(screen.getByText("8192")).toBeInTheDocument();
    expect(screen.getByText("Override")).toBeInTheDocument();
  });

  it("shows a clear assistant action when a character is selected", () => {
    const props = renderInspector();

    fireEvent.click(screen.getByRole("button", { name: "Clear assistant" }));

    expect(props.onClearAssistant).toHaveBeenCalledTimes(1);
  });

  it("shows a clear assistant action when a persona is selected", () => {
    const props = renderInspector({
      assistantSummary: {
        mode: "persona",
        name: "Research Persona",
        detail: "Persona selected",
      },
    });

    fireEvent.click(screen.getByRole("button", { name: "Clear assistant" }));

    expect(props.onClearAssistant).toHaveBeenCalledTimes(1);
  });

  it("opens the supplied manage path for selected assistants", () => {
    const props = renderInspector();

    fireEvent.click(screen.getByRole("button", { name: "Manage assistant" }));

    expect(props.onInspectAssistant).toHaveBeenCalledTimes(1);
  });

  it("keeps Scene Director secondary to the primary assistant selector", () => {
    const props = renderInspector();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Scene Director" }),
    );

    expect(props.onOpenSceneDirector).toHaveBeenCalledTimes(1);
    expect(props.onOpenAssistantSelect).not.toHaveBeenCalled();
  });

  it("keeps Scene Director character-only and explains persona mode", () => {
    renderInspector({
      assistantSummary: {
        mode: "persona",
        name: "Research Persona",
        detail: "Persona selected",
      },
    });

    expect(
      screen.queryByRole("button", { name: "Open Scene Director" }),
    ).toBeNull();
    expect(
      screen.getByText("Scene Director is available for character-backed chats."),
    ).toBeInTheDocument();
  });

  it("changes MCP tool choice and opens MCP settings directly", () => {
    const props = renderInspector({
      toolSummary: {
        state: "available",
        label: "MCP tools",
        detail: "3 chat tools enabled",
        stateCounts: [
          { label: "Discovered", value: 4 },
          { label: "Executable", value: 3 },
          { label: "Chat-enabled", value: 2 },
          { label: "User-disabled", value: 1 },
        ],
      },
    });

    const mcpCounts = screen.getByLabelText("MCP tool state counts");
    expect(within(mcpCounts).getByText("Discovered")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("4")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("Executable")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("3")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("Chat-enabled")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("2")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("User-disabled")).toBeInTheDocument();
    expect(within(mcpCounts).getByText("1")).toBeInTheDocument();

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

  it("keeps unavailable MCP informational without enabling dead-end tool choice", () => {
    const props = renderInspector({
      toolChoice: "auto",
      toolSummary: {
        state: "unavailable",
        label: "MCP unavailable",
        detail: "MCP tools unavailable",
      },
    });

    expect(screen.getByText("MCP unavailable")).toBeInTheDocument();
    expect(screen.getByText("MCP tools unavailable")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "MCP tool choice Auto" }),
    ).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Configure MCP tools" }));

    expect(props.onToolChoiceChange).not.toHaveBeenCalled();
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
      screen.getByRole("button", { name: "Regenerate last response" }),
    ).toBeDisabled();
    expect(
      screen.getByText("Wait for the current turn to finish before regenerating."),
    ).toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: "Stop generation" })).toBeDisabled();
    expect(screen.getByText("No turn is running.")).toBeInTheDocument();
  });

  it("surfaces empty assistant responses beside the regenerate control", () => {
    const props = renderInspector({
      streaming: false,
      runtimeStatus: "ready",
      canStopStreaming: false,
      canRegenerate: true,
      emptyAssistantResponse: true,
    });

    const runControls = screen.getByRole("region", { name: "Run controls" });
    const status = within(runControls).getByRole("status", {
      name: "Empty assistant response",
    });
    expect(status).toHaveTextContent("No response text returned.");
    expect(status).toHaveTextContent(
      "Regenerate this turn or switch model settings before trying again.",
    );

    fireEvent.click(
      within(runControls).getByRole("button", {
        name: "Regenerate last response",
      }),
    );

    expect(props.onRegenerate).toHaveBeenCalledTimes(1);
  });

  it("explains unavailable run controls when shared handlers are unavailable", () => {
    renderInspector({
      canStopStreaming: false,
      canRegenerate: false,
      onStopStreaming: undefined,
      onRegenerate: undefined,
    });

    expect(
      screen.getByRole("button", { name: "Stop generation" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Regenerate last response" }),
    ).toBeDisabled();
    expect(screen.getByText("No turn is running.")).toBeInTheDocument();
    expect(
      screen.getByText("Regenerate becomes available after an assistant response."),
    ).toBeInTheDocument();
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
