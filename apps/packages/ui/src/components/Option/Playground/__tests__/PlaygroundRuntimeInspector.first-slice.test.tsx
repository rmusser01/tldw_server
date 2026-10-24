// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundRuntimeInspector } from "../PlaygroundRuntimeInspector";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?:
        | string
        | { defaultValue?: string; [key: string]: unknown },
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions;
      const template = fallbackOrOptions?.defaultValue || key;
      return template.replace(/\{\{(\w+)\}\}/g, (_, token: string) => {
        const value = fallbackOrOptions?.[token];
        return value == null ? `{{${token}}}` : String(value);
      });
    },
  }),
}));

const renderInspector = (
  overrides: Partial<
    React.ComponentProps<typeof PlaygroundRuntimeInspector>
  > = {},
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
    onOpenModelSettings: vi.fn(),
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
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual(["Current run", "Model & settings", "MCP tools"]);
  });

  it("keeps first-time runtime controls discoverable when nothing is configured", () => {
    renderInspector({
      selectedProvider: null,
      selectedModel: null,
      providerRouteLabel: null,
      messageCount: 0,
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

    expect(screen.queryByRole("region", { name: "Assistant" })).toBeNull();

    const tools = screen.getByRole("region", { name: "MCP tools" });
    expect(within(tools).getByText("MCP unavailable")).toBeInTheDocument();
    expect(
      within(tools).getByText("MCP tools unavailable"),
    ).toBeInTheDocument();
    expect(
      within(tools).getByRole("button", { name: "Configure MCP tools" }),
    ).toBeInTheDocument();
    expect(
      within(tools).queryByRole("button", { name: "MCP tool choice Auto" }),
    ).toBeNull();

    const runControls = screen.getByRole("region", { name: "Run controls" });
    expect(within(runControls).queryByText("0 messages")).toBeNull();
    expect(within(runControls).queryByText("Search closed")).toBeNull();
    expect(
      within(runControls).getByText("No turn is running."),
    ).toBeInTheDocument();
    expect(
      within(runControls).getByText(
        "Regenerate becomes available after an assistant response.",
      ),
    ).toBeInTheDocument();
  });

  it("lets each right rail section collapse without replacing the side rail", () => {
    renderInspector();
    const rail = screen.getByTestId("playground-runtime-inspector");
    for (const title of ["Current run", "Model & settings", "MCP tools"]) {
      const toggle = within(rail).getByRole("button", {
        name: `Collapse ${title}`,
      });
      const panel = document.getElementById(
        toggle.getAttribute("aria-controls")!,
      );
      fireEvent.click(toggle);
      expect(panel).toHaveAttribute("aria-hidden", "true");
      expect(
        within(rail).getByRole("button", { name: `Expand ${title}` }),
      ).toBeVisible();
    }
  });

  it("keeps recovery visible while secondary tools collapse in setup mode", () => {
    renderInspector({
      setupRecoveryMode: true,
      runtimeStatus: "error",
      runtimeStatusDetail: "Provider setup needed",
    });
    expect(screen.getByText("Provider setup needed")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Open model settings" }),
    ).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Expand MCP tools" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.getByRole("button", { name: "Regenerate last response" }),
    ).toBeVisible();
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

  it("opens scoped model settings through the supplied callback", () => {
    const props = renderInspector();
    fireEvent.click(
      screen.getByRole("button", { name: "Open model settings" }),
    );
    expect(props.onOpenModelSettings).toHaveBeenCalledOnce();
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
    expect(screen.getByText("0.7")).not.toBeVisible();
    fireEvent.click(screen.getByText("Default settings"));
    expect(screen.getByText("0.7")).toBeVisible();
    expect(screen.getByText("Context")).toBeInTheDocument();
    expect(screen.getByText("8192")).toBeInTheDocument();
    expect(screen.getByText("Override")).toBeInTheDocument();
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
    fireEvent.click(
      screen.getByRole("button", { name: "Configure MCP tools" }),
    );

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

    fireEvent.click(
      screen.getByRole("button", { name: "Configure MCP tools" }),
    );

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
      screen.getByText(
        "Wait for the current turn to finish before regenerating.",
      ),
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
    expect(
      screen.getByRole("button", { name: "Stop generation" }),
    ).toBeDisabled();
    expect(screen.getByText("No turn is running.")).toBeInTheDocument();
  });

  it("surfaces empty assistant responses beside the regenerate control", () => {
    const props = renderInspector({
      selectedProvider: "anthropic",
      selectedModel: "claude-3-haiku",
      providerRouteLabel: "anthropic:claude-3-haiku",
      streaming: false,
      runtimeStatus: "ready",
      canStopStreaming: false,
      canRegenerate: true,
      emptyAssistantResponse: true,
      emptyAssistantResponseRouteLabel: "openai:gpt-4.1-mini",
    });

    const runControls = screen.getByRole("region", { name: "Run controls" });
    const status = within(runControls).getByRole("status", {
      name: "Empty assistant response",
    });
    expect(status).toHaveTextContent(
      "openai:gpt-4.1-mini returned no response text.",
    );
    expect(status).toHaveTextContent(
      "Regenerate this turn, or choose a different model if this provider keeps returning empty output.",
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
      screen.getByText(
        "Regenerate becomes available after an assistant response.",
      ),
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

  it("keeps streaming primary over blocked model usability", () => {
    renderInspector({
      streaming: true,
      runtimeStatus: undefined,
      modelUsabilityStatus: "provider_unconfigured",
      modelUsabilityCanSend: false,
      modelUsabilityDetail:
        "Configure the selected model provider before chatting as Ada",
    });

    expect(screen.getByText("Streaming")).toBeInTheDocument();
    expect(screen.queryByText("Error")).toBeNull();
    expect(
      screen.getByText(
        "Configure the selected model provider before chatting as Ada",
      ),
    ).toBeInTheDocument();
  });

  it("shows blocked model usability when not streaming", () => {
    renderInspector({
      runtimeStatus: undefined,
      modelUsabilityStatus: "degraded",
      modelUsabilityCanSend: false,
      modelUsabilityDetail: "Character chat is preparing",
    });

    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("Character chat is preparing")).toBeInTheDocument();
  });

  it("shows model readiness loading without treating it as an error", () => {
    renderInspector({
      runtimeStatus: "loading",
      runtimeStatusDetail: "Checking chat model readiness",
      modelUsabilityStatus: "loading",
      modelUsabilityCanSend: false,
      modelUsabilityDetail: "Checking chat model readiness",
    });

    expect(screen.getByText("Checking model")).toBeInTheDocument();
    expect(
      screen.getAllByText("Checking chat model readiness").length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("Error")).toBeNull();
    expect(screen.queryByText("Ready")).toBeNull();
  });
});
