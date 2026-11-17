import React from "react";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  PlaygroundContextRail,
  type PlaygroundContextRailProps,
} from "../PlaygroundContextRail";
import { PlaygroundRuntimeInspector } from "../PlaygroundRuntimeInspector";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string | { defaultValue?: string }) =>
      typeof fallback === "string"
        ? fallback
        : (fallback?.defaultValue ?? _key),
  }),
}));

const contextProps: PlaygroundContextRailProps = {
  hasContext: false,
  contextSummary: [],
  sessionLabel: "Local chat",
  historyLinked: false,
  webSearch: false,
  onToggleWebSearch: vi.fn(),
  temporaryChat: false,
  onToggleTemporaryChat: vi.fn(),
  contextCounts: { files: 0, knowledge: 0, media: 0, research: 0 },
  onOpenSearchContext: vi.fn(),
};
const runtimeProps = {
  streaming: false,
  selectedProvider: "openai",
  selectedModel: "gpt-4.1-mini",
  messageCount: 2,
  threadSearchOpen: false,
  onOpenModelSettings: vi.fn(),
};

describe("calm two-rail hierarchy", () => {
  it("puts assistant controls beside prompt controls in Context", () => {
    render(
      <PlaygroundContextRail
        {...contextProps}
        promptSelectControl={<button>Select prompt</button>}
        assistantControl={<button>Select persona</button>}
      />,
    );
    const instructions = screen.getByRole("region", { name: "Instructions" });
    expect(
      within(instructions).getByRole("button", { name: "Select persona" }),
    ).toBeVisible();
    expect(
      within(instructions).getByRole("button", { name: "Select prompt" }),
    ).toBeVisible();
  });

  it("does not repeat prompt sources when dedicated prompt controls are present", () => {
    render(
      <PlaygroundContextRail
        {...contextProps}
        promptSelectControl={<button>Select prompt</button>}
        contextSources={[
          {
            id: "prompt",
            kind: "prompt",
            label: "Prompt",
            title: "Research instructions",
          },
          { id: "file", kind: "file", label: "File", title: "Methods.pdf" },
        ]}
      />,
    );
    const sources = screen.getByRole("list", { name: "Context sources" });
    expect(within(sources).queryByText("Research instructions")).toBeNull();
    expect(within(sources).getByText("Methods.pdf")).toBeVisible();
  });

  it("keeps assistant selection out of Runtime", () => {
    render(<PlaygroundRuntimeInspector {...runtimeProps} />);
    expect(screen.queryByRole("region", { name: "Assistant" })).toBeNull();
  });

  it("keeps streaming status beside the stop action", () => {
    render(
      <PlaygroundRuntimeInspector
        {...runtimeProps}
        streaming
        canStopStreaming
        onStopStreaming={vi.fn()}
      />,
    );
    const run = screen.getByRole("region", { name: "Run controls" });
    expect(within(run).getByText("Streaming")).toBeVisible();
    expect(
      within(run).getByRole("button", { name: "Stop generation" }),
    ).toBeEnabled();
  });

  it("shows overrides while inherited generation values wait behind details", () => {
    render(
      <PlaygroundRuntimeInspector
        {...runtimeProps}
        settingSummaries={[
          { label: "Temperature", value: "0.7", source: "default" },
          { label: "Context", value: "8192", source: "override" },
        ]}
      />,
    );
    expect(screen.getByText("8192")).toBeVisible();
    expect(screen.getByText("0.7")).not.toBeVisible();
  });
});
