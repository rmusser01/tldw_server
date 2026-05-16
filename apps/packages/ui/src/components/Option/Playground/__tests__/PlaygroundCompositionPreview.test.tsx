// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundCompositionPreview } from "../PlaygroundCompositionPreview";
import type { PlaygroundCompositionPreviewSummary } from "../playground-composition-preview";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback,
  }),
}));

const summary = (
  overrides: Partial<PlaygroundCompositionPreviewSummary> = {},
): PlaygroundCompositionPreviewSummary => ({
  overallState: "ready",
  settingsScopeLabel: "openai:gpt-4.1-mini",
  entries: [
    {
      id: "prompt",
      kind: "prompt",
      label: "Prompt",
      title: "Research brief",
      detail: "System prompt",
      state: "active",
    },
    {
      id: "assistant",
      kind: "assistant",
      label: "Assistant",
      title: "Research Persona",
      detail: "Persona selected - memory read/write",
      state: "active",
    },
    {
      id: "model",
      kind: "model",
      label: "Model",
      title: "openai:gpt-4.1-mini",
      detail: "openai",
      state: "active",
    },
    {
      id: "settings",
      kind: "settings",
      label: "Settings",
      title: "openai:gpt-4.1-mini",
      detail: "Temperature: 0.31",
      state: "active",
    },
    {
      id: "context",
      kind: "context",
      label: "Context",
      title: "1 active source",
      detail: "1 configured source",
      state: "active",
    },
    {
      id: "tools",
      kind: "tools",
      label: "Tools",
      title: "MCP tools",
      detail: "2 chat tools available",
      state: "active",
    },
  ],
  contextStack: [
    {
      id: "prompt",
      kind: "prompt",
      label: "Prompt",
      title: "Research brief",
      detail: "System prompt",
      state: "active",
    },
    {
      id: "assistant",
      kind: "assistant",
      label: "Assistant",
      title: "Research Persona",
      detail: "Persona selected - memory read/write",
      state: "active",
    },
    {
      id: "source-knowledge-1",
      kind: "knowledge",
      label: "Knowledge",
      title: "Launch notes",
      detail: "2 snippets",
      state: "active",
    },
    {
      id: "tools",
      kind: "tools",
      label: "Tools",
      title: "MCP tools",
      detail: "2 chat tools available",
      state: "active",
    },
  ],
  footprint: {
    providerMessageCount: 1,
    previewSectionCount: 1,
    contextPieceCount: 1,
    warningCount: 0,
    readiness: "ready",
  },
  ...overrides,
});

describe("PlaygroundCompositionPreview", () => {
  it("renders a scannable next-message composition summary", () => {
    render(<PlaygroundCompositionPreview summary={summary()} />);

    const region = screen.getByRole("region", {
      name: "Next message composition",
    });

    expect(within(region).getByText("Ready")).toBeInTheDocument();
    expect(within(region).getByText("Research brief")).toBeInTheDocument();
    expect(within(region).getByText("Research Persona")).toBeInTheDocument();
    expect(within(region).getAllByText("openai:gpt-4.1-mini")).toHaveLength(2);
    expect(within(region).getByText("MCP tools")).toBeInTheDocument();
    expect(within(region).getByText("Scope: openai:gpt-4.1-mini")).toBeInTheDocument();
  });

  it("keeps detailed context stack and footprint behind a disclosure", () => {
    render(<PlaygroundCompositionPreview summary={summary()} />);

    expect(screen.queryByText("Launch notes")).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Show composition details" }),
    );

    expect(screen.getByText("Launch notes")).toBeInTheDocument();
    expect(screen.getByText("1 provider message")).toBeInTheDocument();
    expect(screen.getByText("1 preview section")).toBeInTheDocument();
    expect(screen.getByText("1 context piece")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Hide composition details" }),
    );

    expect(screen.queryByText("Launch notes")).not.toBeInTheDocument();
  });

  it("labels degraded and unavailable rows without presenting them as ready", () => {
    render(
      <PlaygroundCompositionPreview
        summary={summary({
          overallState: "degraded",
          entries: [
            {
              id: "model",
              kind: "model",
              label: "Model",
              title: "openai:gpt-4.1-mini",
              detail: null,
              state: "active",
            },
            {
              id: "tools",
              kind: "tools",
              label: "Tools",
              title: "MCP unavailable",
              detail: "MCP tools unavailable",
              state: "unavailable",
            },
          ],
          contextStack: [
            {
              id: "tools",
              kind: "tools",
              label: "Tools",
              title: "MCP unavailable",
              detail: "MCP tools unavailable",
              state: "unavailable",
            },
          ],
        })}
      />,
    );

    expect(screen.getByText("Degraded")).toBeInTheDocument();
    expect(screen.getByText("MCP unavailable")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(screen.queryByText("Ready")).not.toBeInTheDocument();
  });
});
