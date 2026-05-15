// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundContextRail } from "../PlaygroundContextRail";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback,
  }),
}));

const renderRail = (
  overrides: Partial<React.ComponentProps<typeof PlaygroundContextRail>> = {},
) => {
  const props = {
    hasContext: false,
    contextSummary: [],
    sessionLabel: "Local chat",
    historyLinked: false,
    webSearch: false,
    onToggleWebSearch: vi.fn(),
    temporaryChat: false,
    onToggleTemporaryChat: vi.fn(),
    contextCounts: {
      files: 0,
      knowledge: 0,
      media: 0,
      research: 0,
    },
    promptSummary: {
      state: "none" as const,
      label: "No prompt selected",
      detail: "No prompt context will be added.",
    },
    promptSelectControl: null,
    onClearPrompt: vi.fn(),
    onOpenSearchContext: vi.fn(),
    onClearFiles: vi.fn(),
    onClearKnowledge: vi.fn(),
    onClearMedia: vi.fn(),
    onClearResearch: vi.fn(),
    ...overrides,
  };

  render(<PlaygroundContextRail {...props} />);

  return props;
};

describe("PlaygroundContextRail first-slice controls", () => {
  it("exposes web search as a pressed-state control", () => {
    const props = renderRail();

    const toggle = screen.getByRole("button", { name: "Web search" });
    expect(toggle).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(toggle);

    expect(props.onToggleWebSearch).toHaveBeenCalledTimes(1);
  });

  it("reflects enabled web search state", () => {
    renderRail({ webSearch: true });

    expect(screen.getByRole("button", { name: "Web search" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("opens Search & Context through the supplied callback", () => {
    const props = renderRail();

    fireEvent.click(
      screen.getByRole("button", { name: "Open Search & Context" }),
    );

    expect(props.onOpenSearchContext).toHaveBeenCalledTimes(1);
  });

  it("renders prompt state as first-class context and clears only prompt context", () => {
    const props = renderRail({
      hasContext: true,
      promptSummary: {
        state: "system",
        label: "Socratic tutor",
        detail: "System prompt selected",
      },
      promptSelectControl: (
        <button type="button" aria-label="Select a prompt">
          Select prompt
        </button>
      ),
      contextCounts: {
        files: 1,
        knowledge: 1,
        media: 0,
        research: 0,
      },
    });

    expect(screen.getByText("Prompts")).toBeInTheDocument();
    expect(screen.getByText("Socratic tutor")).toBeInTheDocument();
    expect(screen.getByText("System prompt selected")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Select a prompt" }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Clear prompt" }));

    expect(props.onClearPrompt).toHaveBeenCalledTimes(1);
    expect(props.onClearFiles).not.toHaveBeenCalled();
    expect(props.onClearKnowledge).not.toHaveBeenCalled();
  });

  it("identifies inline custom prompt context without requiring a template", () => {
    renderRail({
      hasContext: true,
      promptSummary: {
        state: "custom",
        label: "Custom prompt",
        detail: "Inline system prompt active",
      },
    });

    expect(screen.getByText("Custom prompt")).toBeInTheDocument();
    expect(screen.getByText("Inline system prompt active")).toBeInTheDocument();
  });

  it("switches between saved and temporary session modes through the supplied callback", () => {
    const props = renderRail({ temporaryChat: false });

    fireEvent.click(screen.getByRole("button", { name: "Use temporary chat" }));

    expect(props.onToggleTemporaryChat).toHaveBeenCalledWith(true);
  });

  it("surfaces server-session loading state without hiding history state", () => {
    renderRail({
      sessionLabel: "Server chat",
      historyLinked: true,
      sessionTitle: "Research follow-up",
      sessionStatus: "loading",
      sessionDetail: "Loading conversation",
    } as any);

    expect(screen.getByText("Server chat")).toBeInTheDocument();
    expect(screen.getByText("Research follow-up")).toBeInTheDocument();
    expect(screen.getAllByText("Loading conversation").length).toBeGreaterThan(0);
    expect(screen.getByText("History linked")).toBeInTheDocument();
  });

  it("surfaces recoverable server-session errors", () => {
    renderRail({
      sessionLabel: "Server chat",
      historyLinked: false,
      sessionTitle: "Archived investigation",
      sessionStatus: "failed",
      sessionDetail: "Failed to load conversation",
      sessionError: "Conversation no longer exists",
    } as any);

    expect(screen.getByText("Load failed")).toBeInTheDocument();
    expect(screen.getByText("Archived investigation")).toBeInTheDocument();
    expect(screen.getByText("Failed to load conversation")).toBeInTheDocument();
    expect(screen.getByText("Conversation no longer exists")).toBeInTheDocument();
    expect(screen.getByText("No saved history yet")).toBeInTheDocument();
  });

  it("shows explicit context counts even when no summary strings are supplied", () => {
    renderRail({
      hasContext: true,
      contextCounts: {
        files: 2,
        knowledge: 1,
        media: 3,
        research: 1,
      },
    });

    expect(screen.getByText("2 files")).toBeInTheDocument();
    expect(screen.getByText("1 knowledge item")).toBeInTheDocument();
    expect(screen.getByText("3 media scopes")).toBeInTheDocument();
    expect(screen.getByText("1 research attachment")).toBeInTheDocument();
  });

  it("clears active context groups through supplied callbacks", () => {
    const props = renderRail({
      hasContext: true,
      contextCounts: {
        files: 2,
        knowledge: 1,
        media: 3,
        research: 1,
      },
    });

    fireEvent.click(screen.getByRole("button", { name: "Clear files" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear knowledge" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear media scopes" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear research context" }));

    expect(props.onClearFiles).toHaveBeenCalledTimes(1);
    expect(props.onClearKnowledge).toHaveBeenCalledTimes(1);
    expect(props.onClearMedia).toHaveBeenCalledTimes(1);
    expect(props.onClearResearch).toHaveBeenCalledTimes(1);
  });

  it("keeps Search & Context available in the empty state", () => {
    renderRail({ hasContext: false, contextSummary: [] });

    expect(screen.getByText("No extra context")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open Search & Context" }),
    ).toBeInTheDocument();
  });
});
