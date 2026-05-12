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
    onOpenSearchContext: vi.fn(),
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

  it("switches between saved and temporary session modes through the supplied callback", () => {
    const props = renderRail({ temporaryChat: false });

    fireEvent.click(screen.getByRole("button", { name: "Use temporary chat" }));

    expect(props.onToggleTemporaryChat).toHaveBeenCalledWith(true);
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

  it("keeps Search & Context available in the empty state", () => {
    renderRail({ hasContext: false, contextSummary: [] });

    expect(screen.getByText("No extra context")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open Search & Context" }),
    ).toBeInTheDocument();
  });
});
