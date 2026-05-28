// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundContextRail } from "../PlaygroundContextRail";
import type { PlaygroundCompositionPreviewSummary } from "../playground-composition-preview";

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

const compositionSummary = (): PlaygroundCompositionPreviewSummary => ({
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
      id: "model",
      kind: "model",
      label: "Model",
      title: "openai:gpt-4.1-mini",
      detail: "openai",
      state: "active",
    },
    {
      id: "tools",
      kind: "tools",
      label: "MCP tools",
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
  ],
  footprint: {
    providerMessageCount: 0,
    previewSectionCount: 0,
    contextPieceCount: 0,
    warningCount: 0,
    readiness: "unavailable",
  },
});

const noPromptCompositionSummary = (): PlaygroundCompositionPreviewSummary => ({
  ...compositionSummary(),
  entries: [
    {
      id: "prompt",
      kind: "prompt",
      label: "Prompt",
      title: "No prompt selected",
      detail: "No prompt context will be added.",
      state: "disabled",
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
      id: "tools",
      kind: "tools",
      label: "MCP tools",
      title: "MCP tools",
      detail: "2 chat tools available",
      state: "active",
    },
  ],
  contextStack: [],
});

describe("PlaygroundContextRail first-slice controls", () => {
  it("keeps left rail groups in cockpit comprehension order", () => {
    renderRail({
      compositionPreviewSummary: compositionSummary(),
      hasContext: true,
      contextSources: [
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
          title: "Mira Vale",
          detail: "Character selected",
          state: "active",
        },
      ],
      promptSummary: {
        state: "system",
        label: "Research brief",
        detail: "System prompt selected",
      },
      contextCounts: {
        files: 1,
        knowledge: 1,
        media: 1,
        research: 1,
      },
    });

    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual([
      "Composition",
      "Context stack",
      "Prompt",
      "Search & sources",
      "Session",
    ]);
  });

  it("places the optional character sessions panel before the generic session controls", () => {
    renderRail({
      compositionPreviewSummary: compositionSummary(),
      characterSessionsPanel: (
        <section>
          <h2>Character sessions</h2>
          <p>Recent character chats</p>
        </section>
      ),
    });

    expect(
      screen
        .getAllByRole("heading", { level: 2 })
        .map((heading) => heading.textContent),
    ).toEqual([
      "Composition",
      "Context stack",
      "Prompt",
      "Search & sources",
      "Character sessions",
      "Session",
    ]);
  });

  it("keeps first-time rail controls discoverable when no extra context is active", () => {
    renderRail({
      compositionPreviewSummary: compositionSummary(),
      hasContext: false,
      contextSummary: [],
      promptSummary: {
        state: "none",
        label: "No prompt selected",
        detail: "No prompt context will be added.",
      },
      promptSelectControl: (
        <button type="button" aria-label="Select a prompt">
          Select prompt
        </button>
      ),
    });

    expect(
      screen.getByRole("region", { name: "Next message composition" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Prompt" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Select a prompt" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Search & sources" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open Search & Context" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Web search" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Session" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Use temporary chat" }),
    ).toBeInTheDocument();
  });

  it("lets each left rail section collapse without removing the rail", () => {
    renderRail({
      compositionPreviewSummary: compositionSummary(),
      hasContext: false,
      contextSummary: [],
      promptSelectControl: (
        <button type="button" aria-label="Select a prompt">
          Select prompt
        </button>
      ),
    });

    const rail = screen.getByTestId("playground-context-rail");
    const collapseContext = within(rail).getByRole("button", {
      name: "Collapse Context stack",
    });
    const contextPanelId = collapseContext.getAttribute("aria-controls");
    const contextPanel = contextPanelId
      ? document.getElementById(contextPanelId)
      : null;

    expect(collapseContext).toHaveAttribute("aria-expanded", "true");
    expect(contextPanel).not.toBeNull();
    expect(contextPanel).toHaveAttribute("aria-hidden", "false");
    expect(contextPanel).not.toHaveClass("hidden");

    fireEvent.click(collapseContext);

    expect(rail).toBeInTheDocument();
    expect(collapseContext).toHaveAttribute("aria-expanded", "false");
    expect(contextPanel).toHaveAttribute("aria-hidden", "true");
    expect(contextPanel).toHaveClass("hidden");
    expect(
      within(rail).getByRole("button", { name: "Expand Context stack" }),
    ).toBeInTheDocument();

    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse Composition" }),
    );
    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse Prompt" }),
    );
    fireEvent.click(
      within(rail).getByRole("button", {
        name: "Collapse Search & sources",
      }),
    );
    fireEvent.click(
      within(rail).getByRole("button", { name: "Collapse Session" }),
    );

    expect(
      screen.getByRole("heading", { name: "Composition" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Context stack" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Prompt" })).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Search & sources" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "Session" }),
    ).toBeInTheDocument();
  });

  it("starts secondary context sections collapsed in setup recovery mode", () => {
    renderRail({
      setupRecoveryMode: true,
      compositionPreviewSummary: compositionSummary(),
      promptSelectControl: (
        <button type="button" aria-label="Select a prompt">
          Select prompt
        </button>
      ),
    });

    const rail = screen.getByTestId("playground-context-rail");
    expect(
      within(rail).getByRole("button", { name: "Collapse Context stack" }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(
      within(rail).getByRole("button", { name: "Expand Prompt" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      within(rail).getByRole("button", { name: "Expand Search & sources" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      within(rail).getByRole("button", { name: "Expand Session" }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      within(rail).queryByRole("button", { name: "Select a prompt" }),
    ).toBeNull();

    fireEvent.click(within(rail).getByRole("button", { name: "Expand Prompt" }));

    expect(
      within(rail).getByRole("button", { name: "Select a prompt" }),
    ).toBeInTheDocument();
  });

  it("preserves existing left rail actions after regrouping", () => {
    const props = renderRail({
      compositionPreviewSummary: compositionSummary(),
      hasContext: true,
      webSearch: false,
      promptSummary: {
        state: "system",
        label: "Research brief",
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
        media: 1,
        research: 1,
      },
    });

    fireEvent.click(screen.getByRole("button", { name: "Clear prompt" }));
    fireEvent.click(screen.getByRole("button", { name: "Web search" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Open Search & Context" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Clear files" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear knowledge" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear media scopes" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Clear research context" }),
    );

    expect(props.onClearPrompt).toHaveBeenCalledTimes(1);
    expect(props.onToggleWebSearch).toHaveBeenCalledTimes(1);
    expect(props.onOpenSearchContext).toHaveBeenCalledTimes(1);
    expect(props.onClearFiles).toHaveBeenCalledTimes(1);
    expect(props.onClearKnowledge).toHaveBeenCalledTimes(1);
    expect(props.onClearMedia).toHaveBeenCalledTimes(1);
    expect(props.onClearResearch).toHaveBeenCalledTimes(1);
  });

  it("renders the next-message composition preview when supplied", () => {
    renderRail({
      compositionPreviewSummary: compositionSummary(),
    });

    const preview = screen.getByRole("region", {
      name: "Next message composition",
    });

    expect(preview).toHaveTextContent("Research brief");
    expect(preview).toHaveTextContent("openai:gpt-4.1-mini");
    expect(preview).toHaveTextContent("MCP tools");
  });

  it("does not repeat the no-prompt empty state in prompt management", () => {
    renderRail({
      compositionPreviewSummary: noPromptCompositionSummary(),
      promptSummary: {
        state: "none",
        label: "No prompt selected",
        detail: "No prompt context will be added.",
      },
      promptSelectControl: (
        <button type="button" aria-label="Select a prompt">
          Select prompt
        </button>
      ),
    });

    expect(
      screen.getByRole("region", { name: "Next message composition" }),
    ).toHaveTextContent("No prompt selected");

    const promptManagement = screen.getByRole("region", {
      name: "Prompt management",
    });
    expect(
      within(promptManagement).queryByText("No prompt selected"),
    ).toBeNull();
    expect(
      within(promptManagement).getByText("Ready to add prompt"),
    ).toBeInTheDocument();
    expect(
      within(promptManagement).getByText(
        "Select a prompt to add system instructions.",
      ),
    ).toBeInTheDocument();
  });

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

    expect(screen.getByRole("heading", { name: "Prompt" })).toBeInTheDocument();
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
    expect(screen.getAllByText("Loading conversation").length).toBeGreaterThan(
      0,
    );
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
    expect(
      screen.getByText("Conversation no longer exists"),
    ).toBeInTheDocument();
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
    fireEvent.click(
      screen.getByRole("button", { name: "Clear research context" }),
    );

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
