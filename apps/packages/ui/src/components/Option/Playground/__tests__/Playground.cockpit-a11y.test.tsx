// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell";
import { PlaygroundCompositionPreview } from "../PlaygroundCompositionPreview";
import { PlaygroundModelCatalogControls } from "../PlaygroundModelCatalogControls";
import { PlaygroundRuntimeInspector } from "../PlaygroundRuntimeInspector";
import { PlaygroundStatusStrip } from "../PlaygroundStatusStrip";
import { buildPlaygroundCompositionPreviewSummary } from "../playground-composition-preview";

const translate = vi.hoisted(
  () =>
    (
      key: string,
      fallbackOrOptions?: unknown,
      maybeOptions?: Record<string, unknown>,
    ) => {
      let template = key;
      let options: Record<string, unknown> | undefined;

      if (typeof fallbackOrOptions === "string") {
        template = fallbackOrOptions;
        options = maybeOptions;
      } else if (
        fallbackOrOptions &&
        typeof fallbackOrOptions === "object" &&
        "defaultValue" in (fallbackOrOptions as Record<string, unknown>)
      ) {
        template = String(
          (fallbackOrOptions as { defaultValue?: unknown }).defaultValue ?? key,
        );
        options = fallbackOrOptions as Record<string, unknown>;
      }

      if (!options) return template;
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options?.[token];
        return value == null ? "" : String(value);
      });
    },
);

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: translate }),
}));

describe("Playground cockpit accessibility", () => {
  it("labels cockpit landmarks and exposes the layout toggle state", () => {
    const onModeChange = vi.fn();

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={onModeChange}
        leftRailVisible
        rightRailVisible
        onLeftRailVisibleChange={vi.fn()}
        onRightRailVisibleChange={vi.fn()}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
        statusStrip={
          <PlaygroundStatusStrip
            mode="cockpit"
            streaming={false}
            selectedModel="openai:gpt-4o"
            messageCount={0}
            sessionLabel="Local chat"
            hasContext={false}
          />
        }
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    expect(
      screen.getByRole("complementary", { name: "Chat cockpit context" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit runtime" }),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-mobile-rails"),
    ).toHaveTextContent("Context");
    expect(
      screen.getByTestId("playground-cockpit-mobile-rails"),
    ).toHaveTextContent("Runtime");

    const toggle = screen.getByRole("button", {
      name: "Enter focus chat",
    });
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(toggle);
    expect(onModeChange).toHaveBeenCalledWith("focus");
  });

  it("keeps the mobile cockpit panel compact while preserving its accessible summary", () => {
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={vi.fn()}
        leftRailVisible
        rightRailVisible
        onLeftRailVisibleChange={vi.fn()}
        onRightRailVisibleChange={vi.fn()}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
        statusStrip={<div>Ready</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "context");

    const contextPanel = screen.getByRole("tabpanel", { name: "Context" });
    expect(contextPanel.className).toContain("max-h-[30vh]");
    expect(contextPanel.className).not.toContain("max-h-[42vh]");

    const summary = screen.getByTestId(
      "playground-cockpit-mobile-panel-summary",
    );
    expect(summary).toHaveClass("sr-only");
    expect(summary).toHaveTextContent(
      "Context panel active. Composer draft remains available below.",
    );
  });

  it("labels independent rail visibility controls", () => {
    const onLeftRailVisibleChange = vi.fn();
    const onRightRailVisibleChange = vi.fn();

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={vi.fn()}
        leftRailVisible={false}
        rightRailVisible
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
        statusStrip={<div>Ready</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextToggle = screen.getByRole("button", {
      name: "Show context rail",
    });
    const runtimeToggle = screen.getByRole("button", {
      name: "Hide runtime rail",
    });

    expect(contextToggle).toHaveAttribute("aria-pressed", "false");
    expect(runtimeToggle).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(contextToggle);
    fireEvent.click(runtimeToggle);
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(true);
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(false);
  });

  it("labels rail-local sidechannel collapse and edge restore controls", () => {
    const onLeftRailVisibleChange = vi.fn();
    const onRightRailVisibleChange = vi.fn();

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={vi.fn()}
        leftRailVisible
        rightRailVisible={false}
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
        statusStrip={<div>Ready</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextRail = screen.getByRole("complementary", {
      name: "Chat cockpit context",
    });
    const contextCollapse = within(contextRail).getByRole("button", {
      name: "Collapse context sidechannel",
    });
    fireEvent.click(contextCollapse);
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(false);

    const runtimeRestore = screen.getByRole("button", {
      name: "Restore runtime sidechannel",
    });
    expect(runtimeRestore).toHaveAttribute(
      "aria-controls",
      "playground-cockpit-right-rail",
    );
    fireEvent.click(runtimeRestore);
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(true);
  });

  it("activates cockpit rail and mobile tab controls from the keyboard", async () => {
    const user = userEvent.setup();
    const onModeChange = vi.fn();
    const onLeftRailVisibleChange = vi.fn();
    const onRightRailVisibleChange = vi.fn();
    const onMobilePanelChange = vi.fn();

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        onModeChange={onModeChange}
        leftRailVisible
        rightRailVisible
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        mobilePanel="context"
        onMobilePanelChange={onMobilePanelChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
        statusStrip={<div>Ready</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextToggle = screen.getByRole("button", {
      name: "Hide context rail",
    });
    contextToggle.focus();
    await user.keyboard("{Enter}");
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(false);

    const runtimeToggle = screen.getByRole("button", {
      name: "Hide runtime rail",
    });
    runtimeToggle.focus();
    await user.keyboard("{Enter}");
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(false);

    const focusToggle = screen.getByRole("button", {
      name: "Enter focus chat",
    });
    focusToggle.focus();
    await user.keyboard("{Enter}");
    expect(onModeChange).toHaveBeenCalledWith("focus");

    const runtimeTab = screen.getByRole("tab", { name: "Runtime" });
    const contextTab = screen.getByRole("tab", { name: "Context" });
    for (const tab of [contextTab, runtimeTab]) {
      const controlledPanel = document.getElementById(
        tab.getAttribute("aria-controls") ?? "",
      );

      expect(controlledPanel).not.toBeNull();
      expect(controlledPanel).toHaveAttribute("role", "tabpanel");
      expect(controlledPanel).toHaveAttribute("aria-labelledby", tab.id);
    }

    runtimeTab.focus();
    await user.keyboard("{Enter}");
    expect(onMobilePanelChange).toHaveBeenCalledWith("runtime");

    const mobilePanels = screen.getByTestId("playground-cockpit-mobile-rails");
    const panelFocusToggle = within(mobilePanels).getByRole("button", {
      name: "Return to focus chat",
    });
    panelFocusToggle.focus();
    await user.keyboard("{Enter}");
    expect(onModeChange).toHaveBeenCalledWith("focus");
  });

  it("announces compact runtime state through one status region", () => {
    render(
      <PlaygroundStatusStrip
        mode="focus"
        streaming
        selectedModel="anthropic:claude-sonnet-4"
        messageCount={2}
        sessionLabel="Server chat"
        hasContext
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status).toHaveAttribute("aria-atomic", "false");
    expect(status).toHaveTextContent("Streaming");
    expect(status).toHaveTextContent("anthropic:claude-sonnet-4");
    expect(status).toHaveTextContent("2 messages");
    expect(status).not.toHaveTextContent("Focus");
    expect(status).not.toHaveTextContent("Context active");
  });

  it("keeps model catalog controls labeled in the rendered DOM", () => {
    const setModelListScope = vi.fn();
    const setModelSearchQuery = vi.fn();
    const setModelSortMode = vi.fn();
    const { rerender } = render(
      <PlaygroundModelCatalogControls
        t={translate}
        modelListScope="configured"
        setModelListScope={setModelListScope}
        modelSearchQuery=""
        setModelSearchQuery={setModelSearchQuery}
        modelSortMode="provider"
        setModelSortMode={setModelSortMode}
      />,
    );

    const toggle = screen.getByTestId("model-list-scope-toggle");
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    expect(toggle).toHaveTextContent("Search all models");
    fireEvent.click(toggle);
    expect(setModelListScope).toHaveBeenCalledWith("catalog");

    const configuredSearch = screen.getByLabelText("Search models");
    expect(configuredSearch).toHaveAttribute("placeholder", "Search models");

    rerender(
      <PlaygroundModelCatalogControls
        t={translate}
        modelListScope="catalog"
        setModelListScope={setModelListScope}
        modelSearchQuery=""
        setModelSearchQuery={setModelSearchQuery}
        modelSortMode="provider"
        setModelSortMode={setModelSortMode}
      />,
    );

    expect(screen.getByTestId("model-list-scope-toggle")).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByLabelText("Search models")).toHaveAttribute(
      "placeholder",
      "Search all known models",
    );
  });

  it("qualifies empty assistant copy inside composition and runtime rail summaries", () => {
    const assistantSummary = {
      mode: "none" as const,
      name: null,
      detail: "No assistant selected",
    };
    const compositionSummary = buildPlaygroundCompositionPreviewSummary({
      promptSummary: {
        state: "none",
        label: "No prompt selected",
        detail: "No system prompt will be added.",
      },
      assistantSummary,
      providerRoute: {
        selectedProvider: "openai",
        selectedModel: "gpt-4o-mini",
        providerRouteLabel: "openai:gpt-4o-mini",
      },
      settingSummaries: [],
      contextSources: [],
      toolSummary: null,
      compositionStatus: "idle",
      composition: null,
    });

    render(
      <>
        <PlaygroundCompositionPreview summary={compositionSummary} />
        <PlaygroundRuntimeInspector
          streaming={false}
          selectedProvider="openai"
          selectedModel="gpt-4o-mini"
          messageCount={0}
          threadSearchOpen={false}
          assistantSummary={assistantSummary}
          onOpenModelSettings={vi.fn()}
          onOpenAssistantSelect={vi.fn()}
        />
      </>,
    );

    const composition = screen.getByRole("region", {
      name: "Next message composition",
    });
    expect(
      within(composition).getByText("No assistant attached to next message"),
    ).toBeInTheDocument();
    expect(
      within(composition).queryByText("No assistant selected"),
    ).not.toBeInTheDocument();

    const runtimeInspector = screen.getByTestId("playground-runtime-inspector");
    expect(
      within(runtimeInspector).getByText("No runtime assistant selected"),
    ).toBeInTheDocument();
    expect(
      within(runtimeInspector).queryByText("No assistant selected"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryAllByText("No assistant selected"),
    ).toHaveLength(0);
  });
});
