// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import {
  PlaygroundCockpitShell,
  type PlaygroundCockpitMobilePanel,
} from "../PlaygroundCockpitShell";
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

function ControlledMobileShell({
  initialPanel = null,
  leftRailVisible = true,
  rightRailVisible = true,
  onLeftRailVisibleChange = () => {},
  onRightRailVisibleChange = () => {},
}: {
  initialPanel?: PlaygroundCockpitMobilePanel;
  leftRailVisible?: boolean;
  rightRailVisible?: boolean;
  onLeftRailVisibleChange?: (visible: boolean) => void;
  onRightRailVisibleChange?: (visible: boolean) => void;
}) {
  const [leftVisible, setLeftVisible] = React.useState(leftRailVisible);
  const [rightVisible, setRightVisible] = React.useState(rightRailVisible);
  const [panel, setPanel] = React.useState(initialPanel);

  return (
    <PlaygroundCockpitShell
      mode="cockpit"
      leftRailVisible={leftVisible}
      rightRailVisible={rightVisible}
      onLeftRailVisibleChange={(visible) => {
        onLeftRailVisibleChange(visible);
        setLeftVisible(visible);
      }}
      onRightRailVisibleChange={(visible) => {
        onRightRailVisibleChange(visible);
        setRightVisible(visible);
      }}
      mobilePanel={panel}
      onMobilePanelChange={setPanel}
      leftRail={<div>Context tools</div>}
      rightRail={<div>Runtime tools</div>}
    >
      <textarea aria-label="Composer draft" defaultValue="Keep this draft" />
    </PlaygroundCockpitShell>
  );
}

describe("Playground cockpit accessibility", () => {
  it.each(["cockpit", "focus"] as const)(
    "keeps one page main when the %s shell is inside its page layout",
    (mode) => {
      render(
        <main aria-label="Chat page">
          <PlaygroundCockpitShell
            mode={mode}
            leftRail={<div>Context tools</div>}
            rightRail={<div>Runtime tools</div>}
          >
            <div>Chat transcript</div>
          </PlaygroundCockpitShell>
        </main>,
      );

      expect(screen.getAllByRole("main")).toHaveLength(1);
      expect(
        within(screen.getByRole("main", { name: "Chat page" })).getByText(
          "Chat transcript",
        ),
      ).toBeInTheDocument();
    },
  );

  it("offers one pair of native mobile disclosure buttons with labeled panel relations", () => {
    render(<ControlledMobileShell />);
    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    const rails = within(mobileRails);

    expect(mobileRails.querySelectorAll("button")).toHaveLength(2);
    expect(rails.queryByRole("tablist")).not.toBeInTheDocument();
    expect(rails.queryAllByRole("tab")).toHaveLength(0);
    expect(rails.getAllByRole("region", { hidden: true })).toHaveLength(2);
    for (const name of ["Context", "Chat settings"]) {
      const button = rails.getByRole("button", { name });
      const panel = document.getElementById(
        button.getAttribute("aria-controls") ?? "",
      );
      expect(button).toHaveAttribute("type", "button");
      expect(button).toHaveAttribute("aria-expanded", "false");
      expect(button).not.toHaveAttribute("aria-selected");
      expect(panel).toHaveAttribute("aria-labelledby", button.id);
      expect(panel).not.toBeVisible();
    }
  });

  it("honors an explicitly closed mobile panel while both desktop rails stay enabled", () => {
    render(<ControlledMobileShell />);
    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    const rails = within(mobileRails);

    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
    expect(rails.queryAllByRole("region")).toHaveLength(0);
    expect(rails.getByRole("button", { name: "Context" })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(rails.getByRole("button", { name: "Chat settings" })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit context" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit runtime" }),
    ).toBeInTheDocument();
  });

  it.each([
    ["Enter", "{Enter}"],
    ["Space", " "],
  ])(
    "opens, switches, and closes mobile panels with %s without changing desktop preferences",
    async (_name, key) => {
      const user = userEvent.setup();
      const onLeftRailVisibleChange = vi.fn();
      const onRightRailVisibleChange = vi.fn();
      render(
        <ControlledMobileShell
          onLeftRailVisibleChange={onLeftRailVisibleChange}
          onRightRailVisibleChange={onRightRailVisibleChange}
        />,
      );
      const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
      const rails = within(mobileRails);
      const context = rails.getByRole("button", { name: "Context" });
      const runtime = rails.getByRole("button", { name: "Chat settings" });
      const draft = screen.getByRole("textbox", { name: "Composer draft" });

      context.focus();
      await user.keyboard(key);
      expect(mobileRails).toHaveAttribute("data-mobile-panel", "context");
      expect(rails.getByRole("region", { name: "Context" })).toBeVisible();
      expect(context).toHaveAttribute("aria-expanded", "true");
      expect(runtime).toHaveAttribute("aria-expanded", "false");
      expect(context).toHaveFocus();

      runtime.focus();
      await user.keyboard(key);
      expect(mobileRails).toHaveAttribute("data-mobile-panel", "runtime");
      expect(
        rails.queryByRole("region", { name: "Context" }),
      ).not.toBeInTheDocument();
      expect(rails.getByRole("region", { name: "Chat settings" })).toBeVisible();
      expect(context).toHaveAttribute("aria-expanded", "false");
      expect(runtime).toHaveAttribute("aria-expanded", "true");

      await user.keyboard(key);
      expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
      expect(rails.queryAllByRole("region")).toHaveLength(0);
      expect(context).toHaveAttribute("aria-expanded", "false");
      expect(runtime).toHaveAttribute("aria-expanded", "false");
      expect(runtime).toHaveFocus();
      expect(onLeftRailVisibleChange).not.toHaveBeenCalled();
      expect(onRightRailVisibleChange).not.toHaveBeenCalled();
      expect(draft).toBe(
        screen.getByRole("textbox", { name: "Composer draft" }),
      );
      expect(draft).toHaveValue("Keep this draft");
    },
  );

  it("restores a globally collapsed rail on open and keeps it restored on mobile close", async () => {
    const user = userEvent.setup();
    const onLeftRailVisibleChange = vi.fn();
    const onRightRailVisibleChange = vi.fn();
    render(
      <ControlledMobileShell
        leftRailVisible={false}
        rightRailVisible={false}
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
      />,
    );
    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    const rails = within(mobileRails);
    const context = rails.getByRole("button", { name: "Context" });
    const runtime = rails.getByRole("button", { name: "Chat settings" });

    context.focus();
    await user.keyboard("{Enter}");
    expect(rails.getByRole("region", { name: "Context" })).toBeVisible();
    expect(onLeftRailVisibleChange).toHaveBeenCalledExactlyOnceWith(true);

    await user.keyboard(" ");
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
    expect(onLeftRailVisibleChange).toHaveBeenCalledExactlyOnceWith(true);
    expect(
      screen.getByRole("complementary", { name: "Chat cockpit context" }),
    ).toBeInTheDocument();

    runtime.focus();
    await user.keyboard("{Enter}");
    expect(rails.getByRole("region", { name: "Chat settings" })).toBeVisible();
    expect(onRightRailVisibleChange).toHaveBeenCalledExactlyOnceWith(true);
    expect(onLeftRailVisibleChange).toHaveBeenCalledExactlyOnceWith(true);
  });

  it("falls back to the available mobile rail when the selected rail is globally hidden", () => {
    const { rerender } = render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible={false}
        rightRailVisible
        mobilePanel="context"
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );
    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    const rails = within(mobileRails);
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "runtime");
    expect(rails.getByRole("button", { name: "Chat settings" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(rails.getByRole("region", { name: "Chat settings" })).toBeVisible();

    rerender(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible={false}
        rightRailVisible={false}
        mobilePanel="context"
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
    expect(rails.queryAllByRole("region")).toHaveLength(0);
  });

  it("toggles the mobile panel with the uncontrolled shell API", async () => {
    const user = userEvent.setup();
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );
    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    const rails = within(mobileRails);
    const context = rails.getByRole("button", { name: "Context" });
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "context");

    await user.click(context);
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");
    await user.click(rails.getByRole("button", { name: "Chat settings" }));
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "runtime");
    expect(rails.getByRole("region", { name: "Chat settings" })).toBeVisible();
  });

  it("labels cockpit landmarks without adding a top layout rail", () => {
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        onLeftRailVisibleChange={vi.fn()}
        onRightRailVisibleChange={vi.fn()}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
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
    ).toHaveTextContent("Chat settings");

    expect(
      screen.queryByRole("button", { name: "Enter focus chat" }),
    ).not.toBeInTheDocument();
  });

  it("keeps the mobile cockpit panel compact while preserving its accessible summary", () => {
    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        onLeftRailVisibleChange={vi.fn()}
        onRightRailVisibleChange={vi.fn()}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "context");

    const contextPanel = screen.getByRole("region", { name: "Context" });
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
        leftRailVisible={false}
        rightRailVisible
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextToggle = screen.getByRole("button", {
      name: "Show context",
    });
    const runtimeToggle = within(
      screen.getByRole("complementary", { name: "Chat cockpit runtime" }),
    ).getByRole("button", {
      name: "Hide chat settings",
    });

    expect(contextToggle).toHaveAttribute(
      "aria-controls",
      "playground-cockpit-left-rail",
    );
    expect(runtimeToggle).toHaveAttribute(
      "aria-controls",
      "playground-cockpit-right-rail",
    );
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
        leftRailVisible
        rightRailVisible={false}
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextRail = screen.getByRole("complementary", {
      name: "Chat cockpit context",
    });
    const contextCollapse = within(contextRail).getByRole("button", {
      name: "Hide context",
    });
    fireEvent.click(contextCollapse);
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(false);

    const runtimeRestore = screen.getByRole("button", {
      name: "Show chat settings",
    });
    expect(runtimeRestore).toHaveAttribute(
      "aria-controls",
      "playground-cockpit-right-rail",
    );
    fireEvent.click(runtimeRestore);
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(true);
  });

  it("activates desktop rail controls and mobile disclosures from the keyboard", async () => {
    const user = userEvent.setup();
    const onLeftRailVisibleChange = vi.fn();
    const onRightRailVisibleChange = vi.fn();
    const onMobilePanelChange = vi.fn();

    render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        onLeftRailVisibleChange={onLeftRailVisibleChange}
        onRightRailVisibleChange={onRightRailVisibleChange}
        mobilePanel="context"
        onMobilePanelChange={onMobilePanelChange}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextToggle = within(
      screen.getByRole("complementary", { name: "Chat cockpit context" }),
    ).getByRole("button", { name: "Hide context" });
    contextToggle.focus();
    await user.keyboard("{Enter}");
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(false);

    const runtimeToggle = within(
      screen.getByRole("complementary", { name: "Chat cockpit runtime" }),
    ).getByRole("button", { name: "Hide chat settings" });
    runtimeToggle.focus();
    await user.keyboard("{Enter}");
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(false);

    const mobileRails = within(
      screen.getByTestId("playground-cockpit-mobile-rails"),
    );
    const runtimeButton = mobileRails.getByRole("button", { name: "Chat settings" });
    const contextButton = mobileRails.getByRole("button", { name: "Context" });
    for (const button of [contextButton, runtimeButton]) {
      const controlledPanel = document.getElementById(
        button.getAttribute("aria-controls") ?? "",
      );

      expect(controlledPanel).not.toBeNull();
      expect(controlledPanel).not.toHaveAttribute("role", "tabpanel");
      expect(controlledPanel).toHaveAttribute("aria-labelledby", button.id);
    }

    runtimeButton.focus();
    await user.keyboard("{Enter}");
    expect(onMobilePanelChange).toHaveBeenCalledWith("runtime");

    expect(
      within(screen.getByTestId("playground-cockpit-mobile-rails")).queryByRole(
        "button",
        { name: "Return to focus chat" },
      ),
    ).not.toBeInTheDocument();
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
    expect(status).toHaveTextContent("claude-sonnet-4 (Anthropic)");
    expect(status).not.toHaveTextContent("anthropic:claude-sonnet-4");
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

    // CC-06: a labelled two-way control that always shows both choices.
    expect(screen.getByRole("radio", { name: "Ready to use" })).toBeChecked();
    const allModels = screen.getByRole("radio", { name: "All models" });
    expect(allModels).not.toBeChecked();
    fireEvent.click(allModels);
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

    expect(screen.getByRole("radio", { name: "All models" })).toBeChecked();
    expect(screen.getByLabelText("Search models")).toHaveAttribute(
      "placeholder",
      "Search all models",
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
    expect(screen.queryAllByText("No assistant selected")).toHaveLength(0);
  });
});
