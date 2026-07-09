// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { COCKPIT_LEFT_RESTORE_WRAPPER_CLASS } from "@/components/Layouts/chat-rail-positioning";
import { PlaygroundCockpitShell } from "../PlaygroundCockpitShell";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

const renderShell = ({
  leftRailVisible,
  rightRailVisible,
  onLeftRailVisibleChange = vi.fn(),
  onRightRailVisibleChange = vi.fn(),
}: {
  leftRailVisible: boolean;
  rightRailVisible: boolean;
  onLeftRailVisibleChange?: (visible: boolean) => void;
  onRightRailVisibleChange?: (visible: boolean) => void;
}) => {
  render(
    <PlaygroundCockpitShell
      mode="cockpit"
      leftRailVisible={leftRailVisible}
      rightRailVisible={rightRailVisible}
      onLeftRailVisibleChange={onLeftRailVisibleChange}
      onRightRailVisibleChange={onRightRailVisibleChange}
      leftRail={<div>Context tools</div>}
      rightRail={<div>Runtime tools</div>}
    >
      <div>Chat transcript</div>
    </PlaygroundCockpitShell>,
  );

  return { onLeftRailVisibleChange, onRightRailVisibleChange };
};

describe("Playground cockpit rail restore tabs", () => {
  it("keeps restore controls mounted across rail visibility changes", () => {
    const { rerender } = render(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible={false}
        rightRailVisible={false}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    const contextRestore = screen.getByTestId(
      "playground-cockpit-left-rail-restore",
    );
    const runtimeRestore = screen.getByTestId(
      "playground-cockpit-right-rail-restore",
    );
    expect(contextRestore).not.toHaveAttribute("hidden");
    expect(runtimeRestore).not.toHaveAttribute("hidden");

    rerender(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible={false}
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );
    rerender(
      <PlaygroundCockpitShell
        mode="cockpit"
        leftRailVisible
        rightRailVisible
        leftRail={<div>Context tools</div>}
        rightRail={<div>Runtime tools</div>}
      >
        <div>Chat transcript</div>
      </PlaygroundCockpitShell>,
    );

    expect(screen.getByTestId("playground-cockpit-left-rail-restore")).toBe(
      contextRestore,
    );
    expect(screen.getByTestId("playground-cockpit-right-rail-restore")).toBe(
      runtimeRestore,
    );
    expect(contextRestore).not.toHaveAttribute("hidden");
    expect(runtimeRestore).not.toHaveAttribute("hidden");
  });

  it("mounts the context restore control on the chat content edge while runtime stays expanded", () => {
    const { onLeftRailVisibleChange } = renderShell({
      leftRailVisible: false,
      rightRailVisible: true,
    });

    expect(
      screen.queryByTestId("playground-cockpit-left-rail"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("playground-cockpit-right-rail"),
    ).toBeInTheDocument();

    const contextRestore = screen.getByTestId(
      "playground-cockpit-left-rail-restore",
    );
    expect(contextRestore).toHaveTextContent("Context rail");
    expect(contextRestore.parentElement).toHaveClass(
      ...COCKPIT_LEFT_RESTORE_WRAPPER_CLASS.split(" "),
    );
    expect(contextRestore.parentElement).not.toHaveClass("fixed");
    expect(contextRestore.parentElement).not.toHaveClass("left-12");
    expect(contextRestore.parentElement).not.toHaveClass("top-1/2");
    expect(contextRestore.parentElement).not.toHaveClass("-translate-y-1/2");
    expect(contextRestore.parentElement).not.toHaveClass("relative");
    expect(contextRestore).toHaveClass("h-24", "w-8");
    expect(contextRestore).not.toHaveClass("h-32", "w-9");
    expect(
      screen
        .getByTestId("playground-cockpit-main")
        .parentElement?.style.getPropertyValue("--cockpit-grid-columns"),
    ).toBe("minmax(0,1fr) minmax(240px,300px)");

    fireEvent.click(contextRestore);
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(true);
  });

  it("mounts the runtime restore control on the right edge while context stays expanded", () => {
    const { onRightRailVisibleChange } = renderShell({
      leftRailVisible: true,
      rightRailVisible: false,
    });

    expect(
      screen.getByTestId("playground-cockpit-left-rail"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("playground-cockpit-right-rail"),
    ).not.toBeInTheDocument();

    const runtimeRestore = screen.getByTestId(
      "playground-cockpit-right-rail-restore",
    );
    expect(runtimeRestore).toHaveTextContent("Runtime rail");
    expect(runtimeRestore.parentElement).toHaveClass(
      "absolute",
      "right-0",
      "top-1/2",
      "-translate-y-1/2",
    );
    expect(runtimeRestore.parentElement).not.toHaveClass("relative");
    expect(
      screen
        .getByTestId("playground-cockpit-main")
        .parentElement?.style.getPropertyValue("--cockpit-grid-columns"),
    ).toBe("minmax(220px,280px) minmax(0,1fr)");

    fireEvent.click(runtimeRestore);
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(true);
  });

  it("keeps both collapsed rail restore tabs visible without adding an in-flow collapsed panel", () => {
    const { onLeftRailVisibleChange, onRightRailVisibleChange } = renderShell({
      leftRailVisible: false,
      rightRailVisible: false,
    });

    const contextRestore = screen.getByTestId(
      "playground-cockpit-left-rail-restore",
    );
    const runtimeRestore = screen.getByTestId(
      "playground-cockpit-right-rail-restore",
    );

    expect(contextRestore).toHaveTextContent("Context rail");
    expect(runtimeRestore).toHaveTextContent("Runtime rail");
    expect(
      screen.queryByTestId("playground-collapsed-composition-summary"),
    ).not.toBeInTheDocument();
    expect(
      screen
        .getByTestId("playground-cockpit-main")
        .parentElement?.style.getPropertyValue("--cockpit-grid-columns"),
    ).toBe("minmax(0,1fr)");
    expect(screen.getByTestId("playground-cockpit-main")).toHaveClass("w-full");
    expect(screen.getByTestId("playground-cockpit-main")).not.toHaveClass(
      "max-w-[72rem]",
    );

    fireEvent.click(contextRestore);
    fireEvent.click(runtimeRestore);
    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(true);
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(true);
  });

  it("keeps mobile restore tabs available when cockpit rails are hidden", () => {
    const { onLeftRailVisibleChange, onRightRailVisibleChange } = renderShell({
      leftRailVisible: false,
      rightRailVisible: false,
    });

    const mobileRails = screen.getByTestId("playground-cockpit-mobile-rails");
    expect(mobileRails).toHaveAttribute("data-mobile-panel", "none");

    const contextTab = screen.getByRole("tab", {
      name: /restore context sidechannel/i,
    });
    const runtimeTab = screen.getByRole("tab", {
      name: /restore runtime sidechannel/i,
    });
    const contextPanel = document.getElementById(
      contextTab.getAttribute("aria-controls") ?? "",
    );
    const runtimePanel = document.getElementById(
      runtimeTab.getAttribute("aria-controls") ?? "",
    );

    expect(contextPanel).toHaveAttribute("role", "tabpanel");
    expect(contextPanel).not.toBeVisible();
    expect(runtimePanel).toHaveAttribute("role", "tabpanel");
    expect(runtimePanel).not.toBeVisible();

    fireEvent.click(contextTab);
    fireEvent.click(runtimeTab);

    expect(onLeftRailVisibleChange).toHaveBeenCalledWith(true);
    expect(onRightRailVisibleChange).toHaveBeenCalledWith(true);
  });
});
