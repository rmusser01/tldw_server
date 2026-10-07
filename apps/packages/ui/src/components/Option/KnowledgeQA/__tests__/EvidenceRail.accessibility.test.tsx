import "./dialogTestSetup";
import React, { useState } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvidenceRail } from "../evidence/EvidenceRail";

const layout = vi.hoisted(() => ({ desktop: false }));
vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => layout.desktop,
  useMediaQuery: () => false,
}));
vi.mock("../SourceList", () => ({
  SourceList: () => <button>Inspect source</button>,
}));
vi.mock("../SearchDetailsPanel", () => ({
  SearchDetailsPanel: () => <p>Search details</p>,
}));

function Evidence() {
  const [open, setOpen] = useState(false);
  return (
    <EvidenceRail
      open={open}
      onOpenChange={setOpen}
      tab="sources"
      onTabChange={() => {}}
      resultsCount={1}
      citationsCount={1}
    />
  );
}

describe("evidence headers and focus", () => {
  beforeEach(() => {
    layout.desktop = false;
  });
  it("keeps the desktop rail heading and dismissal control", async () => {
    layout.desktop = true;
    const user = userEvent.setup();
    render(<Evidence />);
    await user.click(
      screen.getByRole("button", { name: "Open evidence panel (1 source)" }),
    );
    const rail = screen.getByRole("complementary", { name: "Evidence panel" });
    expect(
      within(rail).getAllByRole("heading", { name: "Evidence" }),
    ).toHaveLength(1);
    expect(
      within(rail).getAllByRole("button", { name: /close/i }),
    ).toHaveLength(1);
    expect(
      within(rail).getByText("1 sources • 1 citations"),
    ).toBeInTheDocument();
    await user.click(
      within(rail).getByRole("button", { name: "Close evidence panel" }),
    );
    expect(
      screen.queryByRole("complementary", { name: "Evidence panel" }),
    ).toBeNull();
  });
  it("has one accessible mobile Evidence heading", async () => {
    const user = userEvent.setup();
    render(<Evidence />);
    await user.click(
      screen.getByRole("button", { name: "Open evidence panel" }),
    );
    const dialog = screen.getByRole("dialog", { name: "Evidence" });
    expect(within(dialog).getAllByText("Evidence")).toHaveLength(1);
    expect(
      within(dialog).getAllByRole("heading", { name: "Evidence" }),
    ).toHaveLength(1);
  });
  it("has one mobile dismissal control", async () => {
    const user = userEvent.setup();
    render(<Evidence />);
    await user.click(
      screen.getByRole("button", { name: "Open evidence panel" }),
    );
    const dialog = screen.getByRole("dialog", { name: "Evidence" });
    expect(
      within(dialog).getAllByRole("button", { name: /close/i }),
    ).toHaveLength(1);
  });
  it("enters and contains focus and returns to Evidence after Escape", async () => {
    const user = userEvent.setup();
    render(<Evidence />);
    const trigger = screen.getByRole("button", { name: "Open evidence panel" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Evidence" });
    await waitFor(() =>
      expect(dialog.contains(document.activeElement)).toBe(true),
    );
    for (let i = 0; i < 12; i++) {
      await user.tab({ shift: i >= 6 });
      expect(dialog.contains(document.activeElement)).toBe(true);
    }
    await user.keyboard("{Escape}");
    await waitFor(() => expect(trigger).toHaveFocus());
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
