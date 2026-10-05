import "./dialogTestSetup";
import React, { useState } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EvidenceRail } from "../evidence/EvidenceRail";

vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => false,
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

describe("mobile evidence dialog", () => {
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
