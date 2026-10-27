import "./dialogTestSetup";
import React, { useState } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EvidenceRail } from "../evidence/EvidenceRail";

vi.mock("@/hooks/useMediaQuery", () => ({
  useDesktop: () => false,
  useMediaQuery: () => true,
}));
vi.mock("../SourceList", () => ({
  SourceList: () => <button>Inspect source</button>,
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

describe("Evidence with reduced motion", () => {
  it("keeps evidence inspection and closing usable when motion is reduced", async () => {
    const user = userEvent.setup();
    render(<Evidence />);
    const trigger = screen.getByRole("button", { name: "Open evidence panel" });
    await user.click(trigger);
    expect(screen.getByRole("dialog", { name: "Evidence" })).toBeVisible();
    expect(
      screen.getByText(
        /Use each source card to inspect excerpts, unavailable reasons, citations, and supported source links/,
      ),
    ).toBeVisible();
    await user.click(
      screen.getByRole("button", { name: "Close evidence panel" }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(trigger).toHaveFocus();
  });
});
