import React from "react";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { PlaygroundEmpty } from "../PlaygroundEmpty";

const navigate = vi.fn();

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, arg?: string | { defaultValue?: string }) => {
      if (typeof arg === "string") return arg;
      if (arg && typeof arg === "object" && arg.defaultValue) {
        return arg.defaultValue;
      }
      return _key;
    },
  }),
}));

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false }),
}));

vi.mock("@/hooks/useConnectionState", () => ({
  useIsConnected: () => false,
}));

vi.mock("@/store/tutorials", () => ({
  useHelpModal: () => ({ open: vi.fn() }),
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigate,
}));

describe("PlaygroundEmpty – disconnected state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps the blank chat shell neutral when disconnected", () => {
    render(<PlaygroundEmpty />);

    const shell = screen.getByTestId("playground-empty-shell");

    expect(
      within(shell).getByText(
        "Experiment with different models, prompts, and knowledge sources here.",
      ),
    ).toBeInTheDocument();
    expect(
      within(shell).queryByText("Connect to a tldw server to start chatting."),
    ).not.toBeInTheDocument();
    expect(
      within(shell).queryByRole("button", { name: /open settings/i }),
    ).not.toBeInTheDocument();
    expect(navigate).not.toHaveBeenCalled();
  });
});
