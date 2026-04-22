import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
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

  it("renders an Open Settings button inside the unified onboarding shell when disconnected", () => {
    render(<PlaygroundEmpty />);

    const shell = screen.getByTestId("playground-empty-shell");
    const settingsButton = within(shell).getByRole("button", {
      name: /open settings/i,
    });
    expect(settingsButton).toBeInTheDocument();
  });

  it("navigates to /settings/tldw when Open Settings is clicked", () => {
    render(<PlaygroundEmpty />);

    const shell = screen.getByTestId("playground-empty-shell");
    const settingsButton = within(shell).getByRole("button", {
      name: /open settings/i,
    });
    fireEvent.click(settingsButton);

    expect(navigate).toHaveBeenCalledWith("/settings/tldw");
  });
});
