import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { PlaygroundEmpty } from "../PlaygroundEmpty";

const navigate = vi.fn();
const connection = vi.hoisted(() => ({
  state: {} as Record<string, unknown>,
}));
const FAILED_CHECK = {
  phase: "error",
  isConnected: false,
  isChecking: false,
  lastCheckedAt: 1,
  consecutiveFailures: 1,
};
const FIRST_CHECK_IN_FLIGHT = {
  phase: "searching",
  isConnected: false,
  isChecking: true,
  lastCheckedAt: null,
  consecutiveFailures: 0,
};

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
  useConnectionState: () => connection.state,
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
    connection.state = FAILED_CHECK;
  });

  // CC-03 (#3101): no "Connect to start chatting" while the first check runs.
  it("keeps the chat card while the first server check is still running", () => {
    connection.state = FIRST_CHECK_IN_FLIGHT;
    render(<PlaygroundEmpty />);
    const shell = screen.getByTestId("playground-empty-shell");
    expect(within(shell).getByText("Start a new chat")).toBeInTheDocument();
    expect(
      within(shell).getByText(
        "Connecting to your tldw server. You can start writing your first message below.",
      ),
    ).toBeInTheDocument();
    expect(within(shell).queryByRole("button", { name: "Connect server" })).not.toBeInTheDocument();
  });

  it("offers connection setup before model choice and preserves intercepted navigation", () => {
    const setup = vi.fn((event: Event) => event.preventDefault());
    window.addEventListener("tldw:settings-navigation-request", setup);
    try {
      render(<PlaygroundEmpty />);
      const shell = screen.getByTestId("playground-empty-shell");
      expect(within(shell).getByText("Connect to your tldw server, then choose a model and start chatting.")).toBeInTheDocument();
      fireEvent.click(within(shell).getByRole("button", { name: "Connect server" }));
      expect(setup).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ detail: { destination: "/settings/tldw" } }));
      expect(navigate).not.toHaveBeenCalled();
      expect(within(shell).queryByRole("button", { name: "Explore chat modes" })).not.toBeInTheDocument();
    } finally { window.removeEventListener("tldw:settings-navigation-request", setup); }
  });
});
