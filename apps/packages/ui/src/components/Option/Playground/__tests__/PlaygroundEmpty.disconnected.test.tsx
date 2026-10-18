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
