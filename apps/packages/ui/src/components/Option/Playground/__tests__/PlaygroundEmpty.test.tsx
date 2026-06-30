import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { PlaygroundEmpty } from "../PlaygroundEmpty";

const openHelpModal = vi.fn();
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
  useIsConnected: () => true,
}));

vi.mock("@/store/tutorials", () => ({
  useHelpModal: () => ({ open: openHelpModal }),
}));

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigate,
}));

describe("PlaygroundEmpty", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders one unified onboarding shell", () => {
    render(<PlaygroundEmpty />);

    const shell = screen.getByTestId("playground-empty-shell");

    expect(shell).toHaveAttribute("data-ds-component", "EmptyState");
    expect(
      within(shell).getByRole("heading", { name: "Start a new chat" }),
    ).toBeInTheDocument();
    expect(
      within(shell).getByRole("button", { name: "Start chatting" }),
    ).toBeInTheDocument();
    expect(
      within(shell).getByRole("button", { name: "Quick Ingest" }),
    ).toBeInTheDocument();
    expect(
      within(shell).getByRole("button", { name: "Explore chat modes" }),
    ).toBeInTheDocument();
    expect(
      within(shell).queryByRole("button", {
        name: /Compare AI models side-by-side/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(shell).getByRole("button", { name: "Take a quick tour" }),
    ).toBeInTheDocument();
    expect(
      within(shell).queryByRole("button", { name: /open settings/i }),
    ).not.toBeInTheDocument();
    expect(
      within(shell).getByText(
        "Experiment with different models, prompts, and knowledge sources here.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Start with a guided mode:"),
    ).not.toBeInTheDocument();
  });

  it("keeps starter modes behind a discoverable launcher on first render", () => {
    render(<PlaygroundEmpty />);

    const shell = screen.getByTestId("playground-empty-shell");

    expect(
      within(shell).queryByRole("button", {
        name: /Compare AI models side-by-side/i,
      }),
    ).not.toBeInTheDocument();

    fireEvent.click(
      within(shell).getByRole("button", { name: "Explore chat modes" }),
    );

    expect(
      within(shell).getByRole("button", {
        name: /Compare AI models side-by-side/i,
      }),
    ).toBeInTheDocument();
  });

  it("dispatches starter telemetry and starter action events when compare is selected", () => {
    const dispatchSpy = vi.spyOn(window, "dispatchEvent");
    render(<PlaygroundEmpty />);

    fireEvent.click(screen.getByRole("button", { name: "Explore chat modes" }));
    fireEvent.click(
      screen.getByRole("button", { name: /Compare AI models side-by-side/i }),
    );

    expect(dispatchSpy).toHaveBeenCalled();
    const compareEvent = dispatchSpy.mock.calls
      .map((call) => call[0])
      .find((event) => event.type === "tldw:playground-starter") as
      | CustomEvent
      | undefined;
    expect(compareEvent).toBeDefined();
    expect((compareEvent as CustomEvent).detail).toMatchObject({
      mode: "compare",
    });

    const telemetryEvent = dispatchSpy.mock.calls
      .map((call) => call[0])
      .find((event) => event.type === "tldw:playground-starter-selected") as
      | CustomEvent
      | undefined;
    expect(telemetryEvent).toBeDefined();
    expect((telemetryEvent as CustomEvent).detail).toMatchObject({
      mode: "compare",
    });
  });

  it("does not render the page layout guide card", () => {
    render(<PlaygroundEmpty />);

    expect(screen.queryByText("Page layout")).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        "Chat history (left), conversation (center), message input (bottom), sources & tools (right).",
      ),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Open history" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Start typing" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Open sources & tools" }),
    ).not.toBeInTheDocument();
  });

  it("opens the help modal from the quick tour action", () => {
    render(<PlaygroundEmpty />);

    fireEvent.click(screen.getByRole("button", { name: "Take a quick tour" }));

    expect(openHelpModal).toHaveBeenCalledTimes(1);
  });

  it("does not render the stale try-asking prompt suggestions", () => {
    render(<PlaygroundEmpty />);

    expect(screen.queryByText("Try asking:")).not.toBeInTheDocument();
    expect(
      screen.queryByText(
        "Summarize the key points from my last uploaded document",
      ),
    ).not.toBeInTheDocument();
  });

  it("routes the deep research starter to the research console", () => {
    render(<PlaygroundEmpty />);

    fireEvent.click(screen.getByRole("button", { name: "Explore chat modes" }));
    fireEvent.click(screen.getByRole("button", { name: /Deep Research/i }));

    expect(navigate).toHaveBeenCalledWith("/research");
  });
});
