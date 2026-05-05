import React from "react";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConnectionPhase } from "@/types/connection";
import { EmptySidePanel } from "../empty";

const connectionStateMock = vi.hoisted(() => ({
  state: {
    phase: "unconfigured",
    isConnected: false,
    serverUrl: null,
  },
  uxState: {
    uxState: "unconfigured",
    mode: "normal",
    configStep: "url",
    hasCompletedFirstRun: false,
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key,
  }),
}));

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: vi.fn((path: string) => `chrome-extension://test${path}`),
      sendMessage: vi.fn(),
    },
    tabs: {
      create: vi.fn(),
    },
  },
}));

vi.mock("antd", () => ({
  Button: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
  Tooltip: ({ children }: { children?: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => connectionStateMock.state,
  useConnectionUxState: () => connectionStateMock.uxState,
}));

describe("EmptySidePanel", () => {
  const renderPanel = () => render(<EmptySidePanel />);

  beforeEach(() => {
    connectionStateMock.state = {
      phase: "unconfigured",
      isConnected: false,
      serverUrl: null,
    };
    connectionStateMock.uxState = {
      uxState: "unconfigured",
      mode: "normal",
      configStep: "url",
      hasCompletedFirstRun: false,
    };
  });

  it("frames setup around Companion Home instead of a chat-only launch", () => {
    renderPanel();

    const panel = screen.getByTestId("chat-empty-connection");

    expect(panel).toHaveAttribute("data-ds-component", "RecoveryCallout");
    expect(screen.getByText("Setup required")).toBeInTheDocument();
    expect(
      screen.getByText("Finish setup to open Companion Home"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Before you can use Companion Home here, finish the short setup flow in Options to connect tldw Assistant to your tldw server or choose demo mode.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Setup step")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Step 1 of 3 — Add your server URL in Options → tldw Server.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Finish setup" })).toHaveFocus();
  });

  it("shows the auth recovery copy when credentials need attention", () => {
    connectionStateMock.uxState = {
      uxState: "error_auth",
      mode: "normal",
      configStep: "auth",
      hasCompletedFirstRun: true,
    };

    renderPanel();

    expect(screen.getByTestId("chat-empty-connection")).toHaveAttribute(
      "data-ds-component",
      "RecoveryCallout",
    );
    expect(screen.getByText("Sign in required")).toBeInTheDocument();
    expect(screen.getByText("API key needs attention")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Your server is up but the API key is wrong or missing. Fix the key in Settings → tldw server, then retry.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Review settings" }),
    ).toBeInTheDocument();
  });

  it("shows the unreachable-server copy when the server cannot be reached", () => {
    connectionStateMock.state = {
      phase: "error",
      isConnected: false,
      serverUrl: "http://localhost:8000",
    };
    connectionStateMock.uxState = {
      uxState: "error_unreachable",
      mode: "normal",
      configStep: "health",
      hasCompletedFirstRun: true,
    };

    renderPanel();

    expect(screen.getByTestId("chat-empty-connection")).toHaveAttribute(
      "data-ds-component",
      "RecoveryCallout",
    );
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
    expect(
      screen.getByText("Can’t reach your tldw server"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/We couldn’t reach \{\{host\}\}\./),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Review settings" }),
    ).toBeInTheDocument();
  });

  it("shows the post-onboarding connection copy when first run is complete", () => {
    connectionStateMock.uxState = {
      uxState: "unconfigured",
      mode: "normal",
      configStep: "url",
      hasCompletedFirstRun: true,
    };

    renderPanel();

    expect(
      screen.getByText(
        "Connect tldw Assistant to your server to open Companion Home",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "tldw_server is your private AI workspace that keeps your home dashboard, chats, notes, and media on your own machine. Add your server URL to get started.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Open tldw Settings" }),
    ).toBeInTheDocument();
  });

  it("shows suggestion cards when the sidepanel is connected", () => {
    connectionStateMock.state = {
      phase: ConnectionPhase.CONNECTED,
      isConnected: true,
      serverUrl: "http://localhost:8000",
    };
    connectionStateMock.uxState = {
      uxState: "connected_ok",
      mode: "normal",
      configStep: "health",
      hasCompletedFirstRun: true,
    };

    renderPanel();

    expect(screen.getByTestId("chat-empty-connected")).toBeInTheDocument();
    expect(screen.getByText("Try asking")).toBeInTheDocument();
    expect(screen.getByTestId("chat-suggestion-1")).toBeInTheDocument();
    expect(screen.getByTestId("chat-suggestion-2")).toBeInTheDocument();
    expect(screen.getByTestId("chat-suggestion-3")).toBeInTheDocument();
  });
});
