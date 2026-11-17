import React from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ACPToolsPanel } from "../ACPToolsPanel";
import { useACPSessionsStore } from "@/store/acp-sessions";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

beforeEach(() => useACPSessionsStore.getState().reset());
afterEach(() => {
  cleanup();
  useACPSessionsStore.getState().reset();
});

const session = (capabilities?: Record<string, unknown>) => {
  const store = useACPSessionsStore.getState();
  const id = store.createSession({
    cwd: "/public/example",
    agentType: "opencode",
  });
  if (capabilities) store.setSessionCapabilities(id, capabilities);
  return id;
};

describe("ACPToolsPanel truthful scope", () => {
  it("keeps the no-session selection guidance", () => {
    render(<ACPToolsPanel />);
    expect(
      screen.getByText("Select a session to view capabilities"),
    ).toBeInTheDocument();
  });

  it("does not invent capabilities or tools for saved history without a live report", () => {
    session();
    render(<ACPToolsPanel />);
    expect(screen.getByText("Read Files: Not reported")).toBeInTheDocument();
    expect(screen.getByText("Write Files: Not reported")).toBeInTheDocument();
    expect(screen.getByText("Terminal: Not reported")).toBeInTheDocument();
    expect(
      screen.getByText("The agent has not reported a tool list."),
    ).toBeInTheDocument();
    expect(screen.queryByText("fs.read")).not.toBeInTheDocument();
    expect(screen.queryByText("exec.bash")).not.toBeInTheDocument();
  });

  it("retains explicit supported/unsupported reports and an empty tool list", () => {
    session({
      fs: { readTextFile: true, writeTextFile: false },
      terminal: false,
      tools: [],
    });
    render(<ACPToolsPanel />);
    expect(screen.getByText("Read Files: Supported")).toBeInTheDocument();
    expect(screen.getByText("Write Files: Unsupported")).toBeInTheDocument();
    expect(screen.getByText("Terminal: Unsupported")).toBeInTheDocument();
    expect(
      screen.getByText("The agent reported no tools."),
    ).toBeInTheDocument();
    expect(screen.queryByText("fs.read")).not.toBeInTheDocument();
  });

  it("shows only reported names and separates default tiers from authorization", () => {
    session({ tools: ["fs.read", "c11_public_readonly_echo"] });
    render(<ACPToolsPanel />);
    expect(screen.getByText("fs.read")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Other"));
    expect(screen.getByText("c11_public_readonly_echo")).toBeInTheDocument();
    expect(screen.queryByText("fs.write")).not.toBeInTheDocument();
    expect(screen.getByText("Default tier: auto")).toBeInTheDocument();
    expect(
      screen.queryByText(
        "This operation is read-only and will be automatically approved.",
      ),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Tier not reported")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Agent support is not authorization. The active session policy and permission requests determine access.",
      ),
    ).toBeInTheDocument();
  });
  it("treats prototype property names as unclassified reported tools", () => {
    session({ tools: ["__proto__", "constructor"] });
    render(<ACPToolsPanel />);
    fireEvent.click(screen.getByText("Other"));
    expect(screen.getByText("__proto__")).toBeInTheDocument();
    expect(screen.getByText("constructor")).toBeInTheDocument();
    expect(screen.getAllByText("Tier not reported")).toHaveLength(2);
  });
});
