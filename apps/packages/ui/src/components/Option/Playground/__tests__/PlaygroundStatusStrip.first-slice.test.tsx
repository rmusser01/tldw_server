// @vitest-environment jsdom
import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { PlaygroundStatusStrip } from "../PlaygroundStatusStrip";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback,
  }),
}));

describe("PlaygroundStatusStrip first-slice state", () => {
  it("renders model, provider, context, persistence, and message state together", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={3}
        sessionLabel="Server chat"
        hasContext
        contextSummary={["Web search on", "2 files"]}
        temporaryChat={false}
        degradedChecks={[]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Ready");
    expect(status).toHaveTextContent("Cockpit");
    expect(status).toHaveTextContent("Server chat");
    expect(status).toHaveTextContent("Saved");
    expect(status).toHaveTextContent("Context active");
    expect(status).toHaveTextContent("Web search on");
    expect(status).toHaveTextContent("2 files");
    expect(status).toHaveTextContent("openai");
    expect(status).toHaveTextContent("gpt-4.1-mini");
    expect(status).toHaveTextContent("3 messages");
  });

  it("renders degraded checks as warning-only status", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={0}
        sessionLabel="Local chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat
        degradedChecks={["Embeddings unavailable"]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Degraded");
    expect(status).toHaveTextContent("Embeddings unavailable");
    expect(status).toHaveTextContent("Temporary");
  });

  it("renders recoverable error state ahead of degraded state", () => {
    render(
      <PlaygroundStatusStrip
        mode="focus"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={0}
        sessionLabel="Local chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        degradedChecks={["Embeddings unavailable"]}
        errorMessage="Provider failed"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Error");
    expect(status).toHaveTextContent("Provider failed");
  });
});
