// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
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
    expect(status).toHaveTextContent("Chat remains available.");
    expect(status).toHaveTextContent("Temporary");
  });

  it("keeps active streaming primary while degraded health remains warning-only", () => {
    const stopStreaming = vi.fn();

    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={2}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        degradedChecks={["Embeddings unavailable"]}
        errorMessage={null}
        onStopStreaming={stopStreaming}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Streaming");
    expect(status).toHaveTextContent("Embeddings unavailable");
    expect(status).toHaveTextContent("Chat remains available.");
    fireEvent.click(screen.getByRole("button", { name: "Stop generation" }));
    expect(stopStreaming).toHaveBeenCalledTimes(1);
  });

  it("surfaces missing model as a recoverable send blocker", () => {
    const openModelSettings = vi.fn();

    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider={null}
        selectedModel={null}
        messageCount={0}
        sessionLabel="Local chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat
        degradedChecks={[]}
        errorMessage={null}
        onOpenModelSettings={openModelSettings}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("No model selected");
    expect(status).toHaveTextContent("Choose a model before sending.");
    fireEvent.click(screen.getByRole("button", { name: "Open model settings" }));
    expect(openModelSettings).toHaveBeenCalledTimes(1);
  });

  it("shows context preview loading without treating the chat route as degraded", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext
        contextSummary={["2 knowledge items"]}
        temporaryChat={false}
        degradedChecks={[]}
        errorMessage={null}
        compositionStatus="loading"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Loading context");
    expect(status).toHaveTextContent("Context preview is loading.");
    expect(status).not.toHaveTextContent("Degraded");
  });

  it("distinguishes blocked server readiness from warning-only degraded health", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        degradedChecks={["Embeddings unavailable"]}
        errorMessage={null}
        serverBlocked
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Server unavailable");
    expect(status).toHaveTextContent(
      "Reconnect to the server or review server settings before sending.",
    );
    expect(status).not.toHaveTextContent("Chat remains available.");
  });

  it("renders recoverable error state ahead of degraded state", () => {
    const openModelSettings = vi.fn();

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
        onOpenModelSettings={openModelSettings}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Error");
    expect(status).toHaveTextContent("Provider failed");
    fireEvent.click(screen.getByRole("button", { name: "Review model settings" }));
    expect(openModelSettings).toHaveBeenCalledTimes(1);
  });

  it("keeps server-session failures visible when cockpit rails are hidden", () => {
    const openModelSettings = vi.fn();

    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={5}
        sessionLabel="Server chat"
        sessionTitle="Archived investigation"
        sessionStatusLabel="Load failed"
        sessionDetail="Failed to load conversation"
        sessionError="Conversation no longer exists"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        degradedChecks={["Chacha notes unavailable"]}
        errorMessage={null}
        onOpenModelSettings={openModelSettings}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Server chat");
    expect(status).toHaveTextContent("Archived investigation");
    expect(status).toHaveTextContent("Load failed");
    expect(status).toHaveTextContent("Conversation no longer exists");
    expect(status).toHaveTextContent("Chacha notes unavailable");
    expect(status).toHaveTextContent("Chat remains available.");
  });
});
