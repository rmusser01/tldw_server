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
  it("shows active context source chips without routine session noise", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={3}
        sessionLabel="Server chat"
        sessionStatus="loaded"
        sessionStatusLabel="Local history"
        sessionDetail="History linked"
        hasContext
        contextSummary={["Web search on", "2 files"]}
        temporaryChat={false}
        degradedChecks={[]}
        errorMessage={null}
        onOpenSearchContext={vi.fn()}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Ready");
    expect(status).toHaveTextContent("openai:gpt-4.1-mini");
    expect(status).toHaveTextContent("3 messages");
    expect(status).not.toHaveTextContent("Cockpit");
    expect(status).not.toHaveTextContent("Server chat");
    expect(status).not.toHaveTextContent("Local history");
    expect(status).not.toHaveTextContent("History linked");
    expect(status).not.toHaveTextContent("Saved");
    expect(status).not.toHaveTextContent("Context active");
    expect(status).toHaveTextContent("Web search on");
    expect(status).toHaveTextContent("2 files");
  });

  it("does not render stale context source chips when context is inactive", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={3}
        sessionLabel="Server chat"
        sessionStatus="loaded"
        hasContext={false}
        contextSummary={["Web search on"]}
        temporaryChat={false}
        degradedChecks={[]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).not.toHaveTextContent("Web search on");
    expect(
      screen.queryByRole("button", { name: "Open Search & Context" }),
    ).toBeNull();
  });

  it("does not treat routine temporary-session labels as critical status", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={1}
        sessionLabel="Temporary chat"
        sessionStatus="idle"
        sessionStatusLabel="Local only"
        sessionDetail="Not saved"
        hasContext={false}
        contextSummary={[]}
        temporaryChat
        degradedChecks={[]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Ready");
    expect(status).toHaveTextContent("openai:gpt-4.1-mini");
    expect(status).toHaveTextContent("1 message");
    expect(status).not.toHaveTextContent("Temporary chat");
    expect(status).not.toHaveTextContent("Local only");
    expect(status).not.toHaveTextContent("Not saved");
  });

  it("labels temporary Character Chat persistence explicitly", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={1}
        sessionLabel="Temporary chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Temporary character chat");
  });

  it("labels saved Character Chat persistence when session history is loaded", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={4}
        sessionLabel="Server chat"
        sessionStatus="loaded"
        sessionStatusLabel="Local history"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Saved character chat");
  });

  it("labels blocked-server Character Chat as a local draft", () => {
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
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        serverBlocked
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Local character chat draft");
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
    expect(status).not.toHaveTextContent("Temporary");
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

  it("does not show Ready when the selected Character Chat model is unavailable", () => {
    const openModelSettings = vi.fn();

    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="tldw"
        selectedModel="gpt-4o"
        messageCount={1}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUnavailable
        modelUnavailableMessage="Choose a chat model before chatting as Ada"
        onOpenModelSettings={openModelSettings}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model unavailable");
    expect(status).toHaveTextContent("Choose a chat model before chatting as Ada");
    expect(status).not.toHaveTextContent("Ready");
    fireEvent.click(screen.getByRole("button", { name: "Open model settings" }));
    expect(openModelSettings).toHaveBeenCalledTimes(1);
  });

  it("surfaces provider setup model usability without positive health copy", () => {
    const openModelSettings = vi.fn();

    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={1}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="provider_unconfigured"
        modelUsabilityMessage="Configure the selected model provider before chatting as Ada"
        onOpenModelSettings={openModelSettings}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model setup needed");
    expect(status).toHaveTextContent(
      "Configure the selected model provider before chatting as Ada",
    );
    expect(status).not.toHaveTextContent("Ready");
    expect(status).not.toHaveTextContent("Healthy");
    fireEvent.click(screen.getByRole("button", { name: "Open model settings" }));
    expect(openModelSettings).toHaveBeenCalledTimes(1);
  });

  it("surfaces not-callable model usability without positive health copy", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={1}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="model_unavailable"
        modelUsabilityMessage="The selected chat model is not callable right now"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model not callable");
    expect(status).toHaveTextContent(
      "The selected chat model is not callable right now",
    );
    expect(status).not.toHaveTextContent("Ready");
    expect(status).not.toHaveTextContent("Healthy");
  });

  it("surfaces loading model usability before ready copy", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="loading"
        modelUsabilityMessage="Checking chat model readiness"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Checking model");
    expect(status).toHaveTextContent("Checking chat model readiness");
    expect(status).not.toHaveTextContent("Ready");
  });

  it("uses character-specific model usability copy for no selected model", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider={null}
        selectedModel={null}
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="no_selection"
        modelUsabilityMessage="Choose a chat model before chatting as Ada"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("No model selected");
    expect(status).toHaveTextContent(
      "Choose a chat model before chatting as Ada",
    );
    expect(status).not.toHaveTextContent("Choose a model before sending.");
    expect(status).not.toHaveTextContent("Ready");
  });

  it("does not reuse selected-character fallback copy as model usability detail", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider={null}
        selectedModel={null}
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="no_selection"
        modelUnavailable={false}
        modelUnavailableMessage="Choose a character to start character chat"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("No model selected");
    expect(status).toHaveTextContent("Choose a model before sending.");
    expect(status).not.toHaveTextContent(
      "Choose a character to start character chat",
    );
    expect(status).not.toHaveTextContent("Ready");
  });

  it("does not reuse legacy fallback copy for explicit model usability blockers", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="provider_unconfigured"
        modelUnavailable={false}
        modelUnavailableMessage="Choose a character to start character chat"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model setup needed");
    expect(status).toHaveTextContent("Review model settings before sending.");
    expect(status).not.toHaveTextContent(
      "Choose a character to start character chat",
    );
    expect(status).not.toHaveTextContent("Ready");
  });

  it("does not show positive degraded copy for legacy unavailable models", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degraded
        degradedChecks={["Provider health degraded"]}
        errorMessage={null}
        modelUnavailable
        modelUnavailableMessage="Choose a chat model before chatting as Ada"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model unavailable");
    expect(status).toHaveTextContent(
      "Choose a chat model before chatting as Ada",
    );
    expect(status).not.toHaveTextContent("Chat remains available.");
    expect(status).not.toHaveTextContent("Ready");
  });

  it("treats disallowed degraded model usability as blocked", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degraded
        degradedChecks={["Provider health degraded"]}
        errorMessage={null}
        modelUsabilityStatus="degraded"
        modelUsabilityCanSend={false}
        modelUsabilityMessage="Character chat is preparing"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Model unavailable");
    expect(status).toHaveTextContent("Character chat is preparing");
    expect(status).not.toHaveTextContent("Chat remains available.");
    expect(status).not.toHaveTextContent("Ready");
  });

  it("maps explicit no-server model usability to server blocked copy", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={0}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="no_server"
        modelUsabilityMessage="Connect to tldw_server before starting character chat"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Server unavailable");
    expect(status).toHaveTextContent(
      "Connect to tldw_server before starting character chat",
    );
    expect(status).not.toHaveTextContent("Ready");
  });

  it("keeps streaming primary over blocked model usability", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming
        selectedProvider="openai"
        selectedModel="gpt-4o"
        messageCount={1}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive
        degradedChecks={[]}
        errorMessage={null}
        modelUsabilityStatus="provider_unconfigured"
        modelUsabilityMessage="Configure the selected model provider before chatting as Ada"
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Streaming");
    expect(status).not.toHaveTextContent("Model setup needed");
  });

  it("preserves non-character chat ready behavior without model usability", () => {
    render(
      <PlaygroundStatusStrip
        mode="cockpit"
        streaming={false}
        selectedProvider="openai"
        selectedModel="gpt-4.1-mini"
        messageCount={1}
        sessionLabel="Server chat"
        hasContext={false}
        contextSummary={[]}
        temporaryChat={false}
        characterChatActive={false}
        degradedChecks={[]}
        errorMessage={null}
      />,
    );

    const status = screen.getByRole("status", { name: "Chat status" });
    expect(status).toHaveTextContent("Ready");
    expect(status).toHaveTextContent("openai:gpt-4.1-mini");
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
        sessionStatus="failed"
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
