// @vitest-environment jsdom
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { FirstChatStep } from "../steps/FirstChatStep";

describe("FirstChatStep", () => {
  it("requires a successful first chat before calling complete", async () => {
    const verifyFirstChat = vi.fn().mockResolvedValue({
      status: "ready",
      provider: "openai",
      model: "gpt-4.1-mini",
      response_text: "Hello.",
    });
    const complete = vi.fn().mockResolvedValue({
      success: true,
      message: "completed",
      requires_restart: false,
      install_plan_submitted: false,
    });
    const onComplete = vi.fn();

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={complete}
        onComplete={onComplete}
        onBack={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText("Hello.");
    await waitFor(() => expect(complete).toHaveBeenCalled());
    expect(onComplete).toHaveBeenCalled();
  });

  it("does not complete setup when first chat verification fails", async () => {
    const verifyFirstChat = vi.fn().mockResolvedValue({
      status: "failed",
      provider: "openai",
      model: "gpt-4.1-mini",
      failure_category: "auth",
      message: "Invalid API key",
    });
    const complete = vi.fn();
    const onBack = vi.fn();

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={complete}
        onComplete={vi.fn()}
        onBack={onBack}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText(/invalid api key/i);
    expect(complete).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /back to providers/i }));
    expect(onBack).toHaveBeenCalled();
  });

  it("reports setup completion errors separately from first chat verification", async () => {
    const verifyFirstChat = vi.fn().mockResolvedValue({
      status: "ready",
      provider: "openai",
      model: "gpt-4.1-mini",
      response_text: "Hello.",
    });
    const complete = vi
      .fn()
      .mockRejectedValue(new Error("Completion state write failed"));
    const onComplete = vi.fn();

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={complete}
        onComplete={onComplete}
        onBack={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText("Hello.");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /setup completion failed/i,
    );
    expect(screen.getByRole("alert")).not.toHaveTextContent(
      /first chat failed/i,
    );
    expect(onComplete).not.toHaveBeenCalled();
  });
});
