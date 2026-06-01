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
        onEditProvider={vi.fn()}
        onSwitchProvider={vi.fn()}
        onSkip={vi.fn()}
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
      failure_category: "auth_failed",
      message: "Invalid API key",
    });
    const complete = vi.fn();
    const onEditProvider = vi.fn();
    const onSwitchProvider = vi.fn();
    const onSkip = vi.fn();

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={complete}
        onComplete={vi.fn()}
        onBack={vi.fn()}
        onEditProvider={onEditProvider}
        onSwitchProvider={onSwitchProvider}
        onSkip={onSkip}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText(/invalid api key/i);
    expect(
      screen.getByText(/update the provider credentials/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/or switch to another provider/i),
    ).toBeInTheDocument();
    expect(complete).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /edit provider/i }));
    fireEvent.click(screen.getByRole("button", { name: /switch provider/i }));
    fireEvent.click(screen.getByRole("button", { name: /skip setup/i }));
    expect(onEditProvider).toHaveBeenCalled();
    expect(onSwitchProvider).toHaveBeenCalled();
    expect(onSkip).toHaveBeenCalled();
  });

  it.each(["auth", "authentication_failed", "provider_api_key_invalid"])(
    "preserves auth recovery copy for %s first chat failure aliases",
    async (failureCategory) => {
      const verifyFirstChat = vi.fn().mockResolvedValue({
        status: "failed",
        provider: "openai",
        model: "gpt-4.1-mini",
        failure_category: failureCategory,
        message: "Invalid API key",
      });

      render(
        <FirstChatStep
          provider="openai"
          model="gpt-4.1-mini"
          verifyFirstChat={verifyFirstChat}
          complete={vi.fn()}
          onComplete={vi.fn()}
          onBack={vi.fn()}
          onEditProvider={vi.fn()}
          onSwitchProvider={vi.fn()}
          onSkip={vi.fn()}
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

      await screen.findByText(/invalid api key/i);
      expect(screen.getByRole("alert")).toHaveTextContent(
        /credentials need attention/i,
      );
      expect(screen.getByRole("alert")).toHaveTextContent(
        /update the provider credentials/i,
      );
      expect(
        screen.getByRole("button", { name: /edit provider/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /switch provider/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /skip setup/i }),
      ).toBeInTheDocument();
    },
  );

  it("shows endpoint recovery when the first chat endpoint is unreachable", async () => {
    const verifyFirstChat = vi.fn().mockResolvedValue({
      status: "failed",
      provider: "ollama",
      model: "llama3.1",
      failure_category: "local_provider_unreachable",
      message: "Connection refused",
    });
    const onCheckEndpoint = vi.fn();

    render(
      <FirstChatStep
        provider="ollama"
        model="llama3.1"
        verifyFirstChat={verifyFirstChat}
        complete={vi.fn()}
        onComplete={vi.fn()}
        onBack={vi.fn()}
        onEditProvider={vi.fn()}
        onSwitchProvider={vi.fn()}
        onSkip={vi.fn()}
        onCheckEndpoint={onCheckEndpoint}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText(/connection refused/i);
    expect(
      screen.getByText(/check the local endpoint URL and API compatibility/i),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /check endpoint/i }));
    expect(onCheckEndpoint).toHaveBeenCalled();
  });

  it.each([
    {
      failureCategory: "rate_limited",
      message: "The provider rate limit was reached.",
      expectedCopy: /provider limit reached/i,
      expectedCategory: /category: quota_or_rate_limit/i,
    },
    {
      failureCategory: "network_error",
      message: "The provider could not be reached.",
      expectedCopy: /endpoint could not be reached/i,
      expectedCategory: /category: endpoint_unreachable/i,
    },
    {
      failureCategory: "provider_error",
      message: "The provider returned an error.",
      expectedCopy: /provider returned an error/i,
      expectedCategory: /category: provider_error/i,
    },
    {
      failureCategory: "configuration_error",
      message: "Provider configuration is incomplete.",
      expectedCopy: /provider configuration needs attention/i,
      expectedCategory: /category: configuration_error/i,
    },
    {
      failureCategory: "empty_response",
      message: "The provider returned an empty chat response.",
      expectedCopy: /provider returned an empty response/i,
      expectedCategory: /category: empty_response/i,
    },
  ])(
    "maps backend first-chat failure category $failureCategory to recovery copy",
    async ({ failureCategory, message, expectedCopy, expectedCategory }) => {
      const verifyFirstChat = vi.fn().mockResolvedValue({
        status: "failed",
        provider: "openai",
        model: "gpt-4.1-mini",
        failure_category: failureCategory,
        message,
      });

      render(
        <FirstChatStep
          provider="openai"
          model="gpt-4.1-mini"
          verifyFirstChat={verifyFirstChat}
          complete={vi.fn()}
          onComplete={vi.fn()}
          onBack={vi.fn()}
          onEditProvider={vi.fn()}
          onSwitchProvider={vi.fn()}
          onSkip={vi.fn()}
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

      await screen.findByText(message);
      expect(screen.getByRole("alert")).toHaveTextContent(expectedCopy);
      expect(screen.getByRole("alert")).toHaveTextContent(expectedCategory);
    },
  );

  it("shows endpoint diagnostics for backend request_invalid first-chat failures", async () => {
    const verifyFirstChat = vi.fn().mockResolvedValue({
      status: "failed",
      provider: "openai_compatible",
      model: "local-model",
      failure_category: "request_invalid",
      message: "The provider rejected the first-chat request.",
    });
    const onCheckEndpoint = vi.fn();

    render(
      <FirstChatStep
        provider="openai_compatible"
        model="local-model"
        verifyFirstChat={verifyFirstChat}
        complete={vi.fn()}
        onComplete={vi.fn()}
        onBack={vi.fn()}
        onEditProvider={vi.fn()}
        onSwitchProvider={vi.fn()}
        onSkip={vi.fn()}
        onCheckEndpoint={onCheckEndpoint}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText(/the provider rejected the first-chat request/i);
    expect(screen.getByRole("alert")).toHaveTextContent(
      /endpoint API shape is unsupported/i,
    );
    fireEvent.click(screen.getByRole("button", { name: /check endpoint/i }));
    expect(onCheckEndpoint).toHaveBeenCalled();
  });

  it("keeps the failed attempt visible while retrying until the new result arrives", async () => {
    let resolveRetry!: (value: {
      status: string;
      provider: string;
      model: string;
      response_text: string;
    }) => void;
    const retryPromise = new Promise<{
      status: string;
      provider: string;
      model: string;
      response_text: string;
    }>((resolve) => {
      resolveRetry = resolve;
    });
    const verifyFirstChat = vi
      .fn()
      .mockResolvedValueOnce({
        status: "failed",
        provider: "openai",
        model: "gpt-4.1-mini",
        failure_category: "auth",
        message: "Invalid API key",
      })
      .mockReturnValueOnce(retryPromise);

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={vi.fn().mockResolvedValue({
          success: true,
          message: "completed",
          requires_restart: false,
          install_plan_submitted: false,
        })}
        onComplete={vi.fn()}
        onBack={vi.fn()}
        onEditProvider={vi.fn()}
        onSwitchProvider={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));
    await screen.findByText(/invalid api key/i);

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    expect(verifyFirstChat).toHaveBeenCalledTimes(2);
    expect(screen.getByText(/invalid api key/i)).toBeInTheDocument();

    resolveRetry({
      status: "ready",
      provider: "openai",
      model: "gpt-4.1-mini",
      response_text: "Hello again.",
    });
    await screen.findByText("Hello again.");
    expect(screen.queryByText(/invalid api key/i)).not.toBeInTheDocument();
  });

  it("does not keep stale categorized copy when retry throws before a response", async () => {
    const verifyFirstChat = vi
      .fn()
      .mockResolvedValueOnce({
        status: "failed",
        provider: "openai",
        model: "gpt-4.1-mini",
        failure_category: "auth_failed",
        message: "Invalid API key",
      })
      .mockRejectedValueOnce(new Error("Network request failed before response"));

    render(
      <FirstChatStep
        provider="openai"
        model="gpt-4.1-mini"
        verifyFirstChat={verifyFirstChat}
        complete={vi.fn()}
        onComplete={vi.fn()}
        onBack={vi.fn()}
        onEditProvider={vi.fn()}
        onSwitchProvider={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));
    await screen.findByText(/invalid api key/i);
    expect(screen.getByRole("alert")).toHaveTextContent(
      /credentials need attention/i,
    );

    fireEvent.click(screen.getByRole("button", { name: /retry/i }));

    await screen.findByText(/network request failed before response/i);
    expect(screen.getByRole("alert")).toHaveTextContent(
      /first chat did not complete/i,
    );
    expect(screen.getByRole("alert")).not.toHaveTextContent(
      /credentials need attention/i,
    );
  });

  it("reports completion state errors separately from first chat verification", async () => {
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
        onEditProvider={vi.fn()}
        onSwitchProvider={vi.fn()}
        onSkip={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /send test chat/i }));

    await screen.findByText("Hello.");
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /setup state could not be completed/i,
    );
    expect(screen.getByRole("alert")).not.toHaveTextContent(
      /first chat failed/i,
    );
    expect(onComplete).not.toHaveBeenCalled();
  });
});
