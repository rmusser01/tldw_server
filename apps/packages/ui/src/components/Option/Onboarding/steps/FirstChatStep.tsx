import React from "react";
import { MessageCircle } from "lucide-react";

import type {
  FirstChatVerifyRequest,
  FirstChatVerifyResponse,
  FirstRunCompleteRequest,
  SetupCompleteResponse,
} from "@/types/setup-onboarding";

type FirstChatStepProps = {
  provider: string;
  model: string;
  verifyFirstChat: (
    payload: FirstChatVerifyRequest,
  ) => Promise<FirstChatVerifyResponse>;
  complete: (
    payload?: FirstRunCompleteRequest,
  ) => Promise<SetupCompleteResponse>;
  onComplete: () => void;
  onBack: () => void;
};

const DEFAULT_FIRST_PROMPT = "Say hello in one short sentence.";

export function FirstChatStep({
  provider,
  model,
  verifyFirstChat,
  complete,
  onComplete,
  onBack,
}: FirstChatStepProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_FIRST_PROMPT);
  const [response, setResponse] =
    React.useState<FirstChatVerifyResponse | null>(null);
  const [running, setRunning] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const handleSend = async () => {
    setRunning(true);
    setError(null);
    setResponse(null);
    try {
      let verification: FirstChatVerifyResponse;
      try {
        verification = await verifyFirstChat({
          provider,
          model,
          prompt,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : "First chat failed.");
        return;
      }
      setResponse(verification);
      if (verification.status !== "ready") {
        setError(verification.message || "First chat did not complete.");
        return;
      }
      try {
        await complete({
          acknowledged_steps: ["first_chat"],
        });
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Try again.";
        setError(
          `Setup completion failed after the first chat succeeded. ${detail}`,
        );
        return;
      }
      onComplete();
    } catch (err) {
      setError(err instanceof Error ? err.message : "First chat failed.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <section aria-labelledby="first-chat-title" className="space-y-5">
      <div className="flex items-start gap-3">
        <span className="inline-flex size-10 items-center justify-center rounded-md bg-surface2 text-primary">
          <MessageCircle className="size-5" aria-hidden="true" />
        </span>
        <div>
          <h2 id="first-chat-title" className="text-lg font-semibold text-text">
            First chat
          </h2>
          <p className="mt-1 text-sm text-text-muted">
            Completion requires an actual successful response from {provider}.
          </p>
        </div>
      </div>

      <div className="rounded-md border border-border bg-surface px-4 py-3 text-sm text-text">
        <div className="font-medium">Default chat target</div>
        <div className="mt-1 text-text-muted">
          {provider} / {model}
        </div>
      </div>

      <label className="block text-sm font-medium text-text">
        <span>First prompt</span>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.currentTarget.value)}
          rows={3}
          className="mt-1 w-full rounded-md border border-border bg-bg px-3 py-2 text-sm text-text"
        />
      </label>

      {response?.response_text ? (
        <div className="rounded-md border border-success/40 bg-success/10 px-4 py-3 text-sm text-text">
          {response.response_text}
        </div>
      ) : null}

      {error ? (
        <div
          role="alert"
          className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {response?.failure_category ? (
            <span className="font-medium">{response.failure_category}: </span>
          ) : null}
          {error}
        </div>
      ) : null}

      <div className="flex flex-wrap justify-between gap-2">
        <button
          type="button"
          onClick={onBack}
          disabled={running}
          className="rounded-md border border-border bg-surface px-3 py-2 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
        >
          Back to providers
        </button>
        <button
          type="button"
          onClick={handleSend}
          disabled={running || !provider || !model}
          className="rounded-md bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
        >
          {running ? "Sending..." : "Send test chat"}
        </button>
      </div>
    </section>
  );
}
