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
  onEditProvider: () => void;
  onSwitchProvider: () => void;
  onSkip: () => void;
  skipPending?: boolean;
  onCheckEndpoint?: () => void;
};

const DEFAULT_FIRST_PROMPT = "Say hello in one short sentence.";

type FirstChatRecoveryCategory =
  | "auth_failed"
  | "quota_or_rate_limit"
  | "endpoint_unreachable"
  | "unsupported_api_shape"
  | "model_unavailable"
  | "configuration_error"
  | "provider_error"
  | "empty_response"
  | "config_write_failed"
  | "provider_unvalidated"
  | "unknown";

const FIRST_CHAT_CATEGORY_ALIASES: Record<string, FirstChatRecoveryCategory> = {
  auth: "auth_failed",
  auth_failed: "auth_failed",
  authentication_failed: "auth_failed",
  provider_api_key_invalid: "auth_failed",
  rate_limit: "quota_or_rate_limit",
  rate_limited: "quota_or_rate_limit",
  quota: "quota_or_rate_limit",
  quota_exceeded: "quota_or_rate_limit",
  connection_error: "endpoint_unreachable",
  network_error: "endpoint_unreachable",
  timeout: "endpoint_unreachable",
  timeout_error: "endpoint_unreachable",
  local_provider_unreachable: "endpoint_unreachable",
  endpoint_unreachable: "endpoint_unreachable",
  bad_request: "unsupported_api_shape",
  invalid_request: "unsupported_api_shape",
  request_invalid: "unsupported_api_shape",
  unsupported_api_shape: "unsupported_api_shape",
  model_not_found: "model_unavailable",
  model_unavailable: "model_unavailable",
  configuration_error: "configuration_error",
  provider_error: "provider_error",
  server_error: "provider_error",
  upstream_error: "provider_error",
  empty_response: "empty_response",
  provider_unvalidated: "provider_unvalidated",
  config_write_failed: "config_write_failed",
};

const normalizeRecoveryCategory = (
  category?: string | null,
): FirstChatRecoveryCategory => {
  if (!category) return "unknown";
  return FIRST_CHAT_CATEGORY_ALIASES[category] ?? "unknown";
};

const RECOVERY_COPY: Record<
  FirstChatRecoveryCategory,
  { title: string; guidance: string }
> = {
  auth_failed: {
    title: "Credentials need attention",
    guidance:
      "Update the provider credentials, then retry the first chat, or switch to another provider.",
  },
  quota_or_rate_limit: {
    title: "Provider limit reached",
    guidance:
      "Retry later if this is temporary, or switch provider to keep setup moving.",
  },
  endpoint_unreachable: {
    title: "Endpoint could not be reached",
    guidance:
      "Check the local endpoint URL and API compatibility, then retry the first chat.",
  },
  unsupported_api_shape: {
    title: "Endpoint API shape is unsupported",
    guidance:
      "Check the local endpoint URL and API compatibility, then retry with an OpenAI-compatible chat endpoint.",
  },
  model_unavailable: {
    title: "Model is unavailable",
    guidance:
      "Switch model or provider, or edit the provider details before retrying.",
  },
  configuration_error: {
    title: "Provider configuration needs attention",
    guidance:
      "Return to provider setup, confirm the saved provider details, then retry the first chat.",
  },
  provider_error: {
    title: "Provider returned an error",
    guidance:
      "Retry if this looks temporary, or switch provider to keep setup moving.",
  },
  empty_response: {
    title: "Provider returned an empty response",
    guidance:
      "Retry the first chat. If it happens again, switch model or provider.",
  },
  config_write_failed: {
    title: "Configuration could not be saved",
    guidance:
      "Edit provider setup to confirm the saved configuration, then retry the first chat.",
  },
  provider_unvalidated: {
    title: "Provider needs validation",
    guidance:
      "Return to provider setup, validate the provider, then retry the first chat.",
  },
  unknown: {
    title: "First chat did not complete",
    guidance:
      "Retry the first chat, edit provider settings, or skip setup and diagnose from settings later.",
  },
};

const shouldShowCheckEndpoint = (category: FirstChatRecoveryCategory) =>
  category === "endpoint_unreachable" || category === "unsupported_api_shape";

export function FirstChatStep({
  provider,
  model,
  verifyFirstChat,
  complete,
  onComplete,
  onBack,
  onEditProvider,
  onSwitchProvider,
  onSkip,
  skipPending = false,
  onCheckEndpoint,
}: FirstChatStepProps) {
  const [prompt, setPrompt] = React.useState(DEFAULT_FIRST_PROMPT);
  const [response, setResponse] =
    React.useState<FirstChatVerifyResponse | null>(null);
  const [running, setRunning] = React.useState(false);
  const [verificationError, setVerificationError] = React.useState<
    string | null
  >(null);
  const [requestFailureCategory, setRequestFailureCategory] =
    React.useState<FirstChatRecoveryCategory | null>(null);
  const [completionError, setCompletionError] = React.useState<string | null>(
    null,
  );

  const handleSend = async () => {
    setRunning(true);
    setResponse(null);
    setVerificationError(null);
    setRequestFailureCategory(null);
    setCompletionError(null);
    try {
      let verification: FirstChatVerifyResponse;
      try {
        verification = await verifyFirstChat({
          provider,
          model,
          prompt,
        });
      } catch (err) {
        setRequestFailureCategory("unknown");
        setVerificationError(
          err instanceof Error
            ? err.message
            : "First chat request failed. Retry, edit provider, or skip setup.",
        );
        return;
      }
      setResponse(verification);
      setRequestFailureCategory(null);
      setVerificationError(null);
      if (verification.status !== "ready") {
        setVerificationError(
          verification.message || "First chat did not complete.",
        );
        return;
      }
      try {
        await complete({
          acknowledged_steps: ["first_chat"],
        });
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Try again.";
        setCompletionError(
          `Setup state could not be completed after the first chat succeeded. ${detail}`,
        );
        return;
      }
      onComplete();
    } catch (err) {
      setRequestFailureCategory("unknown");
      setVerificationError(
        err instanceof Error ? err.message : "First chat failed.",
      );
    } finally {
      setRunning(false);
    }
  };

  const activeFailureCategory =
    requestFailureCategory ?? response?.failure_category ?? null;
  const recoveryCategory = normalizeRecoveryCategory(activeFailureCategory);
  const recoveryCopy = RECOVERY_COPY[recoveryCategory];
  const hasVerificationFailure =
    Boolean(verificationError) ||
    Boolean(response && response.status !== "ready");
  const failureMessage = verificationError ?? response?.message ?? null;
  const showCheckEndpoint =
    Boolean(onCheckEndpoint) && shouldShowCheckEndpoint(recoveryCategory);

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

      {hasVerificationFailure ? (
        <div
          role="alert"
          className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="font-medium">{recoveryCopy.title}</p>
              <p className="mt-1 text-text-muted">{recoveryCopy.guidance}</p>
              {failureMessage ? (
                <p className="mt-2">{failureMessage}</p>
              ) : null}
              {activeFailureCategory ? (
                <p className="mt-2 font-mono text-xs text-text-muted">
                  Category: {recoveryCategory}
                </p>
              ) : null}
            </div>
            {running ? (
              <span className="rounded-full border border-border bg-surface px-2 py-0.5 text-xs font-medium text-text-muted">
                Retrying
              </span>
            ) : null}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleSend}
              disabled={running || !provider || !model}
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
            >
              Retry
            </button>
            <button
              type="button"
              onClick={onEditProvider}
              disabled={running}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
            >
              Edit provider
            </button>
            <button
              type="button"
              onClick={onSwitchProvider}
              disabled={running}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
            >
              Switch provider
            </button>
            {showCheckEndpoint ? (
              <button
                type="button"
                onClick={onCheckEndpoint}
                disabled={running}
                className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
              >
                Check endpoint
              </button>
            ) : null}
            <button
              type="button"
              onClick={onSkip}
              disabled={running || skipPending}
              className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-text hover:bg-surface2 disabled:opacity-50"
            >
              {skipPending ? "Skipping..." : "Skip setup"}
            </button>
          </div>
        </div>
      ) : null}

      {completionError ? (
        <div
          role="alert"
          className="rounded-md border border-danger/40 bg-danger/10 px-4 py-3 text-sm text-text"
        >
          {completionError}
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
