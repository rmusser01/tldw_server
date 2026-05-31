import React from "react";
import { useTranslation } from "react-i18next";
import { browser } from "wxt/browser";
import {
  useConnectionState,
  useConnectionUxState,
} from "@/hooks/useConnectionState";
import { ConnectionPhase } from "@/types/connection";
import { cleanUrl } from "@/libs/clean-url";
import {
  MessageSquare,
  Wifi,
  Sparkles,
  FileText,
  Search,
  BookOpen,
} from "lucide-react";
import { RecoveryCallout, type RecoveryState } from "@/components/ui/state";

type EmptySidePanelProps = {
  inputRef?: React.RefObject<HTMLTextAreaElement>;
};

export const EmptySidePanel = ({ inputRef }: EmptySidePanelProps) => {
  const { t } = useTranslation([
    "sidepanel",
    "settings",
    "option",
    "playground",
  ]);
  const { phase, isConnected, serverUrl } = useConnectionState();
  const { uxState, mode, configStep, hasCompletedFirstRun } =
    useConnectionUxState();
  const isConnectionReady = isConnected && phase === ConnectionPhase.CONNECTED;
  const primaryButtonRef = React.useRef<HTMLButtonElement | null>(null);

  const openExtensionUrl = (
    path: `/options.html${string}` | `/sidepanel.html${string}`,
  ) => {
    try {
      // Prefer opening the extension's options.html directly so users land
      // on the tldw settings page instead of the generic extensions manager.
      // `browser` is provided by the WebExtension polyfill in WXT.
      if (browser?.runtime?.getURL) {
        const url = browser.runtime.getURL(path);
        if (browser.tabs?.create) {
          browser.tabs.create({ url });
        } else {
          window.open(url, "_blank");
        }
        return;
      }
    } catch (err) {
      // Fall through to chrome.* / window.open below.
      console.debug(
        "[EmptySidePanel] openExtensionUrl browser API unavailable:",
        err,
      );
    }

    try {
      if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
        const url = chrome.runtime.getURL(path);
        window.open(url, "_blank");
        return;
      }
      if (
        typeof chrome !== "undefined" &&
        chrome.runtime?.openOptionsPage &&
        path.includes("/options.html")
      ) {
        chrome.runtime.openOptionsPage();
        return;
      }
    } catch (err) {
      // ignore and fall back to plain window.open
      console.debug(
        "[EmptySidePanel] openExtensionUrl chrome API unavailable:",
        err,
      );
    }

    window.open(path, "_blank");
  };

  const showConnectionCard = !isConnectionReady;
  const host = serverUrl ? cleanUrl(serverUrl) : "tldw_server";

  const activeStep = (() => {
    if (configStep === "url") return 1;
    if (configStep === "auth") return 2;
    if (configStep === "health") return 3;
    if (uxState === "configuring_url" || uxState === "unconfigured") return 1;
    if (uxState === "configuring_auth") return 2;
    if (
      uxState === "testing" ||
      uxState === "connected_ok" ||
      uxState === "connected_degraded" ||
      uxState === "error_auth" ||
      uxState === "error_unreachable"
    ) {
      return 3;
    }
    return 1;
  })();

  const stepSummary = (() => {
    if (hasCompletedFirstRun) return null;
    if (activeStep === 1) {
      return t(
        "sidepanel:firstRun.step1",
        "Step 1 of 3 — Add your server URL in Options → tldw Server.",
      );
    }
    if (activeStep === 2) {
      return t(
        "sidepanel:firstRun.step2",
        "Step 2 of 3 — Add your API key or log in on the tldw Server settings page.",
      );
    }
    return t(
      "sidepanel:firstRun.step3",
      "Step 3 of 3 — Check connection & Knowledge in Health & diagnostics.",
    );
  })();

  const openOnboarding = () => {
    openExtensionUrl("/options.html#/");
  };

  const bannerHeading = (() => {
    if (uxState === "error_auth") {
      return t(
        "option:connectionCard.headlineErrorAuth",
        "API key needs attention",
      );
    }
    if (uxState === "error_unreachable") {
      return t(
        "option:connectionCard.headlineError",
        "Can’t reach your tldw server",
      );
    }
    // First-run emphasis: make it clear that setup must be finished
    // before chat is available from the sidepanel.
    return hasCompletedFirstRun
      ? t(
          "option:connectionCard.headlineMissing",
          "Connect tldw Assistant to your server to open Companion Home",
        )
      : t("sidepanel:firstRun.headline", "Finish setup to open Companion Home");
  })();

  const bannerBody = (() => {
    if (uxState === "error_auth") {
      return t(
        "option:connectionCard.descriptionErrorAuth",
        "Your server is up but the API key is wrong or missing. Fix the key in Settings → tldw server, then retry.",
      );
    }
    if (uxState === "error_unreachable") {
      return t(
        "option:connectionCard.descriptionError",
        "We couldn’t reach {{host}}. Check that your tldw_server is running and that your browser can reach it, then open diagnostics or update the URL.",
        { host },
      );
    }
    if (!hasCompletedFirstRun) {
      return t(
        "sidepanel:firstRun.description",
        "Before you can use Companion Home here, finish the short setup flow in Options to connect tldw Assistant to your tldw server or choose demo mode.",
      );
    }
    return t(
      "option:connectionCard.descriptionMissing",
      "tldw_server is your private AI workspace that keeps your home dashboard, chats, notes, and media on your own machine. Add your server URL to get started.",
    );
  })();

  const insertPrompt = (text: string) => {
    const input =
      inputRef?.current ??
      document.querySelector<HTMLTextAreaElement>('[data-testid="chat-input"]');
    if (input) {
      input.value = text;
      input.focus();
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  };

  React.useEffect(() => {
    if (!showConnectionCard) return;
    if (hasCompletedFirstRun) return;
    try {
      primaryButtonRef.current?.focus();
    } catch {
      // ignore focus failures
    }
  }, [showConnectionCard, hasCompletedFirstRun]);

  if (showConnectionCard) {
    const recoveryState: RecoveryState =
      uxState === "error_auth"
        ? "auth_required"
        : uxState === "error_unreachable"
          ? "unavailable"
          : "setup_required";
    const primaryLabel = !hasCompletedFirstRun
      ? t("sidepanel:firstRun.finishSetup", "Finish setup")
      : uxState === "error_auth" || uxState === "error_unreachable"
        ? t("sidepanel:firstRun.reviewSettings", "Review settings")
        : t("sidepanel:firstRun.openOptionsPrimary", "Open tldw Settings");
    const stepDiagnostics = stepSummary
      ? [
          {
            label: t("sidepanel:firstRun.stepLabel", "Setup step"),
            value: stepSummary,
          },
        ]
      : undefined;

    return (
      <div className="mt-5 flex w-full flex-col items-stretch gap-3 px-4">
        <RecoveryCallout
          state={recoveryState}
          title={bannerHeading}
          message={bannerBody}
          diagnostics={stepDiagnostics}
          primaryAction={{
            label: primaryLabel,
            onClick: openOnboarding,
            ref: primaryButtonRef,
            "data-testid": "chat-connection-cta",
          }}
          data-testid="chat-empty-connection"
          className="rounded-2xl border-border/70 bg-surface/95"
        />

        {/* Quick tips for first-time users */}
        {!hasCompletedFirstRun && (
          <div className="text-center text-label text-text-muted">
            {t(
              "sidepanel:firstRun.quickTip",
              "Need help? Check our docs or try demo mode.",
            )}
          </div>
        )}
      </div>
    );
  }

  // Connected state: show welcoming empty chat guidance
  return (
    <div
      className="mt-4 flex w-full flex-col items-center justify-center px-4"
      data-testid="chat-empty-connected"
    >
      {/* Animated icon */}
      <div className="relative mb-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-success/10">
          <MessageSquare className="size-6 text-success" />
        </div>
        <div className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-bg bg-success" />
      </div>

      {/* Status and heading */}
      <div className="mb-1 flex items-center gap-1 text-xs font-medium text-success">
        <Wifi className="size-3" />
        {t("sidepanel:emptyChat.connected", "Connected")}
      </div>
      <p className="mb-4 text-sm font-medium text-text">
        {mode === "demo"
          ? t(
              "sidepanel:emptyChat.demoHint",
              "Demo mode — try sending a message",
            )
          : t("sidepanel:emptyChat.hint", "Start a conversation below")}
      </p>

      {/* Suggestion cards */}
      <div className="w-full space-y-2 text-left">
        <p className="px-1 text-label font-medium uppercase tracking-wide text-text-muted">
          {t("sidepanel:emptyChat.suggestions", "Try asking")}
        </p>
        <button
          type="button"
          data-testid="chat-suggestion-1"
          className="group flex w-full items-center gap-3 rounded-2xl border border-border/70 bg-surface px-3 py-2 text-left transition-colors hover:border-primary/40 hover:bg-surface2"
          title={t(
            "sidepanel:emptyChat.examplePrompt1",
            '"Summarize this page"',
          )}
          onClick={() =>
            insertPrompt(
              t(
                "sidepanel:emptyChat.examplePrompt1Text",
                "Summarize this page",
              ),
            )
          }
        >
          <FileText className="size-4 text-text-subtle group-hover:text-primary flex-shrink-0" />
          <span className="text-xs text-text-muted">
            {t("sidepanel:emptyChat.examplePrompt1", '"Summarize this page"')}
          </span>
        </button>
        <button
          type="button"
          data-testid="chat-suggestion-2"
          className="group flex w-full items-center gap-3 rounded-2xl border border-border/70 bg-surface px-3 py-2 text-left transition-colors hover:border-primary/40 hover:bg-surface2"
          title={t(
            "sidepanel:emptyChat.examplePrompt2",
            '"What are the key points?"',
          )}
          onClick={() =>
            insertPrompt(
              t(
                "sidepanel:emptyChat.examplePrompt2Text",
                "What are the key points?",
              ),
            )
          }
        >
          <Search className="size-4 text-text-subtle group-hover:text-primary flex-shrink-0" />
          <span className="text-xs text-text-muted">
            {t(
              "sidepanel:emptyChat.examplePrompt2",
              '"What are the key points?"',
            )}
          </span>
        </button>
        <button
          type="button"
          data-testid="chat-suggestion-3"
          className="group flex w-full items-center gap-3 rounded-2xl border border-border/70 bg-surface px-3 py-2 text-left transition-colors hover:border-primary/40 hover:bg-surface2"
          title={t(
            "sidepanel:emptyChat.examplePrompt3",
            '"Explain this in simple terms"',
          )}
          onClick={() =>
            insertPrompt(
              t(
                "sidepanel:emptyChat.examplePrompt3Text",
                "Explain this in simple terms",
              ),
            )
          }
        >
          <BookOpen className="size-4 text-text-subtle group-hover:text-primary flex-shrink-0" />
          <span className="text-xs text-text-muted">
            {t(
              "sidepanel:emptyChat.examplePrompt3",
              '"Explain this in simple terms"',
            )}
          </span>
        </button>
      </div>

      {/* Demo mode indicator with limitations */}
      {mode === "demo" && (
        <div className="mt-4 rounded-2xl border border-purple-200/70 bg-purple-50/70 px-3 py-2 dark:border-purple-800 dark:bg-purple-900/20">
          <div className="flex items-center gap-2 text-xs font-medium text-purple-700 dark:text-purple-300">
            <Sparkles className="size-3.5" />
            {t("sidepanel:emptyChat.demoIndicator", "Running in demo mode")}
          </div>
          <p className="mt-1 text-[10px] text-purple-600 dark:text-purple-400">
            {t(
              "sidepanel:emptyChat.demoLimitations",
              "Some features are limited. Connect to tldw_server for full functionality.",
            )}
          </p>
        </div>
      )}
    </div>
  );
};
