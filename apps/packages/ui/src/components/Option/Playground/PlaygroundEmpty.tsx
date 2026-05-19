import React from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import {
  MessageSquarePlus,
  HelpCircle,
  Sparkles,
  GitBranch,
  UserCircle2,
  Search,
  Microscope,
} from "lucide-react";
import { useDemoMode } from "@/context/demo-mode";
import { useIsConnected } from "@/hooks/useConnectionState";
import { useHelpModal } from "@/store/tutorials";
import { buildResearchLaunchPath } from "@/routes/route-paths";
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open";
import { EmptyState } from "@/components/ui/feedback";

const actionButtonFocusClassName =
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg";

export const PlaygroundEmpty = () => {
  const { t } = useTranslation(["playground", "common", "option"]);
  const { demoEnabled } = useDemoMode();
  const isConnected = useIsConnected();
  const { open: openHelpModal } = useHelpModal();
  const navigate = useNavigate();

  const dispatchStarter = React.useCallback(
    (mode: "general" | "compare" | "character" | "rag", prompt?: string) => {
      window.dispatchEvent(
        new CustomEvent("tldw:playground-starter-selected", {
          detail: { mode },
        }),
      );
      window.dispatchEvent(
        new CustomEvent("tldw:playground-starter", {
          detail: {
            mode,
            prompt,
          },
        }),
      );
    },
    [],
  );

  const handleStartChat = React.useCallback(() => {
    dispatchStarter("general");
    window.dispatchEvent(new CustomEvent("tldw:focus-composer"));
  }, [dispatchStarter]);

  const handleOpenQuickIngest = React.useCallback(() => {
    if (typeof window === "undefined") return;
    const trigger = document.querySelector<HTMLButtonElement>(
      '[data-testid="open-quick-ingest"]',
    );
    if (trigger) {
      trigger.click();
      return;
    }
    requestQuickIngestOpen();
  }, []);

  const handleOpenDeepResearch = React.useCallback(() => {
    navigate(buildResearchLaunchPath());
  }, [navigate]);

  const starterCards = React.useMemo(
    () => [
      {
        key: "general",
        icon: Sparkles,
        title: t("playground:empty.starterGeneralTitle", "General chat"),
        description: t(
          "playground:empty.starterGeneralBody",
          "Start with a single model and ask anything.",
        ),
        action: () => dispatchStarter("general"),
      },
      {
        key: "compare",
        icon: GitBranch,
        title: t(
          "playground:empty.starterCompareTitle",
          "Compare AI models side-by-side",
        ),
        description: t(
          "playground:empty.starterCompareBody",
          "Send the same question to multiple models and compare their answers.",
        ),
        action: () => dispatchStarter("compare"),
      },
      {
        key: "character",
        icon: UserCircle2,
        title: t(
          "playground:empty.starterCharacterTitle",
          "Character chat / role-play",
        ),
        description: t(
          "playground:empty.starterCharacterBody",
          "Pick a character and keep role-play setup visible while you chat.",
        ),
        action: () => dispatchStarter("character"),
      },
      {
        key: "rag",
        icon: Search,
        title: t(
          "playground:empty.starterKnowledgeTitle",
          "Search your documents",
        ),
        description: t(
          "playground:empty.starterKnowledgeBody",
          "Ask questions about your ingested content with cited answers.",
        ),
        action: () => dispatchStarter("rag"),
      },
      {
        key: "research",
        icon: Microscope,
        title: t("playground:empty.starterResearchTitle", "Deep research"),
        description: t(
          "playground:empty.starterResearchBody",
          "Run a thorough, multi-step investigation with citations and checkpoints.",
        ),
        action: handleOpenDeepResearch,
      },
    ],
    [dispatchStarter, handleOpenDeepResearch, t],
  );

  const isDisconnected = !demoEnabled && !isConnected;
  const description = demoEnabled ? (
    t("playground:empty.demoDescription", {
      defaultValue:
        "You're in demo mode — try asking a question to see how the assistant responds. You can connect your own tldw server later.",
    })
  ) : isDisconnected ? (
    <div className="mx-auto flex max-w-2xl flex-col items-center gap-2 sm:text-base">
      <span>
        {t("playground:empty.disconnectedDescription", {
          defaultValue: "Connect to a tldw server to start chatting.",
        })}
      </span>
      <button
        type="button"
        onClick={() => navigate("/settings/tldw")}
        className={`inline-flex items-center text-sm font-medium text-primary transition hover:underline ${actionButtonFocusClassName}`}
      >
        {t("playground:empty.openSettings", {
          defaultValue: "Open Settings",
        })}
      </button>
    </div>
  ) : (
    t("playground:empty.description", {
      defaultValue:
        "Experiment with different models, prompts, and knowledge sources here.",
    })
  );

  return (
    <div className="mx-auto mt-10 max-w-5xl px-4">
      <EmptyState
        data-testid="playground-empty-shell"
        icon={MessageSquarePlus}
        title={t("playground:empty.title", {
          defaultValue: "Start a new chat",
        })}
        description={description}
        primaryAction={{
          label: t("playground:empty.primaryCta", {
            defaultValue: "Start chatting",
          }),
          onClick: handleStartChat,
        }}
        secondaryAction={{
          label: t("option:header.quickIngest", "Quick Ingest"),
          onClick: handleOpenQuickIngest,
        }}
        variant="card"
        size="lg"
        className="max-w-4xl rounded-[28px] p-5 text-sm sm:p-7"
      >
        <div className="flex flex-col gap-6">
          <div className="border-t border-border/60 pt-5">
            <div className="flex flex-col gap-1 text-center sm:text-left">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-text-muted">
                {t("playground:empty.modeTitle", "Choose a mode")}
              </p>
              <p className="text-sm text-text-muted">
                {t(
                  "playground:empty.modeBody",
                  "Pick any starting point. All five stay equally available from the first screen.",
                )}
              </p>
            </div>

            <div
              data-testid="playground-empty-mode-deck"
              className="mt-4 grid gap-3 sm:grid-cols-2"
            >
              {starterCards.map((starter) => {
                const Icon = starter.icon;
                return (
                  <button
                    key={starter.key}
                    type="button"
                    onClick={starter.action}
                    className={`group flex min-h-[118px] flex-col rounded-2xl border border-border/60 bg-bg/20 px-4 py-4 text-left transition hover:border-primary/40 hover:bg-surface2/60 ${actionButtonFocusClassName}`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="rounded-xl border border-border/50 bg-surface2/70 p-2 text-text-muted transition group-hover:text-text">
                        <Icon className="h-4 w-4" aria-hidden="true" />
                      </div>
                      <div className="space-y-1">
                        <span className="block text-base font-semibold text-text">
                          {starter.title}
                        </span>
                        <p className="text-sm leading-6 text-text-muted">
                          {starter.description}
                        </p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="border-t border-border/60 pt-4">
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-xs text-text-muted">
                <span>
                  {t(
                    "playground:empty.tipSlash",
                    "Type / for commands like /search or /web",
                  )}
                </span>
                <span>
                  {t(
                    "playground:empty.tipPrompt",
                    "Set a system prompt to customize AI behavior",
                  )}
                </span>
              </div>

              <button
                type="button"
                onClick={openHelpModal}
                className={`inline-flex items-center gap-1.5 text-xs font-medium text-primary transition hover:underline ${actionButtonFocusClassName}`}
              >
                <HelpCircle className="h-3.5 w-3.5" />
                {t("playground:empty.takeTour", "Take a quick tour")}
              </button>
            </div>
          </div>
        </div>
      </EmptyState>
    </div>
  );
};
