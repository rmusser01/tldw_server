import React from "react"
import { Spin } from "antd"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router-dom"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useDemoMode } from "@/context/demo-mode"
import { useScrollToServerCard } from "@/hooks/useScrollToServerCard"
import {
  useConnectionActions,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import FeatureEmptyState from "@/components/Common/FeatureEmptyState"
import ConnectionProblemBanner from "@/components/Common/ConnectionProblemBanner"
import { StatusBadge } from "@/components/Common/StatusBadge"
import { getDesignSystemState } from "@/design-system"
import { getDemoFlashcardDecks } from "@/utils/demo-content"
const FlipCardDemo = React.lazy(() =>
  import("./components/FlipCardDemo").then((m) => ({ default: m.FlipCardDemo }))
)
const FlashcardsManager = React.lazy(() =>
  import("./FlashcardsManager").then((m) => ({ default: m.FlashcardsManager }))
)

const FLASHCARD_STUDY_MODES = [
  "Study",
  "Manage",
  "Import / Export",
  "Templates",
  "Scheduler"
] as const

const FlashcardsStudyFrame = ({
  children,
  stateLabel,
  constrainContent = false
}: {
  children: React.ReactNode
  stateLabel?: React.ReactNode
  constrainContent?: boolean
}) => (
  <div className="space-y-4" data-testid="flashcards-study-workspace">
    <header
      className="mx-auto max-w-6xl px-4 pt-4"
      data-testid="flashcards-study-header"
    >
      <div className="rounded-lg border border-border bg-surface px-4 py-3 shadow-sm">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="text-xs font-semibold text-text-muted">Study workspace</div>
            <h1 className="mt-1 text-2xl font-semibold text-text">Flashcards</h1>
          </div>
          {stateLabel ? (
            <div className="flex shrink-0 items-center gap-2">{stateLabel}</div>
          ) : null}
        </div>
        <nav
          aria-label="Flashcards modes"
          className="mt-3 flex flex-wrap gap-2"
        >
          {FLASHCARD_STUDY_MODES.map((mode) => (
            <span
              key={mode}
              className="rounded-full border border-border bg-bg px-2.5 py-1 text-xs font-medium text-text-muted"
            >
              {mode}
            </span>
          ))}
        </nav>
      </div>
    </header>
    <div
      className={constrainContent ? "mx-auto max-w-6xl px-4 pb-4" : undefined}
    >
      {children}
    </div>
  </div>
)

const InlineConnectionWarning = ({
  message,
  retryActionLabel,
  onRetry,
  retryDisabled,
  testId
}: {
  message: string
  retryActionLabel?: string
  onRetry?: () => void
  retryDisabled?: boolean
  testId: string
}) => (
  <div
    data-testid={testId}
    className="rounded-2xl border border-warn/40 bg-warn/10 px-4 py-3 text-sm text-text"
  >
    <div className="font-medium">{message}</div>
    {retryActionLabel && onRetry ? (
      <div className="mt-2 flex justify-start text-xs">
        <button
          type="button"
          onClick={onRetry}
          disabled={retryDisabled}
          className="inline-flex items-center gap-1 text-primary hover:text-primary disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:text-primary"
        >
          {retryActionLabel}
        </button>
      </div>
    ) : null}
  </div>
)

/**
 * FlashcardsWorkspace handles connection state, demo mode, and feature availability.
 * When online and feature is available, it renders FlashcardsManager.
 */
export const FlashcardsWorkspace: React.FC = () => {
  const { t } = useTranslation(["option", "common", "settings"])
  const navigate = useNavigate()
  const isOnline = useServerOnline()
  const { demoEnabled } = useDemoMode()
  const { uxState, hasCompletedFirstRun } = useConnectionUxState()
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const scrollToServerCard = useScrollToServerCard("/flashcards")
  const { checkOnce } = useConnectionActions()
  const [checkingConnection, setCheckingConnection] = React.useState(false)

  const flashcardsUnsupported = !capsLoading && !!capabilities && !capabilities.hasFlashcards

  const demoDecks = React.useMemo(() => getDemoFlashcardDecks(t), [t])

  const handleRetryConnection = React.useCallback(() => {
    if (checkingConnection) return
    setCheckingConnection(true)
    Promise.resolve(checkOnce())
      .catch(() => {
        // errors are surfaced via connection UX state
      })
      .finally(() => {
        setCheckingConnection(false)
      })
  }, [checkOnce, checkingConnection])

  const offlineBannerProps = React.useMemo(() => {
    if (uxState === "error_auth" || uxState === "configuring_auth") {
      return {
        badgeLabel: "Credentials required",
        title: t("option:flashcards.authTitle", {
          defaultValue: "Add your credentials to use Flashcards"
        }),
        description: t("option:flashcards.authDescription", {
          defaultValue:
            "Flashcards needs valid credentials before it can review or generate cards against your tldw server."
        }),
        examples: [
          t("option:flashcards.authExample1", {
            defaultValue:
              "Use the connection card at the top of this page to repair your server URL or API key."
          })
        ],
        primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
          defaultValue: "Go to server card"
        }),
        onPrimaryAction: scrollToServerCard
      }
    }
    if (uxState === "unconfigured" || uxState === "configuring_url") {
      const setupRequiredState = getDesignSystemState("setup_required")
      return {
        badgeLabel: setupRequiredState.label,
        title: t("option:flashcards.setupTitle", {
          defaultValue: "Finish setup to use Flashcards"
        }),
        description: hasCompletedFirstRun
          ? t("option:flashcards.setupDescriptionReturning", {
              defaultValue:
                "Flashcards still needs a configured tldw server before it can load your real decks."
            })
          : t("option:flashcards.setupDescriptionFirstRun", {
              defaultValue:
                "Finish connecting your tldw server before Flashcards can load your real decks."
            }),
        examples: [
          t("option:flashcards.setupExample1", {
            defaultValue:
              "Use the connection card at the top of this page to finish server setup."
          })
        ],
        primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
          defaultValue: "Go to server card"
        }),
        onPrimaryAction: scrollToServerCard
      }
    }
    if (uxState === "error_unreachable") {
      return {
        badgeLabel: "Server unreachable",
        title: t("option:flashcards.unreachableTitle", {
          defaultValue: "Can't reach your tldw server right now"
        }),
        description: t("option:flashcards.unreachableDescription", {
          defaultValue:
            "Flashcards depends on a reachable tldw server. Review the server card above, then retry the connection check."
        }),
        examples: [
          t("option:flashcards.unreachableExample1", {
            defaultValue:
              "If your server is running, confirm the URL in the connection card still points to the right host."
          })
        ],
        primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
          defaultValue: "Go to server card"
        }),
        onPrimaryAction: scrollToServerCard,
        retryActionLabel: t("option:buttonRetry", "Retry connection"),
        onRetry: handleRetryConnection,
        retryDisabled: checkingConnection
      }
    }
    return {
      badgeLabel: "Not connected",
      title: t("option:flashcards.emptyConnectTitle", {
        defaultValue: "Connect to use Flashcards"
      }),
      description: t("option:flashcards.emptyConnectDescription", {
        defaultValue:
          "This view needs a connected server. Use the server connection card above to fix your connection, then return here to review and generate flashcards."
      }),
      examples: [
        t("option:flashcards.emptyConnectExample1", {
          defaultValue:
            "Use the connection card at the top of this page to add your server URL and API key."
        })
      ],
      primaryActionLabel: t("option:connectionCard.buttonGoToServerCard", {
        defaultValue: "Go to server card"
      }),
      onPrimaryAction: scrollToServerCard,
      retryActionLabel: t("option:buttonRetry", "Retry connection"),
      onRetry: handleRetryConnection,
      retryDisabled: checkingConnection
    }
  }, [
    checkingConnection,
    handleRetryConnection,
    hasCompletedFirstRun,
    scrollToServerCard,
    t,
    uxState
  ])

  const demoConnectionWarning = React.useMemo(() => {
    if (uxState === "error_auth" || uxState === "configuring_auth") {
      return {
        message:
          "Demo stays available, but your Flashcards credentials need attention."
      }
    }
    if (uxState === "unconfigured" || uxState === "configuring_url") {
      return {
        message: "Demo stays available while you finish Flashcards setup."
      }
    }
    if (uxState === "error_unreachable") {
      return {
        message: "Demo stays available, but your tldw server is unreachable.",
        retryActionLabel: t("option:buttonRetry", "Retry connection"),
        onRetry: handleRetryConnection,
        retryDisabled: checkingConnection
      }
    }
    return null
  }, [checkingConnection, handleRetryConnection, t, uxState])

  // Offline state - show demo or connection banner
  if (!isOnline) {
    return (
      <FlashcardsStudyFrame
        stateLabel={
          demoEnabled ? <StatusBadge variant="demo">Local demo</StatusBadge> : undefined
        }
        constrainContent
      >
        {demoEnabled ? (
          <div className="space-y-4">
            {demoConnectionWarning ? (
              <InlineConnectionWarning
                testId="flashcards-demo-connection-warning"
                message={demoConnectionWarning.message}
                retryActionLabel={demoConnectionWarning.retryActionLabel}
                onRetry={demoConnectionWarning.onRetry}
                retryDisabled={demoConnectionWarning.retryDisabled}
              />
            ) : null}
            <FeatureEmptyState
              title={
                <span className="inline-flex items-center gap-2">
                  <StatusBadge variant="demo">Demo</StatusBadge>
                  <span>
                    {t("option:flashcards.demoTitle", {
                      defaultValue: "Explore Flashcards in demo mode"
                    })}
                  </span>
                </span>
              }
              description={t("option:flashcards.demoDescription", {
                defaultValue:
                  "This demo shows how Flashcards can turn your content into spaced-repetition cards. Connect your own server later to generate and review cards from your own notes and media."
              })}
              examples={[
                t("option:flashcards.demoExample1", {
                  defaultValue:
                    "See how decks, cards, and tags are organized across Review and Manage tabs."
                }),
                t("option:flashcards.demoExample2", {
                  defaultValue:
                    "When you connect, you'll be able to generate cards from lectures, meetings, or notes and review them on a schedule."
                }),
                t("option:flashcards.demoExample3", {
                  defaultValue:
                    "Use Flashcards together with Notes and Media to keep important ideas fresh."
                })
              ]}
              primaryActionLabel={t("option:connectionCard.buttonGoToServerCard", {
                defaultValue: "Go to server card"
              })}
              onPrimaryAction={scrollToServerCard}
            />
            <div className="rounded-lg border border-dashed border-border bg-surface p-4">
              <div className="mb-3 text-center text-xs font-semibold text-text">
                {t("option:flashcards.demoPreviewHeading", {
                  defaultValue: "Try sample flashcards"
                })}
              </div>
              <React.Suspense fallback={null}>
                <FlipCardDemo />
              </React.Suspense>
            </div>
          </div>
        ) : (
          <ConnectionProblemBanner
            badgeLabel={offlineBannerProps.badgeLabel}
            title={offlineBannerProps.title}
            description={offlineBannerProps.description}
            examples={offlineBannerProps.examples}
            primaryActionLabel={offlineBannerProps.primaryActionLabel}
            onPrimaryAction={offlineBannerProps.onPrimaryAction}
            retryActionLabel={offlineBannerProps.retryActionLabel}
            onRetry={offlineBannerProps.onRetry}
            retryDisabled={offlineBannerProps.retryDisabled}
          />
        )}
      </FlashcardsStudyFrame>
    )
  }

  // Feature not supported on this server
  if (flashcardsUnsupported) {
    return (
      <FlashcardsStudyFrame
        stateLabel={<StatusBadge variant="error">Feature unavailable</StatusBadge>}
        constrainContent
      >
        <FeatureEmptyState
          title={
            <span className="inline-flex items-center gap-2">
              <StatusBadge variant="error">Feature unavailable</StatusBadge>
              <span>
                {t("option:flashcards.offlineTitle", {
                  defaultValue: "Flashcards API not available on this server"
                })}
              </span>
            </span>
          }
          description={
            <>
              {t("option:flashcards.offlineDescription", {
                defaultValue:
                  "This tldw server does not advertise the Flashcards endpoints. Upgrade your server to a version that includes /api/v1/flashcards... to use this workspace."
              })}
              {" "}
              {t("option:flashcards.offlineVersionHint", {
                defaultValue:
                  "The Flashcards API requires tldw_server v0.1.25 or later."
              })}
            </>
          }
          examples={[
            t("option:flashcards.offlineExample1", {
              defaultValue:
                "Check Health & diagnostics to confirm your server version and available APIs."
            }),
            t("option:flashcards.offlineExample2", {
              defaultValue:
                "After upgrading, reload the extension and return to Flashcards."
            }),
            t("option:flashcards.offlineExample3", {
              defaultValue:
                "If you just updated your server, it may take a moment for capabilities to refresh."
            })
          ]}
          primaryActionLabel={t("settings:healthSummary.diagnostics", {
            defaultValue: "Health & diagnostics"
          })}
          onPrimaryAction={() => navigate("/settings/health")}
        />
      </FlashcardsStudyFrame>
    )
  }

  // Online and feature supported - render main manager
  return (
    <FlashcardsStudyFrame>
      <React.Suspense
        fallback={
          <div className="flex justify-center py-8">
            <Spin />
          </div>
        }
      >
        <FlashcardsManager />
      </React.Suspense>
    </FlashcardsStudyFrame>
  )
}

export default FlashcardsWorkspace
