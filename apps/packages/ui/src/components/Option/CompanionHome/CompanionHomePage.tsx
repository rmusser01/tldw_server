import React, { useEffect, useRef } from "react"
import { Link } from "react-router-dom"

import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useIsConnected } from "@/hooks/useConnectionState"
import { DESIGN_SYSTEM_STATES, getDesignSystemState } from "@/design-system"
import { useMilestoneStore } from "@/store/milestones"
import {
  type CompanionHomeSnapshot,
  type CompanionHomeSurface
} from "@/services/companion-home"
import {
  DEFAULT_COMPANION_HOME_LAYOUT,
  type CompanionHomeLayoutCard
} from "@/store/companion-home-layout"

import {
  createEmptySnapshot,
  useCompanionHomeData,
  useCompanionHomeLayout,
  useScheduledTaskHomeSignals
} from "./hooks"
import { CustomizeHomeDrawer } from "./CustomizeHomeDrawer"
import { GettingStartedSection } from "./GettingStartedSection"
import { AutomationInboxCard } from "./cards/AutomationInboxCard"
import { GoalsFocusCard } from "./cards/GoalsFocusCard"
import { InboxPreviewCard } from "./cards/InboxPreviewCard"
import { NeedsAttentionCard } from "./cards/NeedsAttentionCard"
import { ReadingQueueCard } from "./cards/ReadingQueueCard"
import { RecentActivityCard } from "./cards/RecentActivityCard"
import { ResumeWorkCard } from "./cards/ResumeWorkCard"
import { WhatsNextCard } from "./cards/WhatsNextCard"
import type { CompanionHomeCardState } from "./cards/CardShell"

type CompanionHomePageProps = {
  surface: CompanionHomeSurface
  onPersonalizationEnabled?: () => void
}

const SETUP_REQUIRED_LABEL =
  getDesignSystemState("setup_required")?.label ??
  DESIGN_SYSTEM_STATES.setup_required?.label ??
  ["Setup", "required"].join(" ")

const formatDegradedSources = (sources: CompanionHomeSnapshot["degradedSources"]): string =>
  sources
    .map((source) => {
      if (source === "workspace") return "companion workspace"
      if (source === "reading") return "reading queue"
      return "notes"
    })
    .join(", ")

export function CompanionHomePage({
  surface,
  onPersonalizationEnabled
}: CompanionHomePageProps) {
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const [customizeOpen, setCustomizeOpen] = React.useState(false)

  // Bootstrap milestones from existing usage evidence for returning users.
  // Also re-bootstrap when connection becomes ready, so mission cards appear
  // immediately after onboarding completes without requiring a page refresh.
  const isConnected = useIsConnected()
  const bootstrapMilestones = useMilestoneStore((s) => s.bootstrapFromExistingUsage)
  const markMilestone = useMilestoneStore((s) => s.markMilestone)
  const milestoneBootstrapped = useRef(false)
  useEffect(() => {
    if (!milestoneBootstrapped.current) {
      milestoneBootstrapped.current = true
      bootstrapMilestones()
    }
  }, [bootstrapMilestones])
  // Re-bootstrap when connection becomes ready so post-onboarding milestones appear
  useEffect(() => {
    if (isConnected && milestoneBootstrapped.current) {
      bootstrapMilestones()
    }
  }, [isConnected, bootstrapMilestones])
  useEffect(() => {
    if (isConnected) {
      markMilestone("first_connection")
    }
  }, [isConnected, markMilestone])

  const hasPersonalization = Boolean(capabilities?.hasPersonalization)
  const { layout, updateLayout } = useCompanionHomeLayout(surface)
  const {
    snapshot,
    profile,
    profileLoaded,
    loading,
    error,
    enablingCompanion,
    refresh,
    enableCompanion
  } = useCompanionHomeData({
    surface,
    capsLoading,
    hasPersonalization,
    onPersonalizationEnabled
  })
  const {
    items: scheduledTaskSignalItems,
    loading: scheduledTaskSignalsLoading,
    partial: scheduledTaskSignalsPartial,
    error: scheduledTaskSignalsError,
    refresh: refreshScheduledTaskSignals
  } = useScheduledTaskHomeSignals({
    enabled: !capsLoading
  })
  const refreshHome = React.useCallback(() => {
    refresh()
    refreshScheduledTaskSignals()
  }, [refresh, refreshScheduledTaskSignals])
  const hasConversationRoute =
    Boolean(capabilities?.hasPersonalization) &&
    Boolean(capabilities?.hasPersona) &&
    profileLoaded &&
    Boolean(profile?.enabled)

  if (capsLoading || (loading && !snapshot)) {
    return (
      <div className="rounded-3xl border border-border/80 bg-surface/90 p-6 shadow-sm" data-testid="companion-home-page">
        <h1 className="text-3xl font-semibold text-text">Companion</h1>
        <p className="mt-3 text-sm text-text-muted">Loading your companion home dashboard.</p>
      </div>
    )
  }

  const resolvedSnapshot = snapshot ?? createEmptySnapshot(surface)
  const workspaceUnavailable = resolvedSnapshot.degradedSources.includes("workspace")
  const readingUnavailable = resolvedSnapshot.degradedSources.includes("reading")
  const notesUnavailable = resolvedSnapshot.degradedSources.includes("notes")

  const resolvePersonalizedCardState = (
    itemsLength: number,
    setupDescription: string,
    unavailableDescription: string
  ): CompanionHomeCardState | undefined => {
    if (itemsLength > 0) return undefined
    if (!hasPersonalization) {
      return {
        label: SETUP_REQUIRED_LABEL,
        description: "Connect your tldw server and enable personalization to unlock this feature. " + setupDescription
      }
    }
    if (profileLoaded && !profile?.enabled) {
      return {
        label: "Enable Companion",
        description: "Turn on personalized recommendations and goal tracking to populate this card. " + setupDescription
      }
    }
    if (workspaceUnavailable) {
      return {
        label: "Temporarily unavailable",
        description: unavailableDescription
      }
    }
    return undefined
  }

  const inboxState = resolvePersonalizedCardState(
    resolvedSnapshot.inbox.length,
    "Companion inbox items unlock once personalization is available for this workspace.",
    "Companion inbox data is temporarily unavailable."
  )

  const needsAttentionState =
    resolvedSnapshot.needsAttention.length > 0
      ? undefined
      : !hasPersonalization
        ? {
            label: SETUP_REQUIRED_LABEL,
            description:
              "Connect your tldw server and enable personalization to unlock needs-attention signals from goals and reading."
          }
        : profileLoaded && !profile?.enabled
          ? {
              label: "Enable Companion",
              description:
                "Turn on personalized recommendations and goal tracking to surface needs-attention signals from goals and reading."
            }
          : workspaceUnavailable || readingUnavailable
            ? {
                label: "Temporarily unavailable",
                description:
                  "Needs-attention signals are limited until companion and reading sources come back."
              }
            : undefined

  const resumeWorkState =
    resolvedSnapshot.resumeWork.length > 0
      ? undefined
      : !hasPersonalization
        ? {
            label: SETUP_REQUIRED_LABEL,
            description:
              "Connect your tldw server and enable personalization to unlock resume suggestions across goals, reading, and notes."
          }
        : profileLoaded && !profile?.enabled
          ? {
              label: "Enable Companion",
              description:
                "Turn on personalized recommendations and goal tracking to surface resume suggestions across goals, reading, and notes."
            }
          : workspaceUnavailable || readingUnavailable || notesUnavailable
            ? {
                label: "Temporarily unavailable",
                description:
                  "Resume suggestions are limited until companion, reading, and note sources are available."
              }
            : undefined

  const goalsState = resolvePersonalizedCardState(
    resolvedSnapshot.goalsFocus.length,
    "Enable Companion to turn goals and focus prompts into a live dashboard block.",
    "Goal focus data is temporarily unavailable."
  )

  const recentActivityState = resolvePersonalizedCardState(
    resolvedSnapshot.recentActivity.length,
    "Enable Companion to review recent activity and capture trails here.",
    "Recent activity is temporarily unavailable."
  )

  const readingState =
    resolvedSnapshot.readingQueue.length === 0 && readingUnavailable
      ? {
          label: "Temporarily unavailable",
          description: "Reading queue data is temporarily unavailable."
        }
      : undefined

  const topBand =
    !hasPersonalization
      ? {
          eyebrow: "Setup",
          title: "Companion setup required",
          description:
            "This server has not enabled personalization yet. The home hub stays available so you can still keep an eye on non-personalized work.",
          action: null
        }
      : profileLoaded && !profile?.enabled
        ? {
            eyebrow: "Setup",
            title: "Companion setup required",
            description:
              "Enable personalization to unlock goals, recent activity, and conversation from this home dashboard.",
            action: (
              <button
                className="rounded-full border border-text bg-text px-4 py-2 text-sm font-medium text-bg hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={enablingCompanion}
                onClick={() => {
                  void enableCompanion()
                }}
                type="button"
              >
                {enablingCompanion ? "Enabling..." : "Enable Companion"}
              </button>
            )
          }
        : resolvedSnapshot.degradedSources.length > 0
          ? {
              eyebrow: "Status",
              title: "Some companion modules are degraded",
              description: `Still showing available cards while ${formatDegradedSources(
                resolvedSnapshot.degradedSources
              )} reloads.`,
              action: null
            }
          : null

  const renderCoreCard = (cardId: CompanionHomeLayoutCard["id"]): React.ReactNode => {
    if (cardId === "resume-work") {
      return <ResumeWorkCard items={resolvedSnapshot.resumeWork} state={resumeWorkState} />
    }
    if (cardId === "goals-focus") {
      return <GoalsFocusCard items={resolvedSnapshot.goalsFocus} state={goalsState} />
    }
    if (cardId === "recent-activity") {
      return (
        <RecentActivityCard
          items={resolvedSnapshot.recentActivity}
          state={recentActivityState}
        />
      )
    }
    if (cardId === "reading-queue") {
      return <ReadingQueueCard items={resolvedSnapshot.readingQueue} state={readingState} />
    }
    return null
  }

  const orderedVisibleCoreCards = (layout ?? [])
    .filter((card) => card.kind === "core" && card.visible)
    .map((card) => ({
      id: card.id,
      node: renderCoreCard(card.id)
    }))
    .filter(
      (
        entry
      ): entry is {
        id: CompanionHomeLayoutCard["id"]
        node: React.ReactNode
      } => entry.node != null
    )

  return (
    <div className="flex flex-col gap-6" data-testid="companion-home-page">
      <header className="rounded-3xl border border-border/80 bg-surface/90 p-6 shadow-sm backdrop-blur-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-text-muted">
              Companion Home
            </p>
            <h1 className="mt-2 text-3xl font-semibold text-text">Companion</h1>
            <p className="mt-3 text-sm leading-6 text-text-muted">
              Review what is waiting in your inbox, what needs attention next, and which work threads are worth resuming right now.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-medium text-text transition-colors hover:border-primary/40 hover:bg-primary/5"
              onClick={() => setCustomizeOpen(true)}
              type="button"
            >
              Customize Home
            </button>
            <button
              className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-medium text-text transition-colors hover:border-primary/40 hover:bg-primary/5"
              onClick={refreshHome}
              type="button"
            >
              Refresh
            </button>
            {hasConversationRoute ? (
              <Link
                className="rounded-full border border-text bg-text px-4 py-2 text-sm font-medium text-bg hover:opacity-90"
                to="/companion/conversation"
              >
                Open conversation
              </Link>
            ) : null}
          </div>
        </div>

        {topBand ? (
          <div className="mt-5 rounded-2xl border border-warn/30 bg-warn/10 px-4 py-4">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-warn">
              {topBand.eyebrow}
            </p>
            <div className="mt-2 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div className="max-w-3xl">
                <h2 className="text-xl font-semibold text-text">{topBand.title}</h2>
                <p className="mt-2 text-sm leading-6 text-text-muted">{topBand.description}</p>
              </div>
              {topBand.action}
            </div>
          </div>
        ) : null}

        <div
          className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4"
          data-testid="companion-home-summary"
        >
          {[
            {
              label: "Inbox",
              value: resolvedSnapshot.summary.inboxCount,
              description: "Unread or newly surfaced items"
            },
            {
              label: "Goals",
              value: resolvedSnapshot.goalsFocus.length,
              description: "Live priorities in focus"
            },
            {
              label: "Reading",
              value: resolvedSnapshot.readingQueue.length,
              description: "Queued items ready to revisit"
            },
            {
              label: "Resume",
              value: resolvedSnapshot.summary.resumeWorkCount,
              description: "Threads worth picking back up"
            }
          ].map((item) => (
            <div
              key={item.label}
              className="rounded-2xl border border-border/70 bg-surface/75 px-4 py-3 shadow-sm"
            >
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-text-muted">
                {item.label}
              </p>
              <p className="mt-2 text-2xl font-semibold text-text">{item.value}</p>
              <p className="mt-1 text-sm text-text-muted">{item.description}</p>
            </div>
          ))}
        </div>

        {error ? (
          <div className="mt-4 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
            {error}
          </div>
        ) : null}
      </header>

      <GettingStartedSection />

      <div className="grid gap-4 xl:grid-cols-2">
        <WhatsNextCard />
        <AutomationInboxCard
          items={scheduledTaskSignalItems}
          loading={scheduledTaskSignalsLoading}
          partial={scheduledTaskSignalsPartial}
          error={scheduledTaskSignalsError}
        />
        <InboxPreviewCard items={resolvedSnapshot.inbox} state={inboxState} />
        <NeedsAttentionCard
          items={resolvedSnapshot.needsAttention}
          state={needsAttentionState}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        {layout == null ? (
          <div
            className="rounded-3xl border border-dashed border-border/70 bg-bg/50 px-5 py-6 text-sm text-text-muted xl:col-span-2"
            data-testid="companion-home-core-layout-loading"
          >
            Loading your home layout.
          </div>
        ) : (
          orderedVisibleCoreCards.map((entry) => (
            <React.Fragment key={entry.id}>{entry.node}</React.Fragment>
          ))
        )}
      </div>

      <CustomizeHomeDrawer
        open={customizeOpen}
        layout={layout ?? DEFAULT_COMPANION_HOME_LAYOUT}
        onClose={() => setCustomizeOpen(false)}
        onLayoutChange={updateLayout}
      />
    </div>
  )
}
