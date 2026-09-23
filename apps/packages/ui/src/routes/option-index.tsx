import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import React from "react"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import {
  FirstSourceMilestonePrompt,
  type FirstSourceKind
} from "@/components/Option/Onboarding/FirstSourceMilestonePrompt"
import { PostSetupApiRecovery } from "@/components/Option/Onboarding/PostSetupApiRecovery"
import {
  useConnectionActions,
  useConnectionState
} from "@/hooks/useConnectionState"
import { useFocusComposerOnConnect } from "@/hooks/useComposerFocus"
import { usePostOnboardingMediaReadiness } from "@/hooks/usePostOnboardingMediaReadiness"
import { useHomeMilestoneScope } from "@/hooks/useHomeMilestoneScope"
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding"
import OptionLayout from "~/components/Layouts/Layout"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import {
  isFirstSourceQuickIngestKind,
  isFirstSourceOpenDetail,
  requestQuickIngestOpen
} from "@/utils/quick-ingest-open"
import { isSetupStatusRequiringWizard } from "./setup-status"
import { ConnectionPhase } from "@/types/connection"
import { useNavigate } from "react-router-dom"
import { useMilestoneStore } from "@/store/milestones"
import { createMediaChatHandoff, buildMediaChatHandoffRoute, removeMediaChatHandoff } from "@/services/tldw/media-chat-handoff"

const LazyUnifiedSetupWizard = React.lazy(() =>
  import("@/components/Option/Onboarding/UnifiedSetupWizard").then((module) => ({
    default: module.UnifiedSetupWizard
  }))
)

const LazyCompanionHomeShell = React.lazy(() =>
  import("@/components/Option/CompanionHome").then((module) => ({
    default: module.CompanionHomeShell
  }))
)

const LazyOptionHostedHome = React.lazy(() => import("./option-hosted-home"))

const FIRST_SOURCE_MILESTONE_DISMISSED_KEY =
  "tldw:first-source-milestone-dismissed"
const FIRST_SOURCE_STARTER_QUESTIONS = [
  "Summarize this source.",
  "List the key claims.",
  "What should I remember?"
] as const

const readFirstSourceDismissed = () => {
  if (typeof window === "undefined") return false
  try {
    return (
      window.localStorage.getItem(FIRST_SOURCE_MILESTONE_DISMISSED_KEY) === "1"
    )
  } catch {
    return false
  }
}

const openFirstSourceQuickIngest = (
  kind: FirstSourceKind,
  ownerScope: string | null
) => {
  if (!ownerScope) return
  requestQuickIngestOpen(
    {
      source: "first_source_milestone",
      ownerScope,
      preferredPreset: "quick",
      firstSource: true,
      firstSourceKind: kind
    },
    { focusTrigger: true }
  )
}

const persistFirstSourceDiscussion = async (payload: {
  mediaId: string
  ownerScope: string
  title: string | null
  question?: string | null
}) => {
  if (typeof window === "undefined") return
  const detail: {
    mediaId: string
    ownerScope: string
    title: string
    mode: "rag_media"
    content?: string
  } = {
    mediaId: payload.mediaId,
    ownerScope: payload.ownerScope,
    title: payload.title || "First source",
    mode: "rag_media"
  }
  const question = payload.question?.trim()
  if (question) {
    detail.content = question
  }
  return createMediaChatHandoff(detail)
}

const SETUP_BANNER_DISMISSED_KEY = "__tldw_setup_banner_dismissed"

const OptionIndex = () => {
  const hostedMode = isHostedTldwDeployment()
  const homeScope = useHomeMilestoneScope()
  const handoffBoundaryRevision = React.useRef(0)
  React.useLayoutEffect(() => watchChatAccountChanges(invalidated => {
    if (invalidated) handoffBoundaryRevision.current += 1
  }), [])
  const handoffLifetime = React.useRef<AbortController | null>(null)
  React.useLayoutEffect(() => {
    const lifetime = new AbortController()
    handoffLifetime.current = lifetime
    return () => lifetime.abort()
  }, [homeScope])
  const { phase, serverUrl } = useConnectionState()
  const { checkOnce } = useConnectionActions()
  const {
    state: firstRunState,
    metadata: firstRunMetadata,
    loading: setupLoading,
    adoptState: adoptFirstRunState
  } = useSetupOnboarding()
  const [didHydrate, setDidHydrate] = React.useState(false)
  const [firstSourceDismissed, setFirstSourceDismissed] = React.useState(
    readFirstSourceDismissed
  )
  const navigate = useNavigate()
  const [handoffError, setHandoffError] = React.useState<string | null>(null)
  const discussFirstSource = async (
    payload: Omit<
      Parameters<typeof persistFirstSourceDiscussion>[0],
      "ownerScope"
    >
  ) => {
    const boundaryRevision = handoffBoundaryRevision.current
    const lifetime = handoffLifetime.current
    if (!homeScope || !lifetime || lifetime.signal.aborted) return
    const requestOwnerScope = homeScope
    setHandoffError(null)
    try {
      const token = await persistFirstSourceDiscussion({
        ...payload,
        ownerScope: requestOwnerScope
      })
      if (!token) return
      if (boundaryRevision !== handoffBoundaryRevision.current || lifetime.signal.aborted) {
        await removeMediaChatHandoff(token)
        return
      }
      navigate(buildMediaChatHandoffRoute(token))
    } catch {
      setHandoffError(
        "Could not prepare this source for Chat. Please try again."
      )
    }
  }
  // Dismissal is scoped per server so switching connections in the same
  // browser profile does not inherit another server's dismissal.
  const setupBannerDismissKey = `${SETUP_BANNER_DISMISSED_KEY}::${
    serverUrl || "unconfigured"
  }`
  const [sessionDismissedBannerKeys, setSessionDismissedBannerKeys] =
    React.useState<ReadonlySet<string>>(() => new Set())
  const setupBannerDismissed = React.useMemo(() => {
    if (sessionDismissedBannerKeys.has(setupBannerDismissKey)) return true
    if (typeof window === "undefined") return false
    try {
      return window.localStorage.getItem(setupBannerDismissKey) === "1"
    } catch {
      return false
    }
  }, [setupBannerDismissKey, sessionDismissedBannerKeys])
  const [lastFirstSourceKind, setLastFirstSourceKind] =
    React.useState<FirstSourceKind>("web_url")
  const quickIngestSession = useQuickIngestSessionStore(
    (state) => state.session
  )

  React.useEffect(() => {
    if (hostedMode) {
      setDidHydrate(true)
      return
    }
    let cancelled = false
    const run = async () => {
      try {
        await checkOnce()
      } finally {
        if (!cancelled) setDidHydrate(true)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [checkOnce, hostedMode])

  useFocusComposerOnConnect(phase ?? null)

  const setupStatus = firstRunState?.status
  const shouldCheckPostOnboardingMedia =
    setupStatus === "completed" && !firstSourceDismissed
  const mediaReadiness = usePostOnboardingMediaReadiness(
    shouldCheckPostOnboardingMedia
  )
  const firstSourceOpenDetail = isFirstSourceOpenDetail(
    quickIngestSession?.openDetail
  )
    ? quickIngestSession?.openDetail
    : null
  const firstSourceSession =
    homeScope &&
    firstSourceOpenDetail &&
    "ownerScope" in firstSourceOpenDetail &&
    firstSourceOpenDetail.ownerScope === homeScope
      ? quickIngestSession
      : null
  const firstSourceRunSummary = firstSourceSession?.resultSummary ?? null
  const firstSourceMediaId =
    firstSourceSession?.lifecycle === "completed" &&
    firstSourceRunSummary?.status === "success" &&
    firstSourceRunSummary.firstMediaId
      ? firstSourceRunSummary.firstMediaId
      : null

  React.useEffect(() => {
    if (
      homeScope &&
      setupStatus === "completed" &&
      firstSourceMediaId &&
      mediaReadiness.status === "ready"
    ) {
      useMilestoneStore
        .getState()
        .markScopedMilestone(homeScope, "first_ingest")
    }
  }, [homeScope, setupStatus, firstSourceMediaId, mediaReadiness.status])

  if (hostedMode) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <React.Suspense
          fallback={
            <PageAssistLoader
              label="Loading home..."
              description="Preparing your workspace"
            />
          }
        >
          <LazyOptionHostedHome />
        </React.Suspense>
      </OptionLayout>
    )
  }

  if ((setupLoading || !didHydrate) && !firstRunState) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <PageAssistLoader
          label="Loading setup..."
          description="Reading first-run readiness from the server"
        />
      </OptionLayout>
    )
  }

  // A connected server means the operator already has a working setup even if
  // the wizard was never finished (e.g. configured via env or the extension).
  // Demote the wizard to a dismissible banner instead of walling the home
  // route (#2871); a true first run (no connection) still gets the wizard.
  const wizardRequired = isSetupStatusRequiringWizard(setupStatus)
  const connectionReady = phase === ConnectionPhase.CONNECTED

  if (wizardRequired && !connectionReady) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <React.Suspense fallback={<PageAssistLoader />}>
          <LazyUnifiedSetupWizard
            initialState={firstRunState}
            initialMetadata={firstRunMetadata}
            onStateChange={adoptFirstRunState}
          />
        </React.Suspense>
      </OptionLayout>
    )
  }

  const dismissFirstSourcePrompt = () => {
    setFirstSourceDismissed(true)
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem(FIRST_SOURCE_MILESTONE_DISMISSED_KEY, "1")
      } catch {
        // Dismissed tips are best-effort frontend-only state.
      }
    }
  }

  const showFirstSourcePrompt =
    Boolean(homeScope) &&
    shouldCheckPostOnboardingMedia &&
    mediaReadiness.status === "ready"
  const firstSourceAskReady =
    Boolean(firstSourceMediaId) && mediaReadiness.status === "ready"
  const firstSourcePromptStatus =
    firstSourceSession?.lifecycle === "processing"
      ? "processing"
      : firstSourceRunSummary?.status === "error"
        ? "error"
        : firstSourceAskReady
          ? "ready"
          : "idle"

  if (
    shouldCheckPostOnboardingMedia &&
    (mediaReadiness.status === "needs_config" ||
      mediaReadiness.status === "error")
  ) {
    return (
      <OptionLayout hideHeader hideSidebar>
        <PostSetupApiRecovery
          errorMessage={mediaReadiness.errorMessage}
          onRecover={mediaReadiness.recoverWithApiKey}
          onRetry={mediaReadiness.retry}
        />
      </OptionLayout>
    )
  }

  const dismissSetupBanner = () => {
    if (typeof window !== "undefined") {
      try {
        window.localStorage.setItem(setupBannerDismissKey, "1")
      } catch {
        // Dismissal is best-effort frontend-only state.
      }
    }
    setSessionDismissedBannerKeys((prev) =>
      new Set(prev).add(setupBannerDismissKey)
    )
  }

  return (
    <OptionLayout>
      {handoffError ? (
        <p role="alert" className="mx-4 mt-4 text-destructive">
          {handoffError}
        </p>
      ) : null}
      {wizardRequired && connectionReady && !setupBannerDismissed ? (
        <div
          role="status"
          data-testid="resume-setup-banner"
          className="mx-4 mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-surface px-4 py-3"
        >
          <p className="m-0 text-sm text-text">
            Server setup isn&apos;t finished. Everything is connected, so you
            can keep working — resume setup whenever you like.
          </p>
          <span className="flex items-center gap-2">
            <button
              type="button"
              className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong"
              onClick={() => navigate("/setup")}
            >
              Resume setup
            </button>
            <button
              type="button"
              className="rounded-md px-3 py-1.5 text-sm text-text-muted hover:bg-surface2"
              onClick={dismissSetupBanner}
            >
              Dismiss
            </button>
          </span>
        </div>
      ) : null}
      {showFirstSourcePrompt ? (
        <FirstSourceMilestonePrompt
          readinessStatus={firstSourcePromptStatus}
          lastSourceLabel={firstSourceRunSummary?.primarySourceLabel}
          errorMessage={firstSourceRunSummary?.errorMessage}
          onAddSource={(kind) => {
            setLastFirstSourceKind(kind)
            openFirstSourceQuickIngest(kind, homeScope)
          }}
          onRetry={() =>
            openFirstSourceQuickIngest(
              firstSourceSession?.firstSourceAddMode ??
                (isFirstSourceQuickIngestKind(
                  firstSourceOpenDetail?.firstSourceKind
                )
                  ? firstSourceOpenDetail.firstSourceKind
                  : null) ??
                lastFirstSourceKind,
              homeScope
            )
          }
          onAskAboutSource={
            firstSourceMediaId && firstSourceAskReady
              ? () =>
                  discussFirstSource({
                    mediaId: firstSourceMediaId,
                    title: firstSourceRunSummary?.primarySourceLabel ?? null
                  })
              : undefined
          }
          starterQuestions={
            firstSourceAskReady ? [...FIRST_SOURCE_STARTER_QUESTIONS] : []
          }
          onAskStarterQuestion={
            firstSourceMediaId && firstSourceAskReady
              ? (question) =>
                  discussFirstSource({
                    mediaId: firstSourceMediaId,
                    title: firstSourceRunSummary?.primarySourceLabel ?? null,
                    question
                  })
              : undefined
          }
          onDismiss={dismissFirstSourcePrompt}
        />
      ) : null}
      <React.Suspense
        fallback={
          <PageAssistLoader
            label="Loading home..."
            description="Preparing your dashboard"
          />
        }
      >
        <LazyCompanionHomeShell surface="options" />
      </React.Suspense>
    </OptionLayout>
  )
}

export default OptionIndex
