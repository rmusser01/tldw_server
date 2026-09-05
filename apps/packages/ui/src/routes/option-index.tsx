import React from "react"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import {
  FirstSourceMilestonePrompt,
  type FirstSourceKind
} from "@/components/Option/Onboarding/FirstSourceMilestonePrompt"
import { PostSetupApiRecovery } from "@/components/Option/Onboarding/PostSetupApiRecovery"
import { UnifiedSetupWizard } from "@/components/Option/Onboarding/UnifiedSetupWizard"
import {
  useConnectionActions,
  useConnectionState
} from "@/hooks/useConnectionState"
import { useFocusComposerOnConnect } from "@/hooks/useComposerFocus"
import { usePostOnboardingMediaReadiness } from "@/hooks/usePostOnboardingMediaReadiness"
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

const openFirstSourceQuickIngest = (kind: FirstSourceKind) => {
  requestQuickIngestOpen(
    {
      source: "first_source_milestone",
      preferredPreset: "quick",
      firstSource: true,
      firstSourceKind: kind
    },
    { focusTrigger: true }
  )
}

const discussFirstSource = (payload: {
  mediaId: string
  title: string | null
  question?: string | null
}) => {
  if (typeof window === "undefined") return
  const detail: {
    mediaId: string
    title: string
    mode: "rag_media"
    content?: string
  } = {
    mediaId: payload.mediaId,
    title: payload.title || "First source",
    mode: "rag_media"
  }
  const question = payload.question?.trim()
  if (question) {
    detail.content = question
  }
  window.dispatchEvent(
    new CustomEvent("tldw:discuss-media", {
      detail
    })
  )
}

const SETUP_BANNER_DISMISSED_KEY = "__tldw_setup_banner_dismissed"

const OptionIndex = () => {
  const hostedMode = isHostedTldwDeployment()
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
        <UnifiedSetupWizard
          initialState={firstRunState}
          initialMetadata={firstRunMetadata}
          onStateChange={adoptFirstRunState}
        />
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
    shouldCheckPostOnboardingMedia && mediaReadiness.status === "ready"
  const firstSourceOpenDetail = isFirstSourceOpenDetail(
    quickIngestSession?.openDetail
  )
    ? quickIngestSession?.openDetail
    : null
  const firstSourceSession = firstSourceOpenDetail
    ? quickIngestSession
    : null
  const firstSourceRunSummary = firstSourceSession?.resultSummary ?? null
  const firstSourceMediaId =
    firstSourceRunSummary?.status === "success" &&
    firstSourceRunSummary.firstMediaId
      ? firstSourceRunSummary.firstMediaId
      : null
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
    setSessionDismissedBannerKeys(
      (prev) => new Set(prev).add(setupBannerDismissKey)
    )
  }

  return (
    <OptionLayout>
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
            openFirstSourceQuickIngest(kind)
          }}
          onRetry={() =>
            openFirstSourceQuickIngest(
              firstSourceSession?.firstSourceAddMode ??
                (isFirstSourceQuickIngestKind(firstSourceOpenDetail?.firstSourceKind)
                  ? firstSourceOpenDetail.firstSourceKind
                  : null) ??
                lastFirstSourceKind
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
            firstSourceAskReady
              ? [...FIRST_SOURCE_STARTER_QUESTIONS]
              : []
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
