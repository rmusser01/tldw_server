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
  isFirstSourceOpenDetail,
  requestQuickIngestOpen
} from "@/utils/quick-ingest-open"
import { isSetupStatusRequiringWizard } from "./setup-status"

const LazyCompanionHomeShell = React.lazy(() =>
  import("@/components/Option/CompanionHome").then((module) => ({
    default: module.CompanionHomeShell
  }))
)

const LazyOptionHostedHome = React.lazy(() => import("./option-hosted-home"))

const FIRST_SOURCE_MILESTONE_DISMISSED_KEY =
  "tldw:first-source-milestone-dismissed"

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
}) => {
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent("tldw:discuss-media", {
      detail: {
        mediaId: payload.mediaId,
        title: payload.title || "First source",
        mode: "rag_media"
      }
    })
  )
}

const OptionIndex = () => {
  const hostedMode = isHostedTldwDeployment()
  const { phase } = useConnectionState()
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

  if (isSetupStatusRequiringWizard(setupStatus)) {
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
  const firstSourceSession = isFirstSourceOpenDetail(
    quickIngestSession?.openDetail
  )
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

  return (
    <OptionLayout>
      {showFirstSourcePrompt ? (
        <FirstSourceMilestonePrompt
          readinessStatus={firstSourcePromptStatus}
          lastSourceLabel={firstSourceRunSummary?.primarySourceLabel}
          errorMessage={firstSourceRunSummary?.errorMessage}
          onAddSource={(kind) => {
            setLastFirstSourceKind(kind)
            openFirstSourceQuickIngest(kind)
          }}
          onRetry={() => openFirstSourceQuickIngest(lastFirstSourceKind)}
          onAskAboutSource={
            firstSourceMediaId && firstSourceAskReady
              ? () =>
                  discussFirstSource({
                    mediaId: firstSourceMediaId,
                    title: firstSourceRunSummary?.primarySourceLabel ?? null
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
