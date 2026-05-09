import React from "react"
import { Moon, Sun } from "lucide-react"
import { useLocation, useNavigate } from "react-router-dom"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { CharacterChatOnboardingLane } from "@/components/Option/Onboarding/CharacterChatOnboardingLane"
import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { ConnectionPhase } from "@/types/connection"
import { useFocusComposerOnConnect } from "@/hooks/useComposerFocus"
import { useDarkMode } from "@/hooks/useDarkmode"
import OptionLayout from "~/components/Layouts/Layout"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import {
  buildCharacterOnboardingRoute,
  CHARACTER_CHAT_ONBOARDING_INTENT,
  getOnboardingReturnToFromSearch,
  resolveOnboardingEntryIntent
} from "@/utils/onboarding-route-intent"

const LazyOnboardingWizard = React.lazy(() =>
  import("@/components/Option/Onboarding/OnboardingWizard").then((module) => ({
    default: module.OnboardingWizard
  }))
)

const LazyCompanionHomeShell = React.lazy(() =>
  import("@/components/Option/CompanionHome").then((module) => ({
    default: module.CompanionHomeShell
  }))
)

const LazyOptionHostedHome = React.lazy(() => import("./option-hosted-home"))

const OptionIndex = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const hostedMode = isHostedTldwDeployment()
  const { phase } = useConnectionState()
  const { hasCompletedFirstRun } = useConnectionUxState()
  const { checkOnce, beginOnboarding, markFirstRunComplete } = useConnectionActions()
  const { mode, toggleDarkMode } = useDarkMode()
  const onboardingInitiated = React.useRef(false)
  const [didHydrate, setDidHydrate] = React.useState(false)
  const onboardingEntryIntent = resolveOnboardingEntryIntent(location)
  const onboardingReturnTo = getOnboardingReturnToFromSearch(location.search)
  const isCharacterChatOnboarding =
    onboardingEntryIntent === CHARACTER_CHAT_ONBOARDING_INTENT

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

  React.useEffect(() => {
    if (hostedMode) return
    if (hasCompletedFirstRun) {
      void checkOnce()
    }
  }, [checkOnce, hasCompletedFirstRun, hostedMode])

  React.useEffect(() => {
    if (hostedMode) return
    if (!didHydrate) return
    if (hasCompletedFirstRun) return
    if (onboardingInitiated.current) return
    if (phase !== ConnectionPhase.UNCONFIGURED) return

    onboardingInitiated.current = true
    void beginOnboarding()
  }, [beginOnboarding, didHydrate, hasCompletedFirstRun, hostedMode, phase])

  useFocusComposerOnConnect(phase ?? null)

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

  // During first-time setup, hide the connection shell entirely and show only
  // the onboarding wizard (“Welcome — Let’s get you connected”).
  if (!hasCompletedFirstRun) {
    const themeToggleLabel =
      mode === "dark" ? "Switch to light theme" : "Switch to dark theme"
    const onboardingTitle = isCharacterChatOnboarding
      ? "Character Chat Onboarding"
      : "Home Onboarding"
    const onboardingDescription = isCharacterChatOnboarding
      ? "Finish setup, then continue creating and chatting with characters."
      : "Start here to connect your server or try local demo mode."
    const navigateToCharacterCreate = () => {
      if (!hasCompletedFirstRun) return
      navigate(
        buildCharacterOnboardingRoute({
          returnTo: onboardingReturnTo,
          action: "create"
        })
      )
    }
    const navigateToCharacterImport = () => {
      if (!hasCompletedFirstRun) return
      navigate(
        buildCharacterOnboardingRoute({
          returnTo: onboardingReturnTo,
          action: "import"
        })
      )
    }
    const navigateToModelSettings = () => {
      if (!hasCompletedFirstRun) return
      navigate("/settings/model?from=character-chat-onboarding")
    }
    const navigateToCharacterChat = () => {
      if (!hasCompletedFirstRun) return
      navigate("/chat?from=character-chat-onboarding")
    }
    return (
      <OptionLayout hideHeader hideSidebar>
        <div className="mx-auto mb-4 w-full max-w-3xl rounded-lg border border-border bg-surface px-4 py-3">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h1 className="text-base font-semibold text-text">
                {onboardingTitle}
              </h1>
              <p className="mt-1 text-xs text-text-muted">
                {onboardingDescription}
              </p>
            </div>
            <button
              type="button"
              onClick={toggleDarkMode}
              aria-label={themeToggleLabel}
              title={themeToggleLabel}
              data-testid="chat-header-theme-toggle"
              className="inline-flex items-center justify-center rounded-md border border-border bg-surface px-2 py-2 text-text-muted transition-colors hover:bg-surface2 hover:text-text"
            >
              {mode === "dark" ? (
                <Sun className="size-4" aria-hidden="true" />
              ) : (
                <Moon className="size-4" aria-hidden="true" />
              )}
            </button>
          </div>
          {isCharacterChatOnboarding && (
            <CharacterChatOnboardingLane
              className="mt-3"
              actionsDisabled
              onCreateCharacter={navigateToCharacterCreate}
              onImportCharacter={navigateToCharacterImport}
              onChooseModel={navigateToModelSettings}
              onStartCharacterChat={navigateToCharacterChat}
            />
          )}
        </div>
        <React.Suspense
          fallback={
            <PageAssistLoader
              label="Loading setup..."
              description="Preparing onboarding"
            />
          }
        >
          <LazyOnboardingWizard
            entryIntent={onboardingEntryIntent}
            returnTo={onboardingReturnTo}
            onFinish={async () => {
              try {
                await markFirstRunComplete()
              } catch {
                // ignore markFirstRunComplete failures here; connection state will self-heal on next load
              }
              if (isCharacterChatOnboarding && onboardingReturnTo) {
                navigate(onboardingReturnTo)
              }
              void checkOnce().catch(() => undefined)
            }}
          />
        </React.Suspense>
      </OptionLayout>
    )
  }

  return (
    <OptionLayout>
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
