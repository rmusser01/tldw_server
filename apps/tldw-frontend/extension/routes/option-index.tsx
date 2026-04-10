import React, { useState } from "react"
import { browser } from "wxt/browser"

import {
  useConnectionActions,
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { useFocusComposerOnConnect } from "@/hooks/useComposerFocus"
import { OnboardingWizard } from "@/components/Option/Onboarding/OnboardingWizard"
import OptionLayout from "@web/components/layout/WebLayout"
import { Playground } from "~/components/Option/Playground/Playground"
import { useDemoMode } from "@/context/demo-mode"

/**
 * Extension-specific intro screen shown before the standard onboarding wizard.
 * Explains what tldw_server is and offers demo mode for users who don't have one yet.
 */
const ExtensionIntro: React.FC<{
  onContinue: () => void
  onDemo: () => void
}> = ({ onContinue, onDemo }) => (
  <div className="mx-auto flex max-w-md flex-col items-center gap-6 px-4 py-12 text-center">
    <h1 className="text-2xl font-semibold">Welcome to tldw</h1>
    <p className="text-sm text-text-subtle">
      tldw is a research assistant that helps you ingest, analyze, and search
      media. This extension connects to a <strong>tldw server</strong> running
      on your computer or network.
    </p>
    <div className="flex flex-col gap-3 w-full">
      <button
        onClick={onContinue}
        className="w-full rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
      >
        I have a tldw server &mdash; connect now
      </button>
      <button
        onClick={onDemo}
        className="w-full rounded-lg border border-border px-4 py-2.5 text-sm font-medium text-text-subtle transition-colors hover:bg-surface-hover"
      >
        <span>Try demo mode (no server needed)</span>
        <span className="block mt-1 text-xs opacity-70">
          Explore with sample data. Chat and search use demo content only.
        </span>
      </button>
    </div>
    <a
      href="https://github.com/rmusser01/tldw_server"
      target="_blank"
      rel="noopener noreferrer"
      className="text-xs text-text-subtle underline hover:text-text"
    >
      Learn how to set up a tldw server
    </a>
  </div>
)

const showFtueNotification = () => {
  if (typeof browser?.notifications?.create === "function") {
    void browser.notifications
      .create("ftue-features", {
        type: "basic",
        iconUrl: browser.runtime.getURL("icon/128.png"),
        title: browser.i18n.getMessage("ftueNotificationTitle") || "You're all set!",
        message:
          browser.i18n.getMessage("ftueNotificationMessage") ||
          "Right-click on any page to use AI actions. Click the tldw icon to open the sidebar for chat."
      })
      .catch(() => undefined)
  }
}

const OptionIndex = () => {
  const { phase } = useConnectionState()
  const { hasCompletedFirstRun } = useConnectionUxState()
  const { checkOnce, beginOnboarding, markFirstRunComplete } = useConnectionActions()
  const { setDemoEnabled } = useDemoMode()
  const onboardingInitiated = React.useRef(false)
  const [showIntro, setShowIntro] = useState(true)

  React.useEffect(() => {
    if (hasCompletedFirstRun) {
      void checkOnce()
    }
  }, [checkOnce, hasCompletedFirstRun])

  React.useEffect(() => {
    if (!hasCompletedFirstRun && !onboardingInitiated.current) {
      onboardingInitiated.current = true
      void beginOnboarding()
    }
  }, [hasCompletedFirstRun, beginOnboarding])

  useFocusComposerOnConnect(phase ?? null)

  // During first-time setup, show extension intro then onboarding wizard.
  if (!hasCompletedFirstRun) {
    if (showIntro) {
      return (
        <OptionLayout hideHeader hideSidebar>
          <ExtensionIntro
            onContinue={() => setShowIntro(false)}
            onDemo={async () => {
              setDemoEnabled(true)
              try {
                await markFirstRunComplete()
                showFtueNotification()
              } catch {
                // markFirstRunComplete failed; skip notification since state may be inconsistent
              }
            }}
          />
        </OptionLayout>
      )
    }

    return (
      <OptionLayout hideHeader hideSidebar>
        <OnboardingWizard
          onFinish={async () => {
            try {
              await markFirstRunComplete()
              showFtueNotification()
            } catch {
              // markFirstRunComplete failed; skip notification since state may be inconsistent
            }
            void checkOnce().catch(() => undefined)
          }}
        />
      </OptionLayout>
    )
  }

  return (
    <OptionLayout>
      <Playground />
    </OptionLayout>
  )
}

export default OptionIndex
