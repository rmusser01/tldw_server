import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { UnifiedSetupWizard } from "@/components/Option/Onboarding/UnifiedSetupWizard"
import { SetupRequiredPanel } from "@/components/ui/state"
import { useConnectionActions } from "@/hooks/useConnectionState"
import { useSetupOnboarding } from "@/hooks/useSetupOnboarding"
import { sanitizeServerErrorMessage } from "@/utils/server-error-message"
import OptionLayout from "~/components/Layouts/Layout"
import { isSetupStatusRequiringWizard } from "./setup-status"

const ASSISTANT_SETUP_DISMISSED_KEY = "assistant_setup_dismissed"

const OptionSetup = () => {
  const navigate = useNavigate()
  const { t } = useTranslation("option")
  const { state, metadata, loading, adoptState } = useSetupOnboarding()
  const { setConfigPartial, testConnectionFromOnboarding } = useConnectionActions()
  const [serverUrl, setServerUrl] = useState("http://127.0.0.1:8000")
  const [apiKey, setApiKey] = useState("")
  const [showKeyHelp, setShowKeyHelp] = useState(false)
  const [testing, setTesting] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const status = state?.status
  const showWizard = isSetupStatusRequiringWizard(status)
  const showLoader = loading && !state
  const showRouteHeading = !showWizard || showLoader
  const routeHeading = t("setupRoute.heading", "Setup")

  return (
    <OptionLayout hideHeader hideSidebar>
      {showRouteHeading ? <h1 className="sr-only">{routeHeading}</h1> : null}
      <section className="mx-auto mb-4 w-full max-w-3xl rounded-lg border border-border bg-surface p-4 text-text shadow-sm">
        <div className="flex flex-col gap-4">
          <div>
            <h2 className="text-base font-semibold text-text">
              {t("setupRoute.selfHostTitle", "Connect your tldw server")}
            </h2>
            <p className="mt-1 text-sm text-text-muted">
              {t(
                "setupRoute.selfHostMessage",
                "Add your local server URL and single-user API key, then test the connection before continuing."
              )}
            </p>
          </div>
          <form
            className="grid gap-3"
            onSubmit={async (event) => {
              event.preventDefault()
              setTesting(true)
              setConnectionError(null)
              try {
                await setConfigPartial({
                  serverUrl: serverUrl.trim(),
                  authMode: "single-user",
                  apiKey: apiKey.trim()
                })
                await testConnectionFromOnboarding()
                navigate("/settings/health")
              } catch (error) {
                const fallbackMessage = t(
                  "setupRoute.selfHostConnectionFailed",
                  "Connection test failed. Check the server URL and API key, then try again."
                )
                setConnectionError(
                  sanitizeServerErrorMessage(error, fallbackMessage)
                )
              } finally {
                setTesting(false)
              }
            }}>
            <label className="grid gap-1 text-sm font-medium text-text" htmlFor="setup-server-url">
              {t("setupRoute.serverUrlLabel", "Server URL")}
              <input
                id="setup-server-url"
                className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
                value={serverUrl}
                onChange={(event) => setServerUrl(event.target.value)}
                placeholder="http://127.0.0.1:8000"
                type="url"
              />
            </label>
            <label className="grid gap-1 text-sm font-medium text-text" htmlFor="setup-api-key">
              {t("setupRoute.apiKeyLabel", "API Key")}
              <input
                id="setup-api-key"
                className="rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text"
                value={apiKey}
                onChange={(event) => setApiKey(event.target.value)}
                placeholder={t("setupRoute.apiKeyPlaceholder", "Enter your API key")}
                type="password"
                aria-describedby={showKeyHelp ? "setup-api-key-help" : undefined}
              />
            </label>
            {connectionError ? (
              <p className="text-sm text-danger" role="alert">
                {connectionError}
              </p>
            ) : null}
            {showKeyHelp ? (
              <p
                id="setup-api-key-help"
                className="rounded-md bg-surface2 p-3 text-sm text-text-muted"
              >
                {t(
                  "setupRoute.keyHelp",
                  "For single-user installs, use the SINGLE_USER_API_KEY value from your server environment or the API key printed in the server startup output."
                )}
              </p>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <button
                type="submit"
                className="inline-flex min-h-[36px] items-center justify-center rounded-md bg-primary px-3.5 py-1.5 text-sm font-medium text-white hover:bg-primaryStrong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg disabled:cursor-not-allowed disabled:opacity-50"
                disabled={testing || !serverUrl.trim()}>
                {testing
                  ? t("setupRoute.testingConnection", "Testing...")
                  : t("setupRoute.testConnection", "Test connection")}
              </button>
              <button
                type="button"
                className="inline-flex min-h-[36px] items-center justify-center rounded-md border border-border px-3.5 py-1.5 text-sm font-medium text-text hover:bg-surface2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
                aria-expanded={showKeyHelp}
                onClick={() => setShowKeyHelp((value) => !value)}>
                {t("setupRoute.keyHelpAction", "Where do I find my key?")}
              </button>
              <button
                type="button"
                className="inline-flex min-h-[36px] items-center justify-center rounded-md px-3.5 py-1.5 text-sm font-medium text-text-muted underline hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
                onClick={() => {
                  try {
                    localStorage.setItem(ASSISTANT_SETUP_DISMISSED_KEY, "true")
                  } catch {
                    // Ignore storage failures; exploration should still proceed.
                  }
                  navigate("/chat")
                }}>
                {t("setupRoute.skipExploreAction", "Skip and explore UI")}
              </button>
            </div>
          </form>
        </div>
      </section>
      <SetupRequiredPanel
        className="mx-auto mb-4 w-full max-w-3xl"
        title={t("setupRoute.recoveryTitle", "Setup operator recovery")}
        titleHeadingLevel={2}
        message={t(
          "setupRoute.recoveryMessage",
          "Use this surface when first-run setup needs local operator recovery."
        )}
        primaryAction={{
          label: showWizard
            ? t("setupRoute.continueAction", "Continue setup")
            : t("setupRoute.returnHomeAction", "Return home"),
          onClick: () => {
            if (!showWizard) {
              navigate("/")
              return
            }
            const setupShell = document.querySelector<HTMLElement>(
              "[data-testid='unified-setup-shell']"
            )
            const fallbackButton = document.querySelector<HTMLElement>("button")
            const focusTarget = setupShell ?? fallbackButton
            focusTarget?.focus()
          }
        }}
      />
      {showLoader ? (
        <PageAssistLoader
          label={t("setupRoute.loadingLabel", "Loading setup...")}
          description={t(
            "setupRoute.loadingDescription",
            "Reading first-run readiness from the server"
          )}
        />
      ) : showWizard ? (
        <UnifiedSetupWizard
          initialState={state}
          initialMetadata={metadata}
          onStateChange={adoptState}
          onComplete={() => navigate("/")}
        />
      ) : null}
    </OptionLayout>
  )
}

export default OptionSetup
