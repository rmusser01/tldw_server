import { useState } from "react"

import type { FirstRunMetadata, FirstRunState } from "@/types/setup-onboarding"
import {
  isBlockedSetupState,
  resolveApiSetupUrl,
} from "./setup-entry-choice-utils"

export type SetupEntryChoiceProps = {
  state: FirstRunState | null
  metadata: FirstRunMetadata | null
  configuredServerUrl?: string | null
  currentOrigin?: string | null
  onStartWebUiSetup: () => void
  onRefreshSetupState: () => Promise<void> | void
}

const fallbackApiSetupCopy =
  "Open the API server setup page on the machine running tldw. For the default local install this is usually http://127.0.0.1:8000/setup."

const setupAccessCopy = (metadata: FirstRunMetadata | null): string => {
  if (metadata?.remote_setup_enabled === true) {
    return "Remote API setup access is enabled and may still be restricted by the server setup allowlist."
  }

  if (metadata?.connection.browser_access === "local") {
    return "API server setup should open locally."
  }

  return "API server setup may need to be opened on the server machine or enabled for remote setup by the operator."
}

const getBrowserOrigin = (): string | null => {
  if (typeof window === "undefined") {
    return null
  }

  return window.location.origin
}

export function SetupEntryChoice({
  state,
  metadata,
  configuredServerUrl = null,
  currentOrigin,
  onStartWebUiSetup,
  onRefreshSetupState,
}: SetupEntryChoiceProps) {
  const [apiSetupOpened, setApiSetupOpened] = useState(false)
  const blocked = isBlockedSetupState(state)
  const resolvedCurrentOrigin =
    currentOrigin === undefined ? getBrowserOrigin() : currentOrigin
  const apiSetupUrl = resolveApiSetupUrl({
    metadata,
    configuredServerUrl,
    currentOrigin: resolvedCurrentOrigin,
  })
  const showRefreshAction = apiSetupOpened || !apiSetupUrl

  return (
    <section className="rounded-md border border-border bg-surface px-4 py-5">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold text-text">
          Choose where to set up tldw
        </h1>
        <p className="text-sm text-text-muted">
          You are in the tldw WebUI setup. Most users can start here to add a
          chat provider, choose a model, and send a first test chat.
        </p>
        <p className="text-sm text-text-muted">
          API server setup opens separately for server settings, recovery, and
          local or remote setup access.
        </p>
      </div>

      <div className="mt-5 space-y-5">
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-text">WebUI setup</h2>
          <p className="text-sm text-text-muted">
            Use the WebUI path for guided provider setup and first-run
            onboarding.
          </p>
          {blocked ? (
            <p className="text-sm text-warn">
              Backend setup is in recovery mode. WebUI setup can continue after
              recovery and a state refresh.
            </p>
          ) : null}
          <button
            type="button"
            onClick={onStartWebUiSetup}
            disabled={blocked}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Set up in WebUI
          </button>
        </div>

        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-text">API server setup</h2>
          <p className="text-sm text-text-muted">{setupAccessCopy(metadata)}</p>
          {apiSetupUrl ? (
            <div className="space-y-2">
              <a
                href={apiSetupUrl.href}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => setApiSetupOpened(true)}
                className="inline-flex rounded-md border border-border bg-bg px-4 py-2 text-sm font-medium text-text hover:bg-surface2"
              >
                Open API server setup
                <span className="sr-only"> (opens in a new tab)</span>
              </a>
              <p className="break-all text-xs text-text-muted">
                Resolved API setup URL:{" "}
                <code className="font-mono">{apiSetupUrl.href}</code>
              </p>
            </div>
          ) : (
            <p className="text-sm text-text-muted">{fallbackApiSetupCopy}</p>
          )}

          {showRefreshAction ? (
            <button
              type="button"
              onClick={onRefreshSetupState}
              className="rounded-md border border-border bg-bg px-4 py-2 text-sm font-medium text-text hover:bg-surface2"
            >
              I finished API server setup
            </button>
          ) : null}
        </div>
      </div>
    </section>
  )
}

export default SetupEntryChoice
