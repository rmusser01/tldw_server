import React from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Link } from "react-router-dom"

type NoProviderBannerProps = {
  onRefresh?: () => void
  className?: string
}

/**
 * Prominent banner shown when no LLM provider API keys are configured on the
 * tldw server.  Directs the user to add a key and offers a "Refresh" action.
 */
export const NoProviderBanner: React.FC<NoProviderBannerProps> = ({
  onRefresh,
  className
}) => {
  const { t } = useTranslation(["playground", "common"])

  return (
    <div
      data-testid="no-provider-banner"
      className={`mx-auto max-w-xl rounded-xl border border-amber-500/30 bg-amber-500/5 px-5 py-4 ${className ?? ""}`}
    >
      <div className="flex items-start gap-3">
        <AlertTriangle
          className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400"
          aria-hidden="true"
        />
        <div className="min-w-0">
          <h3 className="text-sm font-medium text-text">
            {t(
              "playground:noProviderBanner.title",
              "No LLM provider configured"
            )}
          </h3>
          <p className="mt-1 text-xs text-text-muted">
            {t(
              "playground:noProviderBanner.body",
              "Chat requires an LLM provider API key (OpenAI, Anthropic, etc.). Open your server's settings to add one, then restart the server."
            )}
          </p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Link
              to="/settings/tldw"
              className="inline-flex items-center rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-colors hover:bg-primary/90"
            >
              {t("playground:noProviderBanner.openSettings", "Open Settings")}
            </Link>
            {onRefresh && (
              <button
                type="button"
                onClick={onRefresh}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text transition-colors hover:bg-surface2"
              >
                <RefreshCw className="h-3 w-3" aria-hidden="true" />
                {t("common:refresh", "Refresh")}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default NoProviderBanner
