import React from "react"
import { useTranslation } from "react-i18next"

export const ServerOverviewHint: React.FC = () => {
  const { t } = useTranslation("settings")

  const openDocs = () => {
      const fallbackUrl =
        t(
          "onboarding.serverDocsUrl",
          "https://github.com/rmusser01/tldw_browser_assistant"
        ) || "https://github.com/rmusser01/tldw_browser_assistant"
    const docsUrl =
      t("serverOverview.docsUrl", fallbackUrl) || fallbackUrl
    window.open(docsUrl, "_blank", "noopener,noreferrer")
  }

  return (
    <div className="mt-2 w-full rounded-md border border-border bg-surface2 p-3 text-left text-xs text-text">
      <p className="mb-1 text-sm font-medium">
        {t(
          "serverOverview.workspaceTitle",
          "What your tldw server provides"
        )}
      </p>
      <ul className="mb-2 list-disc space-y-1 pl-4">
        <li>
          {t(
            "serverOverview.pointChat",
            "Powers chat, multi-turn memory, and server-backed history."
          )}
        </li>
        <li>
          {t(
            "serverOverview.pointKnowledge",
            "Enables Knowledge search & RAG so chats can reference your own documents."
          )}
        </li>
        <li>
          {t(
            "serverOverview.pointMedia",
            "Handles media ingest, transcription, and metrics so you can review calls and videos."
          )}
        </li>
      </ul>
      <button
        type="button"
        onClick={openDocs}
        className="text-xs font-medium text-primary hover:text-primaryStrong"
      >
        {t(
          "serverOverview.docsCta",
          "View server setup guide"
        )}
      </button>
    </div>
  )
}

export default ServerOverviewHint
