import React from "react"
import { Link } from "react-router-dom"
import { Tooltip } from "antd"
import { ArrowLeft, MessageSquareText } from "lucide-react"
import { useTranslation } from "react-i18next"
import type {
  SharedAllowedActions,
  SharedWorkspaceBootstrap
} from "@/types/shared-workspace"

type SharedWorkspaceHeaderProps = {
  bootstrap: SharedWorkspaceBootstrap
  allowedActions: SharedAllowedActions
  headingRef: React.RefObject<HTMLHeadingElement>
}

export const SharedWorkspaceHeader: React.FC<SharedWorkspaceHeaderProps> = ({
  bootstrap,
  allowedActions,
  headingRef
}) => {
  const { t } = useTranslation("playground")
  const grantedTierExplainsCeiling =
    bootstrap.share.access_level === "view_chat_add" ||
    bootstrap.share.access_level === "full_edit"
  const tooltip = grantedTierExplainsCeiling
    ? t(
        "sharedWorkspace.accessTooltip",
        "This access level is the owner's policy ceiling. Editing shared content is not available here yet."
      )
    : t(
        "sharedWorkspace.readOnlyTooltip",
        "This shared workspace is available for reading and source preview only."
      )
  const accessLabel = allowedActions.ask_grounded_questions.allowed
    ? t("sharedWorkspace.canAsk", "Can ask questions")
    : allowedActions.inspect_sources.allowed
      ? t("sharedWorkspace.viewOnly", "View only")
      : t("sharedWorkspace.accessRestricted", "Access restricted")

  return (
    <header className="flex min-h-[3.75rem] min-w-0 shrink-0 items-center gap-3 border-b border-border bg-surface px-3 py-2 sm:px-4">
      <Tooltip title={t("sharedWorkspace.back", "Back to Shared with me")}>
        <Link
          to="/shared-with-me"
          aria-label={t("sharedWorkspace.back", "Back to Shared with me")}
          className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-md text-text-muted outline-none transition-colors hover:bg-surface2 hover:text-text focus-visible:ring-2 focus-visible:ring-focus sm:h-9 sm:w-9"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
        </Link>
      </Tooltip>
      <div className="min-w-0 flex-1">
        <h1
          ref={headingRef}
          tabIndex={-1}
          className="truncate text-base font-semibold leading-5 outline-none focus-visible:ring-2 focus-visible:ring-focus sm:text-lg"
        >
          {bootstrap.workspace.name}
        </h1>
        <p className="truncate text-xs text-text-muted">
          {t("sharedWorkspace.sharedBy", "Shared by {{owner}}", {
            owner: bootstrap.share.owner_display_name
          })}
        </p>
      </div>
      <Tooltip title={tooltip}>
        <span
          tabIndex={0}
          aria-label={accessLabel}
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-full border border-border bg-surface2 px-2.5 text-xs font-medium text-text-muted outline-none focus-visible:ring-2 focus-visible:ring-focus"
        >
          <MessageSquareText className="h-3.5 w-3.5" aria-hidden="true" />
          <span className="hidden sm:inline">
            {accessLabel}
          </span>
        </span>
      </Tooltip>
    </header>
  )
}
