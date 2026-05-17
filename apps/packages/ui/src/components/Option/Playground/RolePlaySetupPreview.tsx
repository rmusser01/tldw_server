import React from "react"
import { useTranslation } from "react-i18next"

import type { RolePlayState } from "./role-play-state"

type RolePlaySetupPreviewProps = {
  before: RolePlayState
  after: RolePlayState
  onRevert?: () => void
}

const emptyLabel = "None"

const formatIdentity = (state: RolePlayState): string =>
  state.identity?.name || state.identity?.id || emptyLabel

const formatBehavior = (state: RolePlayState): string => {
  if (!state.behavior) return "No behavior"
  const title = state.behavior.title || "Custom"
  return state.behavior.modified ? `${title} modified` : title
}

const formatScene = (state: RolePlayState): string =>
  state.scene?.summary || "No scene"

const formatGeneration = (state: RolePlayState): string =>
  state.generationStyle?.label || "No generation style"

const formatContext = (state: RolePlayState): string => {
  const pinned = state.context.pinnedCount
  const external = state.context.hasExternalContext
  if (pinned > 0 && external) return `${pinned} pinned + external`
  if (pinned > 0) return `${pinned} pinned`
  if (external) return "External"
  return "No context"
}

const PreviewRow = ({
  label,
  before,
  after
}: {
  label: string
  before: string
  after: string
}) => {
  const changed = before !== after
  return (
    <div className="grid gap-1 rounded-md border border-border bg-surface2 p-2 text-xs sm:grid-cols-[120px_1fr]">
      <div className="font-medium text-text">{label}</div>
      <div className="min-w-0 text-text-muted">
        {changed ? (
          <div className="flex min-w-0 flex-wrap items-center gap-1">
            <span className="truncate">{before}</span>
            <span aria-hidden="true">-&gt;</span>
            <span className="truncate font-medium text-text">{after}</span>
          </div>
        ) : (
          <span className="truncate font-medium text-text">{after}</span>
        )}
      </div>
    </div>
  )
}

export const RolePlaySetupPreview: React.FC<RolePlaySetupPreviewProps> = ({
  before,
  after,
  onRevert
}) => {
  const { t } = useTranslation(["playground", "common"])

  return (
    <section
      aria-label={t("playground:composer.rolePlayPreview", "Role-play preview")}
      className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-text">
          {t("playground:composer.rolePlayPreview", "Role-play preview")}
        </h3>
        {onRevert ? (
          <button
            type="button"
            onClick={onRevert}
            className="rounded-md px-2 py-1 text-xs text-text-muted transition hover:bg-surface2 hover:text-text">
            {t("common:revert", "Revert")}
          </button>
        ) : null}
      </div>
      <PreviewRow
        label={t("playground:composer.context.character", "Character")}
        before={formatIdentity(before)}
        after={formatIdentity(after)}
      />
      <PreviewRow
        label={t("playground:composer.context.behavior", "Behavior")}
        before={formatBehavior(before)}
        after={formatBehavior(after)}
      />
      <PreviewRow
        label={t("playground:composer.context.scene", "Scene")}
        before={formatScene(before)}
        after={formatScene(after)}
      />
      <PreviewRow
        label={t("playground:composer.context.generationStyle", "Generation style")}
        before={formatGeneration(before)}
        after={formatGeneration(after)}
      />
      <PreviewRow
        label={t("playground:composer.context.context", "Context")}
        before={formatContext(before)}
        after={formatContext(after)}
      />
    </section>
  )
}
