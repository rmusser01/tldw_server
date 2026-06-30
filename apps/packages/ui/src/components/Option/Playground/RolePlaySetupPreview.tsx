import React from "react"
import { useTranslation } from "react-i18next"

import type { RolePlayState } from "./role-play-state"

type RolePlaySetupPreviewProps = {
  before: RolePlayState
  after: RolePlayState
  onRevert?: () => void
}

type PreviewCopy = {
  none: string
  noBehavior: string
  custom: string
  modified: string
  noScene: string
  noGenerationStyle: string
  external: string
  noContext: string
  pinned: (count: number) => string
  pinnedWithExternal: (count: number) => string
}

const formatIdentity = (state: RolePlayState, copy: PreviewCopy): string =>
  state.identity?.name || state.identity?.id || copy.none

const formatBehavior = (state: RolePlayState, copy: PreviewCopy): string => {
  if (!state.behavior) return copy.noBehavior
  const title = state.behavior.title || copy.custom
  return state.behavior.modified ? `${title} ${copy.modified}` : title
}

const formatScene = (state: RolePlayState, copy: PreviewCopy): string =>
  state.scene?.summary || copy.noScene

const formatGeneration = (state: RolePlayState, copy: PreviewCopy): string =>
  state.generationStyle?.label || copy.noGenerationStyle

const formatContext = (state: RolePlayState, copy: PreviewCopy): string => {
  const pinned = state.context.pinnedCount
  const external = state.context.hasExternalContext
  if (pinned > 0 && external) return copy.pinnedWithExternal(pinned)
  if (pinned > 0) return copy.pinned(pinned)
  if (external) return copy.external
  return copy.noContext
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
  const copy = React.useMemo<PreviewCopy>(
    () => ({
      none: t("common:none", "None"),
      noBehavior: t("playground:composer.noBehavior", "No behavior"),
      custom: t("common:custom", "Custom"),
      modified: t("playground:composer.modifiedSuffix", "modified"),
      noScene: t("playground:composer.noScene", "No scene"),
      noGenerationStyle: t(
        "playground:composer.noGenerationStyle",
        "No generation style"
      ),
      external: t("playground:composer.externalContext", "External"),
      noContext: t("playground:composer.noContext", "No context"),
      pinned: (count: number) =>
        t("playground:composer.pinnedContextCount", "{{count}} pinned", {
          count
        }),
      pinnedWithExternal: (count: number) =>
        t(
          "playground:composer.pinnedAndExternalContextCount",
          "{{count}} pinned + external",
          { count }
        )
    }),
    [t]
  )

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
        before={formatIdentity(before, copy)}
        after={formatIdentity(after, copy)}
      />
      <PreviewRow
        label={t("playground:composer.context.behavior", "Behavior")}
        before={formatBehavior(before, copy)}
        after={formatBehavior(after, copy)}
      />
      <PreviewRow
        label={t("playground:composer.context.scene", "Scene")}
        before={formatScene(before, copy)}
        after={formatScene(after, copy)}
      />
      <PreviewRow
        label={t("playground:composer.context.generationStyle", "Generation style")}
        before={formatGeneration(before, copy)}
        after={formatGeneration(after, copy)}
      />
      <PreviewRow
        label={t("playground:composer.context.context", "Context")}
        before={formatContext(before, copy)}
        after={formatContext(after, copy)}
      />
    </section>
  )
}
