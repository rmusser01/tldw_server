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
  sceneDraftEnabled: string
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
  state.scene ? state.scene.summary || copy.sceneDraftEnabled : copy.noScene

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
  const { t } = useTranslation("playground")
  return (
    <div className="grid gap-1 rounded-md border border-border bg-surface2 p-2 text-xs sm:grid-cols-[120px_1fr]">
      <div className="font-medium text-text">{label}</div>
      <div className="min-w-0 text-text-muted">
        {changed ? (
          <div className="flex min-w-0 flex-wrap items-start gap-x-2 gap-y-1">
            <span className="min-w-0">
              <span className="sr-only">{t("composer.previewCurrent", "Current")}: </span>
              <span className="break-words">{before}</span>
            </span>
            <span aria-hidden="true">-&gt;</span>
            <span className="min-w-0 font-medium text-text">
              <span className="sr-only">{t("composer.previewAfterApplying", "After applying")}: </span>
              <span className="break-words">{after}</span>
            </span>
          </div>
        ) : (
          <span className="break-words font-medium text-text">{after}</span>
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
      sceneDraftEnabled: t(
        "playground:composer.sceneDraftEnabled",
        "Scene draft enabled"
      ),
      noGenerationStyle: t(
        "playground:composer.noGenerationStyle",
        "No generation style"
      ),
      external: t("playground:composer.externalContext", "External"),
      noContext: t(
        "playground:composer.noAdditionalContext",
        "No additional pinned or external context"
      ),
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
        label={t(
          "playground:composer.context.generationStyle",
          "Generation style"
        )}
        before={formatGeneration(before, copy)}
        after={formatGeneration(after, copy)}
      />
      <PreviewRow
        label={t(
          "playground:composer.context.additionalContext",
          "Additional context"
        )}
        before={formatContext(before, copy)}
        after={formatContext(after, copy)}
      />
      {before.identity?.kind === "character" ||
      after.identity?.kind === "character" ? (
        <PreviewRow
          label={t("playground:composer.characterLore", "Character lore")}
          before={
            before.identity?.kind === "character"
              ? t(
                  "playground:composer.characterLoreNotChecked",
                  "Not checked in this preview"
                )
              : t(
                  "playground:composer.noCharacterSelected",
                  "No character selected"
                )
          }
          after={
            after.identity?.kind === "character"
              ? t(
                  "playground:composer.characterLoreNotChecked",
                  "Not checked in this preview"
                )
              : t(
                  "playground:composer.noCharacterSelected",
                  "No character selected"
                )
          }
        />
      ) : null}
    </section>
  )
}
