import React from "react"
import { Button, Tag, Typography } from "antd"
import { Images, PenLine, Sparkles, Upload } from "lucide-react"

import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"

const { Text } = Typography

export type VisualBuddySetupChoiceCardProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  hasActiveVisual: boolean
  packCount: number
  recommendedStarter?: PersonaVisualStarterPackSummary | null
  starterCount?: number
  starterCatalogLoading?: boolean
  starterCatalogError?: string | null
  copyingDefault?: boolean
  compact?: boolean
  onUseDefault?: () => void
  onChooseDefault?: () => void
  onImportPack?: () => void
  onStartBlank?: () => void
  onOpenVisuals?: () => void
}

const formatStarterCount = (starterCount: number | undefined): string => {
  if (!starterCount) return "No bundled defaults available."
  if (starterCount === 1) return "1 bundled default available."
  return `${starterCount} bundled defaults available.`
}

const getDefaultDescription = ({
  recommendedStarter,
  starterCatalogError,
  starterCatalogLoading
}: Pick<
  VisualBuddySetupChoiceCardProps,
  "recommendedStarter" | "starterCatalogError" | "starterCatalogLoading"
>): string => {
  if (starterCatalogLoading) return "Loading bundled defaults."
  if (starterCatalogError) return starterCatalogError
  if (!recommendedStarter) return "Starter catalog unavailable."
  return recommendedStarter.description || "Copy the recommended starter as a draft."
}

export const VisualBuddySetupChoiceCard: React.FC<
  VisualBuddySetupChoiceCardProps
> = ({
  selectedPersonaId,
  selectedPersonaName,
  hasActiveVisual,
  packCount,
  recommendedStarter = null,
  starterCount,
  starterCatalogLoading = false,
  starterCatalogError = null,
  copyingDefault = false,
  compact = false,
  onUseDefault,
  onChooseDefault,
  onImportPack,
  onStartBlank,
  onOpenVisuals
}) => {
  if (compact) {
    return (
      <div
        data-testid="visual-buddy-setup-choice-card"
        data-persona-id={selectedPersonaId}
        className="rounded-lg border border-border bg-surface p-3"
      >
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
              Visual buddy
            </div>
            <div className="mt-1 text-sm font-medium text-text">
              {selectedPersonaName}
            </div>
            <Text className="mt-1 block max-w-2xl text-xs leading-5 text-text-muted">
              Add visuals after the guided assistant setup opens Persona Garden.
            </Text>
          </div>
          <Tag color="blue">optional</Tag>
        </div>

        <Button
          className="mt-3 justify-center"
          size="small"
          icon={<Images className="h-3.5 w-3.5" />}
          onClick={onOpenVisuals}
        >
          Set up visual buddy
        </Button>
      </div>
    )
  }

  const hasDrafts = !hasActiveVisual && packCount > 0
  const defaultDisabled =
    starterCatalogLoading || copyingDefault || !recommendedStarter
  const starterTitle = recommendedStarter?.title || "Recommended default"

  return (
    <div
      data-testid="visual-buddy-setup-choice-card"
      data-persona-id={selectedPersonaId}
      className="rounded-lg border border-border bg-surface p-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            Visual buddy setup
          </div>
          <div className="mt-1 text-sm font-medium text-text">
            {selectedPersonaName}
          </div>
          <Text className="mt-1 block max-w-3xl text-xs leading-5 text-text-muted">
            {hasActiveVisual
              ? "A visual buddy is active for this persona."
              : hasDrafts
                ? "Draft visual packs are ready to review. Activate one when it matches this persona."
                : "No visual buddy is active for this persona. Choose a starter path to create an inactive pack for review."}
          </Text>
        </div>
        <Tag color={hasActiveVisual ? "green" : hasDrafts ? "blue" : "orange"}>
          {hasActiveVisual ? "active" : hasDrafts ? "review" : "first run"}
        </Tag>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-3 xl:grid-cols-4">
        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Bundled default</span>
            <Tag>copy</Tag>
          </div>
          <div className="mt-1 min-h-[3rem] text-xs leading-5 text-text-muted">
            <span className="font-medium text-text">{starterTitle}</span>
            <span className="block">
              {getDefaultDescription({
                recommendedStarter,
                starterCatalogError,
                starterCatalogLoading
              })}
            </span>
            <span className="block">{formatStarterCount(starterCount)}</span>
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            type="primary"
            icon={<Sparkles className="h-3.5 w-3.5" />}
            disabled={defaultDisabled}
            loading={copyingDefault}
            onClick={onUseDefault}
          >
            Use default
          </Button>
          {onChooseDefault && starterCount && starterCount > 1 ? (
            <Button
              className="mt-2 w-full justify-center"
              size="small"
              icon={<Images className="h-3.5 w-3.5" />}
              disabled={starterCatalogLoading}
              onClick={onChooseDefault}
            >
              Choose another default
            </Button>
          ) : null}
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Portable archive</span>
            <Tag>preview</Tag>
          </div>
          <div className="mt-1 min-h-[3rem] text-xs leading-5 text-text-muted">
            Import a visual pack through the existing preview flow, then review
            and enable it when ready.
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<Upload className="h-3.5 w-3.5" />}
            onClick={onImportPack}
          >
            Import pack
          </Button>
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">Empty pack</span>
            <Tag>blank</Tag>
          </div>
          <div className="mt-1 min-h-[3rem] text-xs leading-5 text-text-muted">
            Start blank when this persona needs custom states, poses, or assets
            instead of a bundled starter.
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<PenLine className="h-3.5 w-3.5" />}
            onClick={onStartBlank}
          >
            Start blank
          </Button>
        </div>
      </div>
    </div>
  )
}

export default VisualBuddySetupChoiceCard
