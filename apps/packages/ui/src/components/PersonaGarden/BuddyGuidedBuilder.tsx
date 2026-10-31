import React from "react"
import { Tag } from "antd"
import { CheckCircle2 } from "lucide-react"
import { useTranslation } from "react-i18next"

import type {
  PersonaVisualAsset,
  PersonaVisualImportPreviewResponse,
  PersonaVisualManifest,
  PersonaVisualStarterPackSummary
} from "@/types/persona-visuals"

import { BuddyDraftReviewPanel } from "./BuddyDraftReviewPanel"
import { BuddyImportFormatPanel } from "./BuddyImportFormatPanel"
import { BuddySourcePicker } from "./BuddySourcePicker"
import { BuddyStateConfigurationPanel } from "./BuddyStateConfigurationPanel"
import { BuddyStarterCatalogPicker } from "./BuddyStarterCatalogPicker"
import {
  resetBuddyBuilderForSource,
  type BuddyBuilderSource,
  type BuddyBuilderState
} from "./buddyBuilderState"

export type BuddyGuidedBuilderProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  hasActiveVisual: boolean
  packCount: number
  activePackTitle?: string | null
  starterPacks: PersonaVisualStarterPackSummary[]
  starterCatalogLoading?: boolean
  starterCatalogError?: string | null
  copyingStarterId?: string | null
  requestedSource?: BuddyBuilderSource | null
  requestedSourceRequestId?: number
  importPreviewPanel: React.ReactNode
  draftManifest?: PersonaVisualManifest | null
  assetsById?: Record<string, PersonaVisualAsset>
  importPreview?: PersonaVisualImportPreviewResponse | null
  activationBlockers?: string[]
  savingManifest?: boolean
  onRetryStarterCatalog?: () => void
  onCopyStarterPack: (starterPackId: string) => void
  onStartBlank?: () => void
  onOpenLibrary?: () => void
  onOpenDuplicate?: () => void
  onContinueToActivation?: () => void
  onSaveManifest?: () => void
}

const INITIAL_BUILDER_STATE: BuddyBuilderState = {
  source: "bundled",
  selectedStarterId: null,
  selectedImportFile: null,
  importPreview: null,
  selectedDraftPackId: null,
  activationReady: false
}

const formatTemplate = (
  template: string,
  values: Record<string, string | number>
): string => {
  let formatted = template
  for (const [key, value] of Object.entries(values)) {
    formatted = formatted.replaceAll(`{{${key}}}`, String(value))
  }
  return formatted
}

export const BuddyGuidedBuilder: React.FC<BuddyGuidedBuilderProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  hasActiveVisual,
  packCount,
  activePackTitle,
  starterPacks,
  starterCatalogLoading = false,
  starterCatalogError = null,
  copyingStarterId = null,
  requestedSource = null,
  requestedSourceRequestId = 0,
  importPreviewPanel,
  draftManifest = null,
  assetsById = {},
  importPreview = null,
  activationBlockers = [],
  savingManifest = false,
  onRetryStarterCatalog,
  onCopyStarterPack,
  onStartBlank,
  onOpenLibrary,
  onOpenDuplicate,
  onContinueToActivation,
  onSaveManifest
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [builderState, setBuilderState] =
    React.useState<BuddyBuilderState>(INITIAL_BUILDER_STATE)

  const selectSource = React.useCallback((source: BuddyBuilderSource) => {
    setBuilderState((current) => resetBuddyBuilderForSource(current, source))
  }, [])

  React.useEffect(() => {
    if (!requestedSource) return
    setBuilderState((current) => resetBuddyBuilderForSource(current, requestedSource))
  }, [requestedSource, requestedSourceRequestId])

  const selectedSource = builderState.source
  const displayActivePackTitle =
    activePackTitle ||
    t("sidepanel:personaGarden.visuals.builder.activePackFallback", {
      defaultValue: "Active visual buddy"
    })
  const packCountText =
    packCount === 1
      ? t("sidepanel:personaGarden.visuals.builder.packCountOne", {
          defaultValue: "1 pack"
        })
      : formatTemplate(
          t("sidepanel:personaGarden.visuals.builder.packCount", {
            defaultValue: "{{count}} packs"
          }),
          { count: packCount }
        )

  return (
    <section
      data-testid="buddy-guided-builder"
      data-persona-id={selectedPersonaId}
      aria-label={t("sidepanel:personaGarden.visuals.builder.choicesFor", { defaultValue: "Buddy choices for {{persona}}", persona: selectedPersonaName })}
      className="space-y-3"
    >
        {hasActiveVisual ? (
          <div
            data-testid="buddy-guided-builder-active-pack"
            className="rounded-md border border-border bg-bg px-3 py-2 text-xs"
          >
            <div className="flex items-center gap-2 font-medium text-text">
              <CheckCircle2 className="h-3.5 w-3.5 text-state-success" />
              {displayActivePackTitle}
            </div>
            <div className="mt-1 flex items-center gap-2 text-text-muted">
              <Tag>{t("sidepanel:personaGarden.visuals.builder.active", { defaultValue: "Active" })}</Tag>
              <span>
                {packCountText}
              </span>
            </div>
          </div>
        ) : null}

      {selectedSource === "bundled" ? (
        <BuddyStarterCatalogPicker
          starterPacks={starterPacks}
          loading={starterCatalogLoading}
          error={starterCatalogError}
          onRetry={onRetryStarterCatalog}
          copyingStarterId={copyingStarterId}
          onCopyStarterPack={onCopyStarterPack}
        />
      ) : null}

      <details open={selectedSource !== "bundled" || undefined} className="border-t border-border pt-3">
        <summary className="cursor-pointer rounded text-sm font-medium text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary">
          {t("sidepanel:personaGarden.visuals.builder.otherSources", { defaultValue: "Import or customize a Buddy" })}
        </summary>
        <div className="mt-3 space-y-3">
          <BuddySourcePicker selectedSource={selectedSource} onSelectSource={selectSource} onStartBlank={onStartBlank} onOpenLibrary={onOpenLibrary} onOpenDuplicate={onOpenDuplicate} />
          <BuddyImportFormatPanel source={selectedSource} importPreviewPanel={importPreviewPanel} />
        </div>
      </details>

      {draftManifest || importPreview ? (
        <BuddyDraftReviewPanel
          manifest={draftManifest}
          assetsById={assetsById}
          importPreview={importPreview}
          activationBlockers={activationBlockers}
          onContinueToActivation={onContinueToActivation}
        />
      ) : null}

      {draftManifest ? (
        <BuddyStateConfigurationPanel
          manifest={draftManifest}
          canSave={Boolean(onSaveManifest)}
          saving={savingManifest}
          onSaveManifest={onSaveManifest}
        />
      ) : null}

      {selectedSource === "blank" ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text-muted">
          {t("sidepanel:personaGarden.visuals.builder.blankHelp", {
            defaultValue:
              "Name your custom draft in the advanced creation section below."
          })}
        </div>
      ) : null}
      {selectedSource === "library" ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text-muted">
          {t("sidepanel:personaGarden.visuals.builder.libraryHelp", {
            defaultValue:
              "Choose a saved pack in the library section below."
          })}
        </div>
      ) : null}
      {selectedSource === "duplicate" ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text-muted">
          {t("sidepanel:personaGarden.visuals.builder.duplicateHelp", {
            defaultValue:
              "Choose a persona and pack in the duplicate section below."
          })}
        </div>
      ) : null}
    </section>
  )
}

export default BuddyGuidedBuilder
