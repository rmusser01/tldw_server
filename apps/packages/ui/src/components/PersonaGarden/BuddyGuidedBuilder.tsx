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
import { BuddyStarterCatalogPicker } from "./BuddyStarterCatalogPicker"
import {
  BUDDY_BUILDER_STEPS,
  resetBuddyBuilderForSource,
  type BuddyBuilderSource,
  type BuddyBuilderState,
  type BuddyBuilderStep
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
  importPreviewPanel: React.ReactNode
  draftManifest?: PersonaVisualManifest | null
  assetsById?: Record<string, PersonaVisualAsset>
  importPreview?: PersonaVisualImportPreviewResponse | null
  activationBlockers?: string[]
  onCopyStarterPack: (starterPackId: string) => void
  onStartBlank?: () => void
  onOpenLibrary?: () => void
  onOpenDuplicate?: () => void
}

const INITIAL_BUILDER_STATE: BuddyBuilderState = {
  source: "bundled",
  selectedStarterId: null,
  selectedImportFile: null,
  importPreview: null,
  selectedDraftPackId: null,
  activationReady: false
}

const getStepLabel = (
  step: BuddyBuilderStep,
  t: (key: string, options: { defaultValue: string }) => string
): string => {
  if (step === "source") {
    return t("sidepanel:personaGarden.visuals.builder.sourceStep", {
      defaultValue: "Choose a source"
    })
  }
  if (step === "draft") {
    return t("sidepanel:personaGarden.visuals.builder.draftStep", {
      defaultValue: "Create a draft"
    })
  }
  if (step === "review") {
    return t("sidepanel:personaGarden.visuals.builder.reviewStep", {
      defaultValue: "Review readiness"
    })
  }
  if (step === "configure") {
    return t("sidepanel:personaGarden.visuals.builder.configureStep", {
      defaultValue: "Configure states"
    })
  }
  return t("sidepanel:personaGarden.visuals.builder.activateStep", {
    defaultValue: "Activate"
  })
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
  importPreviewPanel,
  draftManifest = null,
  assetsById = {},
  importPreview = null,
  activationBlockers = [],
  onCopyStarterPack,
  onStartBlank,
  onOpenLibrary,
  onOpenDuplicate
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [builderState, setBuilderState] =
    React.useState<BuddyBuilderState>(INITIAL_BUILDER_STATE)

  const selectSource = React.useCallback((source: BuddyBuilderSource) => {
    setBuilderState((current) => resetBuddyBuilderForSource(current, source))
  }, [])

  const selectedSource = builderState.source
  const displayActivePackTitle =
    activePackTitle ||
    t("sidepanel:personaGarden.visuals.builder.activePackFallback", {
      defaultValue: "Active visual buddy"
    })

  return (
    <section
      data-testid="buddy-guided-builder"
      data-persona-id={selectedPersonaId}
      className="space-y-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("sidepanel:personaGarden.visuals.builder.eyebrow", {
              defaultValue: "Persona visuals"
            })}
          </div>
          <h2 className="mt-1 text-base font-semibold text-text">
            {t("sidepanel:personaGarden.visuals.builder.heading", {
              defaultValue: "Buddy builder"
            })}
          </h2>
          <div className="mt-1 text-xs leading-5 text-text-muted">
            {t("sidepanel:personaGarden.visuals.builder.description", {
              persona: selectedPersonaName,
              defaultValue:
                "Choose, import, review, configure, and activate a visual buddy for {{persona}}."
            })}
          </div>
        </div>
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
              <Tag color="green">active</Tag>
              <span>
                {t("sidepanel:personaGarden.visuals.builder.packCount", {
                  count: packCount,
                  defaultValue: "{{count}} packs"
                })}
              </span>
            </div>
          </div>
        ) : null}
      </div>

      <ol
        aria-label={t("sidepanel:personaGarden.visuals.builder.stepperAria", {
          defaultValue: "Buddy builder steps"
        })}
        className="grid gap-1 text-xs sm:grid-cols-5"
      >
        {BUDDY_BUILDER_STEPS.map((step, index) => (
          <li
            key={step}
            className="rounded border border-border bg-bg px-2 py-1 text-text-muted"
          >
            <span className="font-medium text-text">{index + 1}. </span>
            {getStepLabel(step, t)}
          </li>
        ))}
      </ol>

      <BuddySourcePicker
        selectedSource={selectedSource}
        onSelectSource={selectSource}
        onStartBlank={onStartBlank}
        onOpenLibrary={onOpenLibrary}
        onOpenDuplicate={onOpenDuplicate}
      />

      {selectedSource === "bundled" ? (
        <BuddyStarterCatalogPicker
          starterPacks={starterPacks}
          loading={starterCatalogLoading}
          error={starterCatalogError}
          copyingStarterId={copyingStarterId}
          onCopyStarterPack={onCopyStarterPack}
        />
      ) : null}

      <BuddyImportFormatPanel
        source={selectedSource}
        importPreviewPanel={importPreviewPanel}
      />

      {draftManifest || importPreview ? (
        <BuddyDraftReviewPanel
          manifest={draftManifest}
          assetsById={assetsById}
          importPreview={importPreview}
          activationBlockers={activationBlockers}
        />
      ) : null}

      {selectedSource === "blank" ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text-muted">
          {t("sidepanel:personaGarden.visuals.builder.blankHelp", {
            defaultValue:
              "Blank drafts use the existing title and create-draft controls below."
          })}
        </div>
      ) : null}
      {selectedSource === "library" ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text-muted">
          {t("sidepanel:personaGarden.visuals.builder.libraryHelp", {
            defaultValue:
              "Use the existing library panel below to attach or duplicate saved packs."
          })}
        </div>
      ) : null}
      {selectedSource === "duplicate" ? (
        <div className="rounded-md border border-border bg-bg p-3 text-xs leading-5 text-text-muted">
          {t("sidepanel:personaGarden.visuals.builder.duplicateHelp", {
            defaultValue:
              "Use the existing duplicate controls below to copy a pack from another persona."
          })}
        </div>
      ) : null}
    </section>
  )
}

export default BuddyGuidedBuilder
