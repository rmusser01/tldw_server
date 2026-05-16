import React from "react"
import { Button, Tag, Typography } from "antd"
import { Images, PenLine, Sparkles, Upload } from "lucide-react"
import { useTranslation } from "react-i18next"

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

const formatStarterCount = (
  starterCount: number | undefined,
  starterCatalogLoading: boolean,
  t: (key: string, options: { defaultValue: string; count?: number }) => string
): string => {
  if (starterCatalogLoading) {
    return t("sidepanel:personaGarden.visuals.setup.checkingDefaults", {
      defaultValue: "Checking bundled defaults."
    })
  }
  if (starterCount === undefined) {
    return t("sidepanel:personaGarden.visuals.setup.defaultCountUnknown", {
      defaultValue: "Bundled default count unavailable."
    })
  }
  if (starterCount <= 0) {
    return t("sidepanel:personaGarden.visuals.setup.defaultCountZero", {
      defaultValue: "No bundled defaults available."
    })
  }
  if (starterCount === 1) {
    return t("sidepanel:personaGarden.visuals.setup.defaultCountOne", {
      count: starterCount,
      defaultValue: "1 bundled default available."
    })
  }
  return t("sidepanel:personaGarden.visuals.setup.defaultCountMany", {
    count: starterCount,
    defaultValue: "{{count}} bundled defaults available."
  })
}

const getDefaultDescription = ({
  recommendedStarter,
  starterCatalogError,
  starterCatalogLoading,
  t
}: Pick<
  VisualBuddySetupChoiceCardProps,
  "recommendedStarter" | "starterCatalogError" | "starterCatalogLoading"
> & {
  t: (key: string, options: { defaultValue: string }) => string
}): string => {
  if (starterCatalogLoading) {
    return t("sidepanel:personaGarden.visuals.setup.loadingDefaults", {
      defaultValue: "Loading bundled defaults."
    })
  }
  if (starterCatalogError) return starterCatalogError
  if (!recommendedStarter) {
    return t("sidepanel:personaGarden.visuals.setup.catalogUnavailable", {
      defaultValue: "Starter catalog unavailable."
    })
  }
  return (
    recommendedStarter.description ||
    t("sidepanel:personaGarden.visuals.setup.defaultFallbackDescription", {
      defaultValue: "Copy the recommended starter as a draft."
    })
  )
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
  const { t } = useTranslation(["sidepanel", "common"])

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
              {t("sidepanel:personaGarden.visuals.setup.compactEyebrow", {
                defaultValue: "Visual buddy"
              })}
            </div>
            <div className="mt-1 text-sm font-medium text-text">
              {selectedPersonaName}
            </div>
            <Text className="mt-1 block max-w-2xl text-xs leading-5 text-text-muted">
              {t("sidepanel:personaGarden.visuals.setup.compactHelp", {
                defaultValue:
                  "Add visuals after the guided assistant setup opens Persona Garden."
              })}
            </Text>
          </div>
          <Tag color="blue">
            {t("sidepanel:personaGarden.visuals.setup.optionalTag", {
              defaultValue: "optional"
            })}
          </Tag>
        </div>

        <Button
          className="mt-3 justify-center"
          size="small"
          icon={<Images className="h-3.5 w-3.5" />}
          disabled={!onOpenVisuals}
          onClick={onOpenVisuals}
        >
          {t("sidepanel:personaGarden.visuals.setup.openVisuals", {
            defaultValue: "Set up visual buddy"
          })}
        </Button>
      </div>
    )
  }

  const hasDrafts = !hasActiveVisual && packCount > 0
  const defaultDisabled =
    starterCatalogLoading || copyingDefault || !recommendedStarter || !onUseDefault
  const starterTitle =
    recommendedStarter?.title ||
    t("sidepanel:personaGarden.visuals.setup.recommendedDefault", {
      defaultValue: "Recommended default"
    })

  return (
    <div
      data-testid="visual-buddy-setup-choice-card"
      data-persona-id={selectedPersonaId}
      className="rounded-lg border border-border bg-surface p-3"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
            {t("sidepanel:personaGarden.visuals.setup.heading", {
              defaultValue: "Visual buddy setup"
            })}
          </div>
          <div className="mt-1 text-sm font-medium text-text">
            {selectedPersonaName}
          </div>
          <Text className="mt-1 block max-w-3xl text-xs leading-5 text-text-muted">
            {hasActiveVisual
              ? t("sidepanel:personaGarden.visuals.setup.activeDescription", {
                  defaultValue: "A visual buddy is active for this persona."
                })
              : hasDrafts
                ? t("sidepanel:personaGarden.visuals.setup.draftsDescription", {
                    defaultValue:
                      "Draft visual packs are ready to review. Activate one when it matches this persona."
                  })
                : t("sidepanel:personaGarden.visuals.setup.emptyDescription", {
                    defaultValue:
                      "No visual buddy is active for this persona. Choose a starter path to create an inactive pack for review."
                  })}
          </Text>
        </div>
        <Tag color={hasActiveVisual ? "green" : hasDrafts ? "blue" : "orange"}>
          {hasActiveVisual
            ? t("sidepanel:personaGarden.visuals.setup.activeTag", {
                defaultValue: "active"
              })
            : hasDrafts
              ? t("sidepanel:personaGarden.visuals.setup.reviewTag", {
                  defaultValue: "review"
                })
              : t("sidepanel:personaGarden.visuals.setup.firstRunTag", {
                  defaultValue: "first run"
                })}
        </Tag>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-3 xl:grid-cols-3">
        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">
              {t("sidepanel:personaGarden.visuals.setup.bundledDefault", {
                defaultValue: "Bundled default"
              })}
            </span>
            <Tag>
              {t("sidepanel:personaGarden.visuals.setup.copyTag", {
                defaultValue: "copy"
              })}
            </Tag>
          </div>
          <div className="mt-1 min-h-[3rem] text-xs leading-5 text-text-muted">
            <span className="font-medium text-text">{starterTitle}</span>
            <span className="block">
              {getDefaultDescription({
                recommendedStarter,
                starterCatalogError,
                starterCatalogLoading,
                t
              })}
            </span>
            <span className="block">
              {formatStarterCount(starterCount, starterCatalogLoading, t)}
            </span>
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
            {t("sidepanel:personaGarden.visuals.setup.useDefault", {
              defaultValue: "Use default"
            })}
          </Button>
          {onChooseDefault && starterCount && starterCount > 1 ? (
            <Button
              className="mt-2 w-full justify-center"
              size="small"
              icon={<Images className="h-3.5 w-3.5" />}
              disabled={starterCatalogLoading || !onChooseDefault}
              onClick={onChooseDefault}
            >
              {t("sidepanel:personaGarden.visuals.setup.chooseAnotherDefault", {
                defaultValue: "Choose another default"
              })}
            </Button>
          ) : null}
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">
              {t("sidepanel:personaGarden.visuals.setup.portableArchive", {
                defaultValue: "Portable archive"
              })}
            </span>
            <Tag>
              {t("sidepanel:personaGarden.visuals.setup.previewTag", {
                defaultValue: "preview"
              })}
            </Tag>
          </div>
          <div className="mt-1 min-h-[3rem] text-xs leading-5 text-text-muted">
            {t("sidepanel:personaGarden.visuals.setup.importDescription", {
              defaultValue:
                "Import a visual pack through the existing preview flow, then review and enable it when ready."
            })}
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<Upload className="h-3.5 w-3.5" />}
            disabled={!onImportPack}
            onClick={onImportPack}
          >
            {t("sidepanel:personaGarden.visuals.setup.importPack", {
              defaultValue: "Import pack"
            })}
          </Button>
        </div>

        <div className="rounded border border-border bg-bg p-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-medium text-text">
              {t("sidepanel:personaGarden.visuals.setup.emptyPack", {
                defaultValue: "Empty pack"
              })}
            </span>
            <Tag>
              {t("sidepanel:personaGarden.visuals.setup.blankTag", {
                defaultValue: "blank"
              })}
            </Tag>
          </div>
          <div className="mt-1 min-h-[3rem] text-xs leading-5 text-text-muted">
            {t("sidepanel:personaGarden.visuals.setup.blankDescription", {
              defaultValue:
                "Start blank when this persona needs custom states, poses, or assets instead of a bundled starter."
            })}
          </div>
          <Button
            className="mt-2 w-full justify-center"
            size="small"
            icon={<PenLine className="h-3.5 w-3.5" />}
            disabled={!onStartBlank}
            onClick={onStartBlank}
          >
            {t("sidepanel:personaGarden.visuals.setup.startBlank", {
              defaultValue: "Start blank"
            })}
          </Button>
        </div>
      </div>
    </div>
  )
}

export default VisualBuddySetupChoiceCard
