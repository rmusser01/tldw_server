import React from "react"
import { Button } from "antd"
import { Copy, PackageCheck } from "lucide-react"
import { useTranslation } from "react-i18next"

import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import type { PersonaVisualStarterPackSummary } from "@/types/persona-visuals"

import {
  groupBuddyStarterPacksByTier,
  type BuddyStarterCatalogItem
} from "./buddyBuilderState"
import {
  formatStarterExpectedAssetGroups,
  getStarterComplexityTierLabel,
  getStarterProductionStatusLabel
} from "./VisualBuddySetupChoiceCard"

export type BuddyStarterCatalogPickerProps = {
  starterPacks: PersonaVisualStarterPackSummary[]
  copyingStarterId?: string | null
  loading?: boolean
  error?: string | null
  onCopyStarterPack: (starterPackId: string) => void
}

const TIER_ORDER = ["basic", "intermediate", "intricate"] as const

const getTierTitle = (
  tier: (typeof TIER_ORDER)[number],
  t: (key: string, options: { defaultValue: string }) => string
): string => {
  if (tier === "basic") {
    return t("sidepanel:personaGarden.visuals.builder.basicTier", {
      defaultValue: "Basic defaults"
    })
  }
  if (tier === "intermediate") {
    return t("sidepanel:personaGarden.visuals.builder.intermediateTier", {
      defaultValue: "Intermediate production packets"
    })
  }
  return t("sidepanel:personaGarden.visuals.builder.intricateTier", {
    defaultValue: "Intricate production packets"
  })
}

const renderStarterTags = (
  starter: BuddyStarterCatalogItem,
  t: (key: string, options: { defaultValue: string }) => string
): React.ReactNode => {
  const productionStatus = getStarterProductionStatusLabel(
    starter.production_status,
    t
  )
  const complexityTier = getStarterComplexityTierLabel(starter.complexity_tier, t)
  const productionVariant: BadgeVariant =
    starter.production_status === "art_ready" ? "success" : "warning"
  return (
    <div className="mt-2 flex flex-wrap gap-1">
      {starter.recommended ? (
        <Badge variant="success" size="sm">
          {t("sidepanel:personaGarden.visuals.builder.recommendedStarter", {
            defaultValue: "Recommended"
          })}
        </Badge>
      ) : null}
      {!starter.recommended ? (
        <Badge variant={productionVariant} size="sm">
          {t("sidepanel:personaGarden.visuals.builder.productionPacket", {
            defaultValue: "Production packet"
          })}
        </Badge>
      ) : null}
      {productionStatus ? (
        <Badge variant={productionVariant} size="sm">
          {productionStatus}
        </Badge>
      ) : null}
      {complexityTier ? (
        <Badge variant="secondary" size="sm">
          {complexityTier}
        </Badge>
      ) : null}
      {starter.neutral_anchor_required ? (
        <Badge variant="info" size="sm">
          {t("sidepanel:personaGarden.visuals.setup.neutralAnchorRequired", {
            defaultValue: "Neutral anchor required"
          })}
        </Badge>
      ) : null}
      {starter.tags.map((tag) => (
        <Badge key={tag} variant="secondary" size="sm">
          {tag}
        </Badge>
      ))}
      {starter.license_label ? (
        <Badge variant="secondary" size="sm">
          {starter.license_label}
        </Badge>
      ) : null}
    </div>
  )
}

export const BuddyStarterCatalogPicker: React.FC<
  BuddyStarterCatalogPickerProps
> = ({
  starterPacks,
  copyingStarterId = null,
  loading = false,
  error = null,
  onCopyStarterPack
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const grouped = groupBuddyStarterPacksByTier(starterPacks)

  return (
    <section data-testid="buddy-builder-starter-catalog" className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium text-text">
          <PackageCheck className="h-4 w-4" />
          {t("sidepanel:personaGarden.visuals.builder.defaultCatalog", {
            defaultValue: "Choose a bundled default"
          })}
        </div>
        {loading ? (
          <Badge size="sm">
            {t("sidepanel:personaGarden.visuals.builder.loading", {
              defaultValue: "loading"
            })}
          </Badge>
        ) : null}
      </div>
      {error ? (
        <div className="rounded border border-state-error/50 bg-state-error/10 p-2 text-xs text-state-error">
          {error}
        </div>
      ) : null}
      {TIER_ORDER.map((tier) => {
        const starters = grouped[tier]
        if (!starters.length) return null
        return (
          <section
            key={tier}
            data-testid={`buddy-builder-tier-${tier}`}
            className="space-y-2"
          >
            <div className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
              {getTierTitle(tier, t)}
            </div>
            <div className="grid gap-2 lg:grid-cols-2">
              {starters.map((starter) => {
                const expectedAssetGroups = formatStarterExpectedAssetGroups(
                  starter.expected_asset_groups
                )
                const coverage = starter.animation_coverage_notes.join("; ")
                const copyLabel = starter.recommended
                  ? t("sidepanel:personaGarden.visuals.builder.copyRecommended", {
                      defaultValue: "Copy as draft"
                    })
                  : t("sidepanel:personaGarden.visuals.builder.copyPacket", {
                      defaultValue: "Copy production packet"
                    })
                return (
                  <div
                    key={starter.id}
                    data-testid={`buddy-builder-starter-${starter.id}`}
                    className="rounded-md border border-border bg-bg p-3 text-xs"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div className="min-w-0">
                        <div
                          data-testid="buddy-builder-starter-title"
                          className="text-sm font-medium text-text"
                        >
                          {starter.title}
                        </div>
                        <div className="mt-1 leading-5 text-text-muted">
                          {starter.description}
                        </div>
                      </div>
                      <Badge size="sm">{starter.renderer_type}</Badge>
                    </div>
                    {renderStarterTags(starter, t)}
                    {expectedAssetGroups ? (
                      <div className="mt-2 leading-5 text-text-muted">
                        <span className="font-medium text-text">
                          {t(
                            "sidepanel:personaGarden.visuals.setup.expectedAssetsLabel",
                            {
                              defaultValue: "Expected assets:"
                            }
                          )}{" "}
                        </span>
                        {expectedAssetGroups}
                      </div>
                    ) : null}
                    {coverage ? (
                      <div className="mt-1 leading-5 text-text-muted">
                        <span className="font-medium text-text">
                          {t("sidepanel:personaGarden.visuals.setup.coverageLabel", {
                            defaultValue: "Coverage:"
                          })}{" "}
                        </span>
                        {coverage}
                      </div>
                    ) : null}
                    <Button
                      className="mt-3"
                      size="small"
                      type={starter.recommended ? "primary" : "default"}
                      icon={<Copy className="h-3.5 w-3.5" />}
                      loading={copyingStarterId === starter.id}
                      disabled={Boolean(copyingStarterId)}
                      onClick={() => onCopyStarterPack(starter.id)}
                    >
                      {copyLabel}
                    </Button>
                  </div>
                )
              })}
            </div>
          </section>
        )
      })}
    </section>
  )
}

export default BuddyStarterCatalogPicker
