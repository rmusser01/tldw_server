import React from "react"
import { Button, Tag } from "antd"
import { AlertTriangle, CheckCircle2, Eye, FileArchive } from "lucide-react"
import { useTranslation } from "react-i18next"

import type {
  PersonaVisualAsset,
  PersonaVisualImportPreviewResponse,
  PersonaVisualManifest
} from "@/types/persona-visuals"

import { SpriteFrameRenderer } from "@/components/Common/PersonaBuddy/SpriteFrameRenderer"
import { summarizeBuddyDraftReadiness } from "./buddyBuilderState"

export type BuddyDraftReviewPanelProps = {
  manifest?: PersonaVisualManifest | null
  assetsById?: Record<string, PersonaVisualAsset>
  importPreview?: PersonaVisualImportPreviewResponse | null
  activationBlockers?: string[]
  onContinueToActivation?: () => void
}

const formatResolved = (
  resolved: boolean,
  t: (key: string, options: { defaultValue: string }) => string
): string =>
  resolved
    ? t("sidepanel:personaGarden.visuals.builder.review.readyTag", {
        defaultValue: "ready"
      })
    : t("sidepanel:personaGarden.visuals.builder.review.missingTag", {
        defaultValue: "missing"
      })

export const BuddyDraftReviewPanel: React.FC<BuddyDraftReviewPanelProps> = ({
  manifest = null,
  assetsById = {},
  importPreview = null,
  activationBlockers = [],
  onContinueToActivation
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const summary = summarizeBuddyDraftReadiness({
    manifest,
    importPreview,
    activationBlockers
  })
  const canRenderPreview =
    Boolean(manifest?.states?.idle?.animation_id) && Object.keys(assetsById).length > 0

  return (
    <section data-testid="buddy-draft-review-panel" className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium text-text">
            <FileArchive className="h-4 w-4" />
            {t("sidepanel:personaGarden.visuals.builder.review.heading", {
              defaultValue: "Review draft readiness"
            })}
          </div>
          <div className="mt-1 text-xs leading-5 text-text-muted">
            {summary.sourceLabel}{" "}
            {t("sidepanel:personaGarden.visuals.builder.review.draftSemantics", {
              defaultValue: "imported as a Persona Visual draft"
            })}
          </div>
        </div>
        <Tag color={summary.canActivate ? "green" : "orange"}>
          {summary.canActivate
            ? t("sidepanel:personaGarden.visuals.builder.review.readyTag", {
                defaultValue: "ready"
              })
            : t("sidepanel:personaGarden.visuals.builder.review.needsReviewTag", {
                defaultValue: "needs review"
              })}
        </Tag>
      </div>

      <div
        data-testid="buddy-draft-review-preview"
        className="rounded-md border border-border bg-bg p-3"
      >
        <div className="flex items-center gap-2 text-xs font-medium text-text">
          <Eye className="h-3.5 w-3.5" />
          {t("sidepanel:personaGarden.visuals.builder.review.previewHeading", {
            defaultValue: "Draft preview"
          })}
        </div>
        <div className="mt-2">
          {manifest && canRenderPreview ? (
            <SpriteFrameRenderer
              manifest={manifest}
              assets={assetsById}
              state="idle"
              fallbackLabel={t(
                "sidepanel:personaGarden.visuals.builder.review.previewFallback",
                { defaultValue: "Buddy preview" }
              )}
            />
          ) : (
            <div className="text-xs text-text-muted">
              {t(
                "sidepanel:personaGarden.visuals.builder.review.previewUnavailable",
                { defaultValue: "Preview unavailable" }
              )}
            </div>
          )}
        </div>
      </div>

      {summary.atlasSummary.length ? (
        <div
          data-testid="buddy-draft-review-atlas"
          className="rounded-md border border-border bg-bg p-3 text-xs"
        >
          <div className="font-medium text-text">
            {t("sidepanel:personaGarden.visuals.builder.review.atlasHeading", {
              defaultValue: "Atlas metadata"
            })}
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-5 text-text-muted">
            {summary.atlasSummary.map((atlas, index) => (
              <li key={`${atlas.assetId || "atlas"}-${index}`}>
                <span className="text-text">{atlas.assetId || "atlas"}</span>
                {": "}
                {atlas.width ?? "unknown"}x{atlas.height ?? "unknown"}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-2">
        <section
          data-testid="buddy-draft-review-required-states"
          className="rounded-md border border-border bg-bg p-3 text-xs"
        >
          <div className="font-medium text-text">
            {t("sidepanel:personaGarden.visuals.builder.review.requiredStates", {
              defaultValue: "Required states"
            })}
          </div>
          <ul className="mt-1 space-y-1 text-text-muted">
            {summary.requiredStates.map((state) => (
              <li key={state.id} className="flex items-center justify-between gap-2">
                <span>{state.id}</span>
                <Tag color={state.resolved ? "green" : "orange"}>
                  {formatResolved(state.resolved, t)}
                </Tag>
              </li>
            ))}
          </ul>
        </section>

        <section
          data-testid="buddy-draft-review-movement-states"
          className="rounded-md border border-border bg-bg p-3 text-xs"
        >
          <div className="font-medium text-text">
            {t("sidepanel:personaGarden.visuals.builder.review.movementStates", {
              defaultValue: "Movement states"
            })}
          </div>
          {summary.movementStates.length ? (
            <ul className="mt-1 space-y-1 text-text-muted">
              {summary.movementStates.map((state) => (
                <li key={state.id} className="flex items-center justify-between gap-2">
                  <span>{state.id}</span>
                  <Tag color={state.resolved ? "green" : "orange"}>
                    {formatResolved(state.resolved, t)}
                  </Tag>
                </li>
              ))}
            </ul>
          ) : (
            <div className="mt-1 text-text-muted">
              {t("sidepanel:personaGarden.visuals.builder.review.noMovementStates", {
                defaultValue: "No movement states."
              })}
            </div>
          )}
        </section>
      </div>

      <section
        data-testid="buddy-draft-review-custom-states"
        className="rounded-md border border-border bg-bg p-3 text-xs"
      >
        <div className="font-medium text-text">
          {t("sidepanel:personaGarden.visuals.builder.review.customStates", {
            defaultValue: "Custom states"
          })}
        </div>
        {summary.customStates.length ? (
          <ul className="mt-1 list-disc space-y-1 pl-5 text-text-muted">
            {summary.customStates.map((state) => (
              <li key={state.id}>
                <span className="text-text">{state.label}</span> ({state.kind})
                {state.fallback ? ` -> ${state.fallback}` : ""}
              </li>
            ))}
          </ul>
        ) : (
          <div className="mt-1 text-text-muted">
            {t("sidepanel:personaGarden.visuals.builder.review.noCustomStates", {
              defaultValue: "No custom states."
            })}
          </div>
        )}
      </section>

      {summary.warnings.length ? (
        <div
          data-testid="buddy-draft-review-warnings"
          className="rounded-md border border-warning/40 bg-warning/10 p-3 text-xs text-text-muted"
        >
          <div className="font-medium text-text">
            {t("sidepanel:personaGarden.visuals.builder.review.warnings", {
              defaultValue: "Warnings"
            })}
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {summary.warnings.map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {summary.blockers.length ? (
        <div
          data-testid="buddy-draft-review-blockers"
          className="rounded-md border border-state-error/40 bg-state-error/10 p-3 text-xs text-state-error"
        >
          <div className="flex items-center gap-2 font-medium">
            <AlertTriangle className="h-3.5 w-3.5" />
            {t("sidepanel:personaGarden.visuals.builder.review.blockers", {
              defaultValue: "Activation blockers"
            })}
          </div>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {summary.blockers.map((blocker, index) => (
              <li key={`${blocker}-${index}`}>{blocker}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <Button
        data-testid="buddy-draft-review-activation-path"
        size="small"
        type="primary"
        icon={<CheckCircle2 className="h-3.5 w-3.5" />}
        disabled={!summary.canActivate || !onContinueToActivation}
        onClick={onContinueToActivation}
      >
        {t("sidepanel:personaGarden.visuals.builder.review.activationPath", {
          defaultValue: "Continue to activation"
        })}
      </Button>
    </section>
  )
}

export default BuddyDraftReviewPanel
