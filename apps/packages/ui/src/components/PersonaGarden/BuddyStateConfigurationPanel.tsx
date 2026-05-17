import React from "react"
import { Button } from "antd"
import { Save, Settings2 } from "lucide-react"
import { useTranslation } from "react-i18next"

import { EmptyState as DesignSystemEmptyState } from "@/components/ui/feedback/EmptyState"
import { Badge, type BadgeVariant } from "@/components/ui/primitives"
import type { PersonaVisualManifest } from "@/types/persona-visuals"

import {
  summarizeBuddyStateConfiguration,
  type BuddyStateConfigurationState,
  type BuddyStateConfigurationTrigger
} from "./buddyBuilderState"

export type BuddyStateConfigurationPanelProps = {
  manifest?: PersonaVisualManifest | null
  canSave?: boolean
  saving?: boolean
  onSaveManifest?: () => void
}

const formatAnimationLabel = (state: BuddyStateConfigurationState): string =>
  `${state.label} animation`

const getStateBadgeVariant = (state: BuddyStateConfigurationState): BadgeVariant =>
  state.animationId ? "success" : "warning"

const StateRow: React.FC<{
  state: BuddyStateConfigurationState
  showDescription?: boolean
}> = ({ state, showDescription = false }) => (
  <li
    data-testid="buddy-state-config-state-row"
    data-state-id={state.id}
    className="rounded-md border border-border bg-surface p-2"
  >
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-1.5 text-xs font-medium text-text">
          <span>{state.label}</span>
          {state.required ? (
            <Badge variant="danger" size="sm">
              required
            </Badge>
          ) : null}
          {state.kind ? (
            <Badge variant="info" size="sm">
              {state.kind}
            </Badge>
          ) : null}
        </div>
        {showDescription && state.description ? (
          <div className="mt-1 text-xs leading-5 text-text-muted">
            {state.description}
          </div>
        ) : null}
        {state.tags.length ? (
          <div className="mt-1 flex flex-wrap gap-1">
            {state.tags.map((tag) => (
              <Badge key={`${state.id}-${tag}`} size="sm">
                {tag}
              </Badge>
            ))}
          </div>
        ) : null}
      </div>
      <Badge variant={getStateBadgeVariant(state)} size="sm">
        {state.animationId ? "mapped" : "missing"}
      </Badge>
    </div>
    <div className="mt-2 grid gap-2 sm:grid-cols-2">
      <label className="text-xs text-text-muted">
        <span className="mb-1 block">{formatAnimationLabel(state)}</span>
        <select
          aria-label={formatAnimationLabel(state)}
          className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
          disabled
          value={state.animationId || ""}
        >
          <option value="">Not mapped</option>
          {state.animationId ? (
            <option value={state.animationId}>{state.animationId}</option>
          ) : null}
        </select>
      </label>
      <div className="text-xs text-text-muted">
        <div className="mb-1 font-medium text-text-subtle">Fallbacks</div>
        <div>{state.fallbackIds.length ? state.fallbackIds.join(", ") : "None"}</div>
      </div>
    </div>
  </li>
)

const EmptySection: React.FC<{ title: string }> = ({ title }) => (
  <DesignSystemEmptyState
    title={title}
    variant="inline"
    size="sm"
    className="rounded border border-dashed border-border bg-bg px-3 py-2"
  />
)

const StateSection: React.FC<{
  title: string
  testId: string
  states: BuddyStateConfigurationState[]
  emptyText: string
  showDescription?: boolean
}> = ({ title, testId, states, emptyText, showDescription = false }) => (
  <section data-testid={testId} className="space-y-2">
    <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
      {title}
    </div>
    {states.length ? (
      <ul className="grid gap-2 md:grid-cols-2">
        {states.map((state) => (
          <StateRow
            key={state.id}
            state={state}
            showDescription={showDescription}
          />
        ))}
      </ul>
    ) : (
      <EmptySection title={emptyText} />
    )}
  </section>
)

const TriggerList: React.FC<{
  title: string
  testId: string
  triggers: BuddyStateConfigurationTrigger[]
  emptyText: string
}> = ({ title, testId, triggers, emptyText }) => (
  <section data-testid={testId} className="space-y-2">
    <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
      {title}
    </div>
    {triggers.length ? (
      <ul className="space-y-2">
        {triggers.map((trigger) => (
          <li
            key={trigger.id}
            className="rounded-md border border-border bg-surface p-2 text-xs"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-medium text-text">{trigger.match}</div>
              <Badge variant="primary" size="sm">
                {trigger.source}
              </Badge>
            </div>
            <div className="mt-1 text-text-muted">
              {trigger.stateLabel} ({trigger.durationMs}ms, priority{" "}
              {trigger.priority})
            </div>
          </li>
        ))}
      </ul>
    ) : (
      <EmptySection title={emptyText} />
    )}
  </section>
)

export const BuddyStateConfigurationPanel: React.FC<
  BuddyStateConfigurationPanelProps
> = ({
  manifest = null,
  canSave = false,
  saving = false,
  onSaveManifest
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const summary = summarizeBuddyStateConfiguration(manifest)
  const saveDisabled = !manifest || !canSave || !onSaveManifest

  return (
    <section data-testid="buddy-state-configuration-panel" className="space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm font-medium text-text">
            <Settings2 className="h-4 w-4" />
            {t("sidepanel:personaGarden.visuals.builder.configure.heading", {
              defaultValue: "Configure visual states"
            })}
          </div>
          <div className="mt-1 text-xs leading-5 text-text-muted">
            {t("sidepanel:personaGarden.visuals.builder.configure.description", {
              defaultValue:
                "Review the draft state map, movement states, and authored triggers before activation."
            })}
          </div>
        </div>
        <Button
          data-testid="buddy-state-config-save"
          size="small"
          type="primary"
          icon={<Save className="h-3.5 w-3.5" />}
          loading={saving}
          disabled={saveDisabled}
          onClick={onSaveManifest}
        >
          {t("sidepanel:personaGarden.visuals.builder.configure.save", {
            defaultValue: "Save visual state configuration"
          })}
        </Button>
      </div>

      <StateSection
        title={t("sidepanel:personaGarden.visuals.builder.configure.coreStates", {
          defaultValue: "Core states"
        })}
        testId="buddy-state-config-core-states"
        states={summary.coreStates}
        emptyText={t(
          "sidepanel:personaGarden.visuals.builder.configure.noCoreStates",
          { defaultValue: "No core states available." }
        )}
      />

      <StateSection
        title={t(
          "sidepanel:personaGarden.visuals.builder.configure.movementStates",
          { defaultValue: "Movement states" }
        )}
        testId="buddy-state-config-movement-states"
        states={summary.movementStates}
        emptyText={t(
          "sidepanel:personaGarden.visuals.builder.configure.noMovementStates",
          { defaultValue: "No movement states configured." }
        )}
        showDescription
      />

      <StateSection
        title={t("sidepanel:personaGarden.visuals.builder.configure.customStates", {
          defaultValue: "Custom states"
        })}
        testId="buddy-state-config-custom-states"
        states={summary.customStates}
        emptyText={t(
          "sidepanel:personaGarden.visuals.builder.configure.noCustomStates",
          { defaultValue: "No custom states configured." }
        )}
        showDescription
      />

      <section
        data-testid="buddy-state-config-triggers"
        className="grid gap-3 lg:grid-cols-3"
      >
        <TriggerList
          title={t(
            "sidepanel:personaGarden.visuals.builder.configure.toolNameTriggers",
            { defaultValue: "Exact tool-name triggers" }
          )}
          testId="buddy-state-config-tool-name-triggers"
          triggers={summary.toolNameTriggers}
          emptyText={t(
            "sidepanel:personaGarden.visuals.builder.configure.noToolNameTriggers",
            { defaultValue: "No exact tool-name triggers." }
          )}
        />
        <TriggerList
          title={t(
            "sidepanel:personaGarden.visuals.builder.configure.toolCategoryTriggers",
            { defaultValue: "Tool-category triggers" }
          )}
          testId="buddy-state-config-tool-category-triggers"
          triggers={summary.toolCategoryTriggers}
          emptyText={t(
            "sidepanel:personaGarden.visuals.builder.configure.noToolCategoryTriggers",
            { defaultValue: "No tool-category triggers." }
          )}
        />
        <TriggerList
          title={t(
            "sidepanel:personaGarden.visuals.builder.configure.runtimeTriggers",
            { defaultValue: "Live/runtime triggers" }
          )}
          testId="buddy-state-config-runtime-triggers"
          triggers={summary.runtimeTriggers}
          emptyText={t(
            "sidepanel:personaGarden.visuals.builder.configure.noRuntimeTriggers",
            { defaultValue: "No live/runtime triggers." }
          )}
        />
      </section>

      <div
        data-testid="buddy-state-config-advanced-hint"
        className="rounded-md border border-border bg-bg px-3 py-2 text-xs leading-5 text-text-muted"
      >
        {t("sidepanel:personaGarden.visuals.builder.configure.advancedHint", {
          defaultValue:
            "Use the advanced manifest controls below for inline edits to animations, fallbacks, and trigger rows."
        })}
      </div>
    </section>
  )
}

export default BuddyStateConfigurationPanel
