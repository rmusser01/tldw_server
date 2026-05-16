import React from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useTranslation } from "react-i18next";
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system";
import type {
  PlaygroundCompositionPreviewEntry,
  PlaygroundCompositionPreviewEntryState,
  PlaygroundCompositionPreviewState,
  PlaygroundCompositionPreviewSummary,
} from "./playground-composition-preview";

export type PlaygroundCompositionPreviewProps = {
  summary: PlaygroundCompositionPreviewSummary;
};

const stateClass = (state: PlaygroundCompositionPreviewEntryState) => {
  if (state === "active") return "border-success/40 bg-success/10 text-success";
  if (state === "degraded") return "border-warning/40 bg-warning/10 text-warning";
  if (state === "unavailable") return "border-danger/40 bg-danger/10 text-danger";
  if (state === "loading") return "border-info/40 bg-info/10 text-info";
  return "border-border bg-surface2 text-text-muted";
};

const overallLabel = (
  state: PlaygroundCompositionPreviewState,
  t: ReturnType<typeof useTranslation>["t"],
) => {
  const stateKeyByPreviewState = {
    ready: "ready",
    degraded: "degraded",
    unavailable: "unavailable",
  } satisfies Record<PlaygroundCompositionPreviewState, DesignSystemStateKey>;
  const stateDefinition = getDesignSystemState(stateKeyByPreviewState[state]);
  if (state === "ready") return t("cockpit.compositionReady", stateDefinition.label);
  if (state === "degraded") return t("cockpit.compositionDegraded", stateDefinition.label);
  return t("cockpit.compositionUnavailable", stateDefinition.label);
};

const entryStateLabel = (
  state: PlaygroundCompositionPreviewEntryState,
  t: ReturnType<typeof useTranslation>["t"],
) => {
  const stateKeyByEntryState = {
    active: "ready",
    degraded: "degraded",
    unavailable: "unavailable",
    loading: "loading",
    disabled: "empty",
  } satisfies Record<PlaygroundCompositionPreviewEntryState, DesignSystemStateKey>;
  const stateDefinition = getDesignSystemState(stateKeyByEntryState[state]);
  if (state === "active") return t("cockpit.active", "Active");
  if (state === "degraded") return t("cockpit.degraded", stateDefinition.label);
  if (state === "unavailable") return t("cockpit.unavailable", stateDefinition.label);
  if (state === "loading") return t("cockpit.loading", stateDefinition.label);
  return t("cockpit.disabled", "Disabled");
};

const pluralize = (count: number, singular: string, plural: string) =>
  `${count} ${count === 1 ? singular : plural}`;

const PreviewRow = ({
  entry,
}: {
  entry: PlaygroundCompositionPreviewEntry;
}) => {
  const { t } = useTranslation("playground");
  return (
    <li className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-2 py-1.5">
      <div className="min-w-0">
        <p className="text-[10px] font-semibold uppercase text-text-muted">
          {entry.label}
        </p>
        <p className="truncate text-xs font-medium text-text">{entry.title}</p>
        {entry.detail ? (
          <p className="mt-0.5 line-clamp-2 text-[11px] text-text-muted">
            {entry.detail}
          </p>
        ) : null}
      </div>
      <span
        className={`self-start rounded-full border px-1.5 py-0.5 text-[10px] font-medium ${stateClass(
          entry.state,
        )}`}
      >
        {entryStateLabel(entry.state, t)}
      </span>
    </li>
  );
};

export const PlaygroundCompositionPreview = ({
  summary,
}: PlaygroundCompositionPreviewProps) => {
  const { t } = useTranslation("playground");
  const [open, setOpen] = React.useState(false);
  const panelId = React.useId();
  const detailButtonLabel = open
    ? t("cockpit.hideCompositionDetails", "Hide composition details")
    : t("cockpit.showCompositionDetails", "Show composition details");
  const footprintItems = [
    pluralize(summary.footprint.providerMessageCount, "provider message", "provider messages"),
    pluralize(summary.footprint.previewSectionCount, "preview section", "preview sections"),
    pluralize(summary.footprint.contextPieceCount, "context piece", "context pieces"),
  ];

  if (summary.footprint.warningCount > 0) {
    footprintItems.push(
      pluralize(summary.footprint.warningCount, "warning", "warnings"),
    );
  }

  return (
    <section
      aria-label={t(
        "cockpit.nextMessageComposition",
        "Next message composition",
      )}
      className="min-w-0"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 className="text-[11px] font-semibold uppercase text-text-muted">
            {t("cockpit.composition", "Composition")}
          </h2>
          <p className="mt-1 text-sm font-medium text-text">
            {t("cockpit.nextMessage", "Next message")}
          </p>
          {summary.settingsScopeLabel ? (
            <p className="mt-0.5 truncate text-xs text-text-muted">
              {t("cockpit.settingsScope", `Scope: ${summary.settingsScopeLabel}`, {
                scope: summary.settingsScopeLabel,
              })}
            </p>
          ) : null}
        </div>
        <span
          className={`shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${stateClass(
            summary.overallState === "ready" ? "active" : summary.overallState,
          )}`}
        >
          {overallLabel(summary.overallState, t)}
        </span>
      </div>

      <ul className="mt-2 divide-y divide-border/60">
        {summary.entries
          .filter((entry) => entry.kind !== "composition")
          .map((entry) => (
            <PreviewRow key={entry.id} entry={entry} />
          ))}
      </ul>

      <button
        type="button"
        className="mt-2 inline-flex min-h-[30px] items-center gap-1 rounded-md border border-border bg-surface2 px-2.5 py-1 text-xs font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? (
          <ChevronUp className="h-3.5 w-3.5" aria-hidden="true" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
        )}
        {detailButtonLabel}
      </button>

      {open ? (
        <div id={panelId} className="mt-3 space-y-3">
          <div>
            <p className="text-[10px] font-semibold uppercase text-text-muted">
              {t("cockpit.contextStack", "Context stack")}
            </p>
            {summary.contextStack.length > 0 ? (
              <ul className="mt-1 divide-y divide-border/60">
                {summary.contextStack.map((entry) => (
                  <PreviewRow key={`stack-${entry.id}`} entry={entry} />
                ))}
              </ul>
            ) : (
              <p className="mt-1 text-xs text-text-muted">
                {t("cockpit.noContextStack", "No context sources active.")}
              </p>
            )}
          </div>
          <div>
            <p className="text-[10px] font-semibold uppercase text-text-muted">
              {t("cockpit.contextFootprint", "Context footprint")}
            </p>
            <ul className="mt-1 flex flex-wrap gap-1.5 text-xs text-text-muted">
              {footprintItems.map((item) => (
                <li
                  key={item}
                  className="rounded border border-border bg-surface2 px-2 py-0.5"
                >
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}
    </section>
  );
};
