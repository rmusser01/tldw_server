import React from 'react';
import { PanelLeftOpen, PanelRightOpen } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type {
  PlaygroundCompositionPreviewEntry,
  PlaygroundCompositionPreviewEntryState,
  PlaygroundCompositionPreviewSummary,
} from './playground-composition-preview';

export type PlaygroundCollapsedCompositionSummaryProps = {
  summary: PlaygroundCompositionPreviewSummary;
  contextRailVisible: boolean;
  runtimeRailVisible: boolean;
  onRestoreContextRail?: () => void;
  onRestoreRuntimeRail?: () => void;
};

const compactEntryIds = ['model', 'assistant', 'prompt', 'context', 'tools'];

const stateClass = (state: PlaygroundCompositionPreviewEntryState) => {
  if (state === 'active') return 'border-success/30 bg-success/10 text-text';
  if (state === 'degraded') return 'border-warning/40 bg-warning/10 text-text';
  if (state === 'unavailable') return 'border-danger/40 bg-danger/10 text-text';
  if (state === 'loading') return 'border-info/40 bg-info/10 text-text';
  return 'border-border bg-surface2 text-text-muted';
};

const CompactEntryChip = ({ entry }: { entry: PlaygroundCompositionPreviewEntry }) => {
  const label = `${entry.label}: ${entry.title}${entry.detail ? `. ${entry.detail}` : ''}`;

  return (
    <li
      aria-label={label}
      className={`grid min-h-[44px] min-w-[8.75rem] max-w-full grid-cols-1 rounded-md border px-2 py-1.5 ${stateClass(
        entry.state
      )}`}
    >
      <span className="truncate text-[10px] font-semibold uppercase text-text-muted">
        {entry.label}
      </span>
      <span className="truncate text-xs font-medium text-text">{entry.title}</span>
      {entry.detail ? (
        <span className="truncate text-[10px] text-text-muted">{entry.detail}</span>
      ) : null}
    </li>
  );
};

export const PlaygroundCollapsedCompositionSummary = ({
  summary,
  contextRailVisible,
  runtimeRailVisible,
  onRestoreContextRail,
  onRestoreRuntimeRail,
}: PlaygroundCollapsedCompositionSummaryProps) => {
  const { t } = useTranslation('playground');

  if (contextRailVisible && runtimeRailVisible) {
    return null;
  }

  const entries = compactEntryIds
    .map((entryId) => summary.entries.find((entry) => entry.id === entryId))
    .filter((entry): entry is PlaygroundCompositionPreviewEntry => Boolean(entry));
  const hiddenLabels = [
    !contextRailVisible ? t('cockpit.contextHidden', 'Context hidden') : null,
    !runtimeRailVisible ? t('cockpit.runtimeHidden', 'Runtime hidden') : null,
  ].filter((label): label is string => Boolean(label));

  return (
    <section
      data-testid="playground-collapsed-composition-summary"
      aria-label={t('cockpit.collapsedCompositionSummary', 'Collapsed cockpit summary')}
      className="mx-auto w-full max-w-[64rem] px-4 pt-2"
    >
      <div className="rounded-md border border-border bg-surface/95 px-2.5 py-2 text-xs shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase text-text-muted">
              {t('cockpit.hiddenCockpitState', 'Hidden cockpit state')}
            </p>
            <p className="mt-0.5 truncate text-[11px] text-text-muted">
              {hiddenLabels.join(' / ')}
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap items-center gap-1.5">
            {!contextRailVisible ? (
              <button
                type="button"
                aria-label={t('cockpit.restoreContextRail', 'Restore context rail')}
                title={t('cockpit.restoreContextRail', 'Restore context rail')}
                onClick={onRestoreContextRail}
                className="inline-flex min-h-[30px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-1 text-[11px] font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <PanelLeftOpen className="h-3.5 w-3.5" aria-hidden="true" />
                <span>{t('cockpit.context', 'Context')}</span>
              </button>
            ) : null}
            {!runtimeRailVisible ? (
              <button
                type="button"
                aria-label={t('cockpit.restoreRuntimeRail', 'Restore runtime rail')}
                title={t('cockpit.restoreRuntimeRail', 'Restore runtime rail')}
                onClick={onRestoreRuntimeRail}
                className="inline-flex min-h-[30px] items-center gap-1 rounded-md border border-border bg-surface2 px-2 py-1 text-[11px] font-medium text-text hover:bg-surface focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <PanelRightOpen className="h-3.5 w-3.5" aria-hidden="true" />
                <span>{t('cockpit.runtime', 'Runtime')}</span>
              </button>
            ) : null}
          </div>
        </div>
        <ul className="mt-2 grid grid-cols-1 gap-1.5 sm:grid-cols-2 xl:grid-cols-5">
          {entries.map((entry) => (
            <CompactEntryChip key={entry.id} entry={entry} />
          ))}
        </ul>
      </div>
    </section>
  );
};
