import React, { useMemo } from 'react';
import { Badge } from '@web/components/ui/Badge';
import type { VNPlayEvent, VNPlaySceneAsset, VNPlaySceneState } from '@web/types/vn-play';

export interface SceneStageProps {
  events: VNPlayEvent[];
  sceneState: VNPlaySceneState;
  showDialogue?: boolean;
}

interface DialogueLine {
  speaker: string;
  text: string;
}

function assetUrl(asset: VNPlaySceneAsset | Record<string, unknown> | null | undefined): string | null {
  if (!asset || typeof asset !== 'object') return null;
  for (const key of ['content_url', 'url', 'src']) {
    const value = asset[key as keyof typeof asset];
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }
  return null;
}

function assetLabel(asset: VNPlaySceneAsset | Record<string, unknown> | null | undefined): string | null {
  if (!asset || typeof asset !== 'object') return null;
  const labels = 'labels' in asset && asset.labels && typeof asset.labels === 'object'
    ? asset.labels as Record<string, unknown>
    : null;
  const metadata = 'metadata' in asset && asset.metadata && typeof asset.metadata === 'object'
    ? asset.metadata as Record<string, unknown>
    : null;
  const value =
    labels?.name ??
    labels?.display_name ??
    labels?.location ??
    labels?.emotion ??
    metadata?.name ??
    metadata?.pose;
  return typeof value === 'string' && value.trim() ? value : null;
}

function sceneMetadata(sceneState: VNPlaySceneState): Array<[string, string]> {
  return [
    ['Location', sceneState.location_key],
    ['Mood', sceneState.mood],
    ['Time', sceneState.time_of_day],
    ['Weather', sceneState.weather],
  ].filter((entry): entry is [string, string] => typeof entry[1] === 'string' && entry[1].trim().length > 0);
}

function latestDialogue(events: VNPlayEvent[]): DialogueLine[] {
  for (const event of [...events].reverse()) {
    const payload = event.event_payload ?? {};
    const rawDialogue = payload.dialogue;
    if (Array.isArray(rawDialogue)) {
      const lines = rawDialogue
        .filter((item): item is Record<string, unknown> => item !== null && typeof item === 'object')
        .map((item) => ({
          speaker: typeof item.speaker === 'string' && item.speaker.trim() ? item.speaker : 'Narrator',
          text: typeof item.text === 'string' ? item.text : '',
        }))
        .filter((line) => line.text.trim());
      if (lines.length > 0) return lines;
    }

    const narration = payload.narrative_text ?? payload.narration;
    if (typeof narration === 'string' && narration.trim()) {
      return [{ speaker: 'Narrator', text: narration }];
    }
  }
  return [];
}

function warningText(warning: unknown): string {
  if (warning && typeof warning === 'object') {
    const record = warning as Record<string, unknown>;
    const reason = record.message ?? record.reason ?? record.code ?? record.event_type ?? 'warning';
    const details = [record.asset_type, record.slot_key]
      .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
      .join(' ');
    return details ? `${String(reason)} (${details})` : String(reason);
  }
  return String(warning);
}

export default function SceneStage({ events, sceneState, showDialogue = true }: SceneStageProps) {
  const backgroundUrl = assetUrl(sceneState.background);
  const depthUrl = assetUrl(sceneState.depth);
  const sprites = sceneState.active_sprites ?? sceneState.active_sprite_items ?? [];
  const spriteUrls = sprites.map((sprite) => ({ sprite, url: assetUrl(sprite) }));
  const hasVisuals = Boolean(backgroundUrl || depthUrl || spriteUrls.some((sprite) => sprite.url));
  const dialogue = useMemo(() => latestDialogue(events), [events]);
  const metadata = sceneMetadata(sceneState);
  const warnings = sceneState.warnings ?? [];

  return (
    <section className="grid gap-4">
      <div className="relative aspect-[16/9] min-h-72 overflow-hidden rounded-md border border-border bg-bg">
        {backgroundUrl ? (
          <img
            alt="Scene background"
            className="absolute inset-0 h-full w-full object-cover"
            src={backgroundUrl}
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center px-4 text-center text-sm text-text-muted">
            <p className="font-medium text-text">No scene visuals available</p>
            <p className="mt-1 max-w-md">
              The backend did not provide a background or active sprite for this scene.
            </p>
          </div>
        )}

        {depthUrl && (
          <img
            alt="Scene depth layer"
            className="absolute inset-0 h-full w-full object-cover opacity-20 mix-blend-multiply"
            src={depthUrl}
          />
        )}

        <div className="absolute inset-x-0 bottom-0 flex items-end justify-center gap-4 px-6 pt-10">
          {spriteUrls.map(({ sprite, url }, index) => {
            if (!url) return null;
            const key =
              typeof sprite.item_id === 'number'
                ? sprite.item_id
                : `${url}-${index}`;
            const label = assetLabel(sprite);
            return (
              <img
                key={key}
                alt={label ? `Character sprite ${index + 1}: ${label}` : `Character sprite ${index + 1}`}
                className="max-h-[72%] max-w-[36%] object-contain drop-shadow"
                src={url}
              />
            );
          })}
        </div>

        {sceneState.location_key && (
          <div className="absolute left-3 top-3">
            <Badge variant="neutral">{sceneState.location_key}</Badge>
          </div>
        )}
      </div>

      {metadata.length > 0 && (
        <dl className="flex flex-wrap gap-2 text-xs text-text-muted">
          {metadata.map(([label, value]) => (
            <div key={label} className="rounded-md border border-border bg-bg px-2 py-1">
              <dt className="sr-only">{label}</dt>
              <dd>
                {label}: {value}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {!hasVisuals && warnings.length === 0 && (
        <p className="text-xs text-text-muted">
          Visuals will appear here when the API returns approved scene assets.
        </p>
      )}

      {showDialogue && (
        <div className="rounded-md border border-border bg-bg p-4">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-normal text-text-muted">Dialogue</h3>
          {dialogue.length > 0 ? (
            <div className="grid gap-2">
              {dialogue.map((line, index) => (
                <p key={`${line.speaker}-${index}`} className="text-sm">
                  <span className="font-medium">{line.speaker}: </span>
                  {line.text}
                </p>
              ))}
            </div>
          ) : (
            <p className="text-sm text-text-muted">No dialogue events.</p>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <div className="rounded-md border border-warn/30 bg-warn/10 p-3 text-sm text-warn">
          <div className="font-medium">Warnings</div>
          <ul className="mt-1 grid gap-1">
            {warnings.map((warning, index) => (
              <li key={index}>{warningText(warning)}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
