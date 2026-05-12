import { useEffect, useMemo, useRef, useState } from 'react';
import { Eye, RefreshCcw, RotateCcw, ShieldAlert, XCircle } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import ConfirmDialog from '@web/components/ui/ConfirmDialog';
import { getVNPlayGenerationRevisionDebug } from '@web/lib/api/vnPlay';
import type {
  VNPlayGenerationHistoryItem,
  VNPlayGenerationRevisionDebugResponse,
  VNPlaySceneState,
} from '@web/types/vn-play';

const MODERATION_REVEAL_CONFIRM = 'REVEAL_MODERATION_BLOCKED';

type GenerationInspectorProps = {
  generations: VNPlayGenerationHistoryItem[];
  canViewDebug?: boolean;
  hasMore?: boolean;
  isLoading?: boolean;
  sceneState: VNPlaySceneState;
  sessionId: number;
  onActivateRevision?: (item: VNPlayGenerationHistoryItem) => void | Promise<void>;
  onCancelRequest?: (generationRequestId: number) => void | Promise<void>;
  onConfirmRequest?: (generationRequestId: number) => void | Promise<void>;
  onLoadMore?: () => void | Promise<void>;
  onRegenerate?: (item: VNPlayGenerationHistoryItem) => void | Promise<void>;
};

function badgeVariant(status: string): 'danger' | 'info' | 'neutral' | 'success' | 'warning' {
  if (status === 'succeeded' || status === 'completed') return 'success';
  if (status.includes('blocked') || status.includes('failed')) return 'danger';
  if (status.includes('pending') || status.includes('progress')) return 'warning';
  return 'neutral';
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function readNumber(value: unknown): number | null {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : null;
}

function waitingGenerationRequestId(sceneState: VNPlaySceneState): number | null {
  const direct = readNumber(sceneState.waiting_generation_request_id);
  if (direct !== null) return direct;
  const confirmation = asRecord(sceneState.waiting_generation_confirmation);
  return readNumber(confirmation?.generation_request_id ?? confirmation?.request_id);
}

function publicOutputText(item: VNPlayGenerationHistoryItem): string[] {
  const output = item.public_output ?? {};
  const lines: string[] = [];
  const leadIn = output.lead_in;
  if (typeof leadIn === 'string' && leadIn.trim()) lines.push(leadIn);
  for (const entry of Array.isArray(output.narrative) ? output.narrative : []) {
    const record = asRecord(entry);
    if (typeof record?.text === 'string' && record.text.trim()) lines.push(record.text);
  }
  for (const entry of Array.isArray(output.dialogue) ? output.dialogue : []) {
    const record = asRecord(entry);
    const text = typeof record?.text === 'string' ? record.text : '';
    const speaker = typeof record?.speaker === 'string' ? record.speaker : '';
    if (text.trim()) lines.push(speaker ? `${speaker}: ${text}` : text);
  }
  for (const entry of Array.isArray(output.choices) ? output.choices : []) {
    const record = asRecord(entry);
    if (typeof record?.text === 'string' && record.text.trim()) lines.push(record.text);
  }
  return lines;
}

function compactJson(value: unknown): string {
  if (value === null || value === undefined) return '';
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatTimestamp(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function statusDescription(item: VNPlayGenerationHistoryItem): string | null {
  const code = item.public_error_code ?? item.status;
  if (/stale|conflict/i.test(code)) {
    return 'Scene changed before this generation could apply. Reload the session state before retrying.';
  }
  if (/activation.*blocked|blocked.*activation/i.test(code)) {
    return 'Activation was blocked by backend validation.';
  }
  if (/provider.*unavailable|unavailable/i.test(code)) {
    return 'The configured generation provider was unavailable.';
  }
  if (/parser.*failed|parse.*failed/i.test(code)) {
    return 'The model response could not be parsed into the expected output schema.';
  }
  if (/abandoned|timeout/i.test(code)) {
    return 'The generation request was abandoned or timed out before completion.';
  }
  if (item.public_error_code) {
    return item.public_error_code;
  }
  return null;
}

export default function GenerationInspector({
  canViewDebug = true,
  generations,
  hasMore = false,
  isLoading = false,
  onLoadMore,
  onActivateRevision,
  onCancelRequest,
  onConfirmRequest,
  onRegenerate,
  sceneState,
  sessionId,
}: GenerationInspectorProps) {
  const [debugItem, setDebugItem] = useState<VNPlayGenerationRevisionDebugResponse | null>(null);
  const [debugError, setDebugError] = useState<string | null>(null);
  const [debugLoadingKey, setDebugLoadingKey] = useState<string | null>(null);
  const [revealTarget, setRevealTarget] = useState<VNPlayGenerationRevisionDebugResponse | null>(null);
  const [actionLoadingKey, setActionLoadingKey] = useState<string | null>(null);
  const actionLoadingKeyRef = useRef<string | null>(null);
  const pendingRequestId = useMemo(() => waitingGenerationRequestId(sceneState), [sceneState]);

  useEffect(() => {
    setDebugItem(null);
    setDebugError(null);
    setDebugLoadingKey(null);
    setRevealTarget(null);
    setActionLoadingKey(null);
    actionLoadingKeyRef.current = null;
  }, [sessionId]);

  const runAction = async (key: string, action: () => void | Promise<void>) => {
    if (actionLoadingKeyRef.current) return;
    actionLoadingKeyRef.current = key;
    setActionLoadingKey(key);
    try {
      await action();
    } finally {
      actionLoadingKeyRef.current = null;
      setActionLoadingKey(null);
    }
  };

  const loadDebug = async (
    item: VNPlayGenerationHistoryItem,
    options?: { includeBlockedRaw?: boolean }
  ) => {
    const key = `${item.generation_id}:${item.id}`;
    setDebugLoadingKey(key);
    setDebugError(null);
    try {
      const debug = options?.includeBlockedRaw
        ? await getVNPlayGenerationRevisionDebug(sessionId, item.generation_id, item.id, {
            include_blocked_raw: true,
            confirm: MODERATION_REVEAL_CONFIRM,
          })
        : await getVNPlayGenerationRevisionDebug(sessionId, item.generation_id, item.id);
      setDebugItem(debug);
      setRevealTarget(null);
    } catch (error) {
      setDebugError(error instanceof Error ? error.message : 'Failed to load generation debug detail');
    } finally {
      setDebugLoadingKey(null);
    }
  };

  return (
    <section className="mt-5 border-t border-border pt-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold uppercase tracking-normal text-text-muted">
          Scripted generations
        </h3>
        <Badge variant={generations.length > 0 ? 'info' : 'neutral'}>{generations.length}</Badge>
      </div>

      {pendingRequestId !== null && (
        <div className="mb-3 rounded-md border border-warn/30 bg-warn/10 p-3 text-sm">
          <p className="font-medium text-warn">Generation confirmation pending</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {onConfirmRequest && (
              <Button
                loading={actionLoadingKey === `confirm:${pendingRequestId}`}
                size="xs"
                type="button"
                onClick={() => void runAction(`confirm:${pendingRequestId}`, () => onConfirmRequest(pendingRequestId))}
              >
                Confirm generation
              </Button>
            )}
            {onCancelRequest && (
              <Button
                className="gap-1"
                size="xs"
                type="button"
                variant="secondary"
                loading={actionLoadingKey === `cancel:${pendingRequestId}`}
                onClick={() => void runAction(`cancel:${pendingRequestId}`, () => onCancelRequest(pendingRequestId))}
              >
                <XCircle aria-hidden className="h-3.5 w-3.5" />
                Cancel
              </Button>
            )}
          </div>
        </div>
      )}

      {isLoading && <p className="text-sm text-text-muted">Loading generation history...</p>}
      {!isLoading && generations.length === 0 && (
        <p className="text-sm text-text-muted">No scripted generations for this session.</p>
      )}
      {generations.length > 0 && (
        <ul className="grid gap-3">
          {generations.map((item) => {
            const outputLines = publicOutputText(item);
            const debugKey = `${item.generation_id}:${item.id}`;
            const createdAt = formatTimestamp(item.created_at);
            const errorDescription = statusDescription(item);
            return (
              <li key={`${item.generation_id}:${item.id}`} className="rounded-md border border-border bg-bg p-3 text-sm">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="font-medium">{item.generation_point_key}</p>
                    <p className="text-xs text-text-muted">
                      Revision {item.revision_number} | {item.output_schema} | {item.profile.profile_key}
                    </p>
                  </div>
                  <div className="flex flex-wrap justify-end gap-1">
                    {item.active && <Badge variant="success">Active</Badge>}
                    <Badge variant={badgeVariant(item.status)}>{item.status}</Badge>
                  </div>
                </div>
                {outputLines.length > 0 ? (
                  <div className="mt-2 grid gap-1">
                    {outputLines.map((line, index) => (
                      <p key={`${item.id}:output:${index}`} className="text-sm text-text">
                        {line}
                      </p>
                    ))}
                  </div>
                ) : item.public_error_code ? (
                  <p className="mt-2 text-sm text-danger">{item.public_error_code}</p>
                ) : (
                  <p className="mt-2 text-sm text-text-muted">No public output.</p>
                )}
                <p className="mt-2 text-xs text-text-muted">
                  Snapshot {item.profile.snapshot_id}
                  {item.profile.provider_class ? ` | ${item.profile.provider_class}` : ''}
                  {item.profile.estimated_cost_class ? ` | ${item.profile.estimated_cost_class}` : ''}
                  {createdAt ? ` | ${createdAt}` : ''}
                </p>
                {errorDescription && (
                  <p className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-2 py-1 text-xs text-warn">
                    {errorDescription}
                  </p>
                )}
                <div className="mt-3 flex flex-wrap gap-2">
                  {onRegenerate && (
                    <Button
                      aria-label={`Regenerate ${item.generation_point_key}`}
                      className="gap-1"
                      size="xs"
                      type="button"
                      variant="secondary"
                      loading={actionLoadingKey === `regenerate:${item.generation_id}`}
                      onClick={() =>
                        void runAction(`regenerate:${item.generation_id}`, () => onRegenerate(item))
                      }
                    >
                      <RefreshCcw aria-hidden className="h-3.5 w-3.5" />
                      Regenerate
                    </Button>
                  )}
                  {!item.active && item.status === 'succeeded' && onActivateRevision && (
                    <Button
                      aria-label={`Activate revision ${item.revision_number} for ${item.generation_point_key}`}
                      className="gap-1"
                      size="xs"
                      type="button"
                      variant="secondary"
                      loading={actionLoadingKey === `activate:${debugKey}`}
                      onClick={() => void runAction(`activate:${debugKey}`, () => onActivateRevision(item))}
                    >
                      <RotateCcw aria-hidden className="h-3.5 w-3.5" />
                      Activate
                    </Button>
                  )}
                  {canViewDebug ? (
                    <Button
                      aria-label={`Debug ${item.generation_point_key} revision ${item.revision_number}`}
                      className="gap-1"
                      loading={debugLoadingKey === debugKey}
                      size="xs"
                      type="button"
                      variant="ghost"
                      onClick={() => void loadDebug(item)}
                    >
                      <Eye aria-hidden className="h-3.5 w-3.5" />
                      Debug
                    </Button>
                  ) : (
                    <Badge variant="neutral">Debug restricted</Badge>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
      {hasMore && onLoadMore && (
        <Button
          className="mt-3"
          loading={isLoading}
          size="xs"
          type="button"
          variant="secondary"
          onClick={() => void onLoadMore()}
        >
          Load more generations
        </Button>
      )}

      {debugError && <p className="mt-3 text-sm text-danger">{debugError}</p>}
      {debugItem && (
        <div className="mt-3 rounded-md border border-border bg-bg p-3 text-sm">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="font-medium">Revision debug detail</p>
              <p className="text-xs text-text-muted">
                {debugItem.generation_point_key} | revision {debugItem.revision_number}
              </p>
            </div>
            <Badge variant={badgeVariant(debugItem.status)}>{debugItem.status}</Badge>
          </div>
          <p className="mt-2 text-xs text-text-muted">
            Raw output: {debugItem.raw_output_debug_state}
          </p>
          {debugItem.raw_output_debug_state === 'redacted' && (
            <Button
              className="mt-2 gap-1"
              size="xs"
              type="button"
              variant="danger"
              onClick={() => setRevealTarget(debugItem)}
            >
              <ShieldAlert aria-hidden className="h-3.5 w-3.5" />
              Reveal moderation-blocked raw output
            </Button>
          )}
          {debugItem.raw_output_debug && (
            <pre className="mt-2 max-h-52 overflow-auto rounded-md bg-surface p-2 text-xs">
              {compactJson(debugItem.raw_output_debug)}
            </pre>
          )}
          {Object.keys(debugItem.parser_diagnostics ?? {}).length > 0 && (
            <pre className="mt-2 max-h-40 overflow-auto rounded-md bg-surface p-2 text-xs">
              {compactJson(debugItem.parser_diagnostics)}
            </pre>
          )}
          {Object.keys(debugItem.moderation_diagnostics ?? {}).length > 0 && (
            <pre className="mt-2 max-h-40 overflow-auto rounded-md bg-surface p-2 text-xs">
              {compactJson(debugItem.moderation_diagnostics)}
            </pre>
          )}
        </div>
      )}

      <ConfirmDialog
        open={Boolean(revealTarget)}
        title="Reveal moderation-blocked output"
        message="This shows raw model output that the backend moderation path blocked. Use it only for debugging and audit review."
        confirmText="Reveal raw output"
        cancelText="Keep redacted"
        destructive
        onCancel={() => setRevealTarget(null)}
        onConfirm={() => {
          if (!revealTarget) return;
          void loadDebug(
            {
              id: revealTarget.id,
              generation_id: revealTarget.generation_id,
              generation_point_key: revealTarget.generation_point_key,
              revision_number: revealTarget.revision_number,
              status: revealTarget.status,
              active: false,
              output_schema: revealTarget.output_schema,
              public_output: revealTarget.public_output,
              profile: revealTarget.profile,
            },
            { includeBlockedRaw: true }
          );
        }}
      />
    </section>
  );
}
