import { useEffect, useState } from 'react';
import { z } from 'zod';
import { ApiError } from '@/services/tldw/api-error';
import type { VNCommandScope } from '@web/lib/vnGenerationRecovery';
import type { ApiRequestConfig } from '@web/types/common';
import type { useVNGenerationRecovery } from './useVNGenerationRecovery';

export type VNPortabilityAuthority = Pick<
  ReturnType<typeof useVNGenerationRecovery>,
  'authority' | 'ready' | 'captureCurrent' | 'isCurrent'
>;

export function vnPortabilityRequestConfig(scope: VNCommandScope): ApiRequestConfig {
  return { baseURL: scope.server, headers: { 'X-TLDW-Expected-User-ID': scope.principal } };
}

const savedId = z.number().int().positive();
const jobId = z.string().regex(/^[a-zA-Z0-9_-]{1,80}$/);
const idempotencyKey = z.string().regex(/^[a-zA-Z0-9._~-]{1,160}$/);
const receiptsSchema = z.strictObject({
  export: z.strictObject({
    packId: savedId,
    request: z.strictObject({
      include_character_payload: z.boolean(),
      include_world_book_payloads: z.boolean(),
      include_full_provenance: z.boolean(),
      strict: z.boolean(),
      warn_for_sharing: z.literal(true),
      idempotency_key: idempotencyKey,
    }),
    jobId: jobId.optional(),
  }).optional(),
  preview: z.strictObject({ idempotencyKey, previewId: savedId.optional() }).optional(),
  import: z.strictObject({
    request: z.strictObject({
      preview_id: savedId,
      trust_mode: z.enum(['trusted_restore', 'untrusted_import']),
      target_mode: z.enum(['create_new', 'update_existing']),
      character_action: z.enum(['link_existing_character', 'import_included_character']),
      target_character_id: savedId.nullable(),
      target_pack_id: savedId.nullable(),
      conflict_decisions: z.strictObject({ confirm_all_risky_diffs: z.boolean() }),
      idempotency_key: idempotencyKey,
    }),
    jobId: jobId.optional(),
  }).optional(),
});

type Receipts = z.infer<typeof receiptsSchema>;
type ReceiptSnapshot = { key: string; receipts: Receipts | null };

function storageKey(scope: VNCommandScope): string {
  return `tldw:vn-portability:v1:${encodeURIComponent(scope.server)}:${scope.principal}`;
}

function readReceipts(key: string): Receipts {
  const raw = window.sessionStorage.getItem(key);
  if (raw === null) return {};
  return receiptsSchema.parse(JSON.parse(raw));
}

export function useVNPortabilityReceipts(authority: VNCommandScope | null) {
  const key = authority ? storageKey(authority) : null;
  const [snapshot, setSnapshot] = useState<ReceiptSnapshot | null>(null);

  useEffect(() => {
    if (!key) return;
    try {
      setSnapshot({ key, receipts: readReceipts(key) });
    } catch {
      setSnapshot({ key, receipts: null });
    }
  }, [key]);

  function save(scope: VNCommandScope, patch: Partial<Receipts>): void {
    const currentKey = storageKey(scope);
    const next = receiptsSchema.parse({ ...readReceipts(currentKey), ...patch });
    window.sessionStorage.setItem(currentKey, JSON.stringify(next));
    setSnapshot({ key: currentKey, receipts: next });
  }

  const current = snapshot?.key === key ? snapshot : null;
  return { receipts: current?.receipts ?? null, storageUnavailable: current?.receipts === null, save };
}

const terminalStatuses = new Set(['completed', 'failed', 'cancelled', 'quarantined', 'deleting', 'deleted', 'expired']);

export const isPortabilityTerminal = (status: string): boolean => terminalStatuses.has(status);

type StatusSnapshot<Job> = { key: string; job?: Job; paused?: boolean; unavailable?: boolean };

export function useVNPortabilityStatus<Id extends string | number, Job extends { status: string }>(
  recovery: VNPortabilityAuthority,
  id: Id | undefined,
  read: (id: Id, config: ApiRequestConfig) => Promise<Job>
) {
  const { authority, ready, captureCurrent, isCurrent } = recovery;
  const key = authority && id !== undefined ? JSON.stringify([authority.server, authority.principal, id]) : null;
  const [snapshot, setSnapshot] = useState<StatusSnapshot<Job> | null>(null);
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (!key || !ready || id === undefined) return;
    const capture = captureCurrent();
    if (!capture || JSON.stringify([capture.scope.server, capture.scope.principal, id]) !== key) return;
    const controller = new AbortController();
    const config = { ...vnPortabilityRequestConfig(capture.scope), signal: controller.signal };
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;

    const poll = async (jobId: Id) => {
      if (controller.signal.aborted || !isCurrent(capture)) return;
      try {
        const job = await read(jobId, config);
        if (controller.signal.aborted || !isCurrent(capture)) return;
        const terminal = isPortabilityTerminal(job.status);
        const paused = !terminal && ++attempts >= 60;
        setSnapshot({ key, job, paused });
        if (!terminal && !paused) timer = setTimeout(() => void poll(jobId), 1000);
      } catch (error) {
        if (controller.signal.aborted || !isCurrent(capture)) return;
        const unavailable = error instanceof ApiError && (error.status === 404 || error.status === 410);
        setSnapshot({ key, paused: true, unavailable });
      }
    };

    void poll(id);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [captureCurrent, id, isCurrent, key, read, ready, refresh]);

  const current = ready && key && snapshot?.key === key ? snapshot : null;
  return {
    job: current?.job,
    paused: current?.paused,
    unavailable: current?.unavailable,
    check: () => {
      setSnapshot(null);
      setRefresh((revision) => revision + 1);
    },
  };
}
