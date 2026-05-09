export interface VNPlayErrorInfo {
  code?: string;
  message: string;
  status?: number;
}

const RECOVERABLE_TURN_CODES = new Set(['stale_scene_version', 'turn_in_progress']);

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function readNumber(value: unknown): number | undefined {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : undefined;
}

function readString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function readCode(record: Record<string, unknown> | null): string | undefined {
  if (!record) return undefined;
  return (
    readString(record.error_code) ??
    readString(record.code) ??
    readCode(asRecord(record.detail)) ??
    readCode(asRecord(record.details))
  );
}

function readMessage(record: Record<string, unknown> | null): string | undefined {
  if (!record) return undefined;
  const detail = record.detail;
  const details = record.details;
  return (
    readString(detail) ??
    readString(record.message) ??
    readString(record.error) ??
    readMessage(asRecord(detail)) ??
    readMessage(asRecord(details)) ??
    readCode(record)
  );
}

export function createVNPlayIdempotencyKey(prefix: string): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${prefix}-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

export function getVNPlayErrorInfo(error: unknown): VNPlayErrorInfo {
  const record = asRecord(error);
  const status =
    readNumber(record?.status) ??
    readNumber(record?.statusCode) ??
    readNumber(asRecord(record?.response)?.status);
  const code = readCode(record);
  const message =
    readMessage(record) ??
    (error instanceof Error && error.message.trim() ? error.message : undefined) ??
    'Failed to submit turn';

  return {
    ...(code ? { code } : {}),
    message,
    ...(status !== undefined ? { status } : {}),
  };
}

export function isRecoverableVNPlayConflict(error: unknown): boolean {
  const info = getVNPlayErrorInfo(error);
  if (info.code && RECOVERABLE_TURN_CODES.has(info.code)) return true;
  if (info.status === 409) return true;
  return /stale_scene_version|turn_in_progress/i.test(info.message);
}
