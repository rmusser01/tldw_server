import type { IdempotentCommand } from '@/lib/idempotent-command';
import {
  WebhookApiError,
  WebhookContractError,
  WebhookTransportError,
} from '@/lib/http';
import type { WebhookRegistration, WebhookSecretResponse, WebhookStatus } from '@/types';

export const WEBHOOK_PAGE_SIZE = 20;

export type WebhookMode = 'canonical' | 'legacy' | null;
export type SecretOperation = 'create' | 'rotate';

export type SecretCommandResult = {
  data: WebhookSecretResponse;
  etag: string;
  status: number;
  requestId: string | null;
};

export type PendingSecretCommand = {
  command: IdempotentCommand<SecretCommandResult>;
  operation: SecretOperation;
  webhookId: number | null;
};

export type SecretState = {
  value: string;
  replayed: boolean;
  operation: SecretOperation;
};

export type ConflictState = {
  status: 412 | 428;
  action: string;
  registration: WebhookRegistration;
};

export type SafeError = {
  message: string;
  requestId: string | null;
};

export const canLoadCanonicalData = (status: WebhookStatus): boolean => (
  status.mode !== 'off'
  && status.schema_ready
  && status.migration.phase === 'complete'
);

export const safeWebhookError = (error: unknown, fallback: string): SafeError => {
  if (
    error instanceof WebhookApiError
    || error instanceof WebhookContractError
    || error instanceof WebhookTransportError
  ) {
    return { message: error.message, requestId: error.requestId };
  }
  return { message: fallback, requestId: null };
};

export const isConditionalWebhookError = (
  error: unknown,
): error is WebhookApiError & { status: 412 | 428 } => (
  error instanceof WebhookApiError && (error.status === 412 || error.status === 428)
);

export const activationBlockReason = (
  registration: WebhookRegistration,
  status: WebhookStatus,
): string | null => {
  if (registration.active) return null;
  if (!status.delivery_capability_ready) return 'Delivery capability is unavailable';
  if (registration.secret_rotation_required) return 'Generate a new signing secret before activation';
  if (status.key_state !== 'available') return 'Webhook signing key is unavailable';
  if (status.limits.active_registrations_over_limit) return 'Active registration limit is exceeded';
  return null;
};

export const registrationReviewSummary = (registration: WebhookRegistration): string => (
  `Webhook ${registration.id}, revision ${registration.revision}: `
  + `${registration.target_display}; ${registration.active ? 'active' : 'inactive'}; `
  + `${registration.timeout_seconds}s timeout; `
  + `description "${registration.description || 'None'}"; `
  + `events ${registration.event_types.join(', ')}.`
);
