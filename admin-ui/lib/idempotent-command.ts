import { ApiError, WebhookContractError } from './http';

type CommandState = 'ready' | 'running' | 'retryable' | 'completed' | 'failed';

export type IdempotentCommandRequest<TBody> = Readonly<{
  operation: string;
  body: Readonly<TBody>;
  idempotencyKey: string;
}>;

export type IdempotentCommand<TResult> = {
  readonly idempotencyKey: string;
  readonly canRetry: boolean;
  run: () => Promise<TResult>;
  retry: () => Promise<TResult>;
};

export class IdempotentCommandStateError extends Error {
  constructor(message = 'Idempotent command is not retryable') {
    super(message);
    this.name = 'IdempotentCommandStateError';
  }
}

const deepFreeze = <T>(value: T): T => {
  if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const nested of Object.values(value)) {
      deepFreeze(nested);
    }
  }
  return value;
};

const normalizeBody = <TBody>(body: TBody): Readonly<TBody> => {
  const encoded = JSON.stringify(body);
  if (encoded === undefined) {
    throw new TypeError('Idempotent command body must be JSON serializable');
  }
  return deepFreeze(JSON.parse(encoded) as TBody);
};

const isTransportFailure = (error: unknown): boolean => {
  if (error instanceof TypeError) return true;
  if (error instanceof Error
    && (
      error.name === 'AbortError'
      || error.name === 'NetworkError'
      || error.name === 'WebhookTransportError'
    )) return true;
  return error instanceof ApiError
    && (
      error.status >= 500
      || (
        error instanceof WebhookContractError
        && error.status >= 200
        && error.status < 300
      )
    );
};

export const generateIdempotencyKey = (): string => {
  if (!globalThis.crypto?.getRandomValues) {
    throw new Error('Secure random generation is unavailable');
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, '0')).join('');
};

export const createIdempotentCommand = <TBody, TResult>(
  operation: string,
  body: TBody,
  request: (command: IdempotentCommandRequest<TBody>) => Promise<TResult>,
): IdempotentCommand<TResult> => {
  if (!/^[a-z][a-z0-9_-]{0,63}$/.test(operation)) {
    throw new TypeError('Idempotent command operation is invalid');
  }

  const idempotencyKey = generateIdempotencyKey();
  const normalizedRequest = Object.freeze({
    operation,
    body: normalizeBody(body),
    idempotencyKey,
  });
  let state: CommandState = 'ready';

  const execute = async (): Promise<TResult> => {
    state = 'running';
    try {
      const result = await request(normalizedRequest);
      state = 'completed';
      return result;
    } catch (error) {
      state = isTransportFailure(error) ? 'retryable' : 'failed';
      throw error;
    }
  };

  return {
    idempotencyKey,
    get canRetry() {
      return state === 'retryable';
    },
    run: () => {
      if (state !== 'ready') {
        return Promise.reject(new IdempotentCommandStateError());
      }
      return execute();
    },
    retry: () => {
      if (state !== 'retryable') {
        return Promise.reject(new IdempotentCommandStateError());
      }
      return execute();
    },
  };
};
