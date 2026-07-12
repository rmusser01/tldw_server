import { captureSessionIdFromHeaders } from '@web/lib/session';

export type SSEMessageHandler = (delta: string) => void;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type SSEJSONHandler = (json: any) => void;
export interface StructuredSSEEvent {
  event: string;
  id?: number;
  payload?: unknown;
}
export type StructuredSSEHandler = (event: StructuredSSEEvent) => void;

export interface SSEOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  signal?: AbortSignal;
  credentials?: RequestCredentials;
}

const readRetryAfterSeconds = (headers: Headers): number | undefined => {
  const raw = headers.get('retry-after');
  if (!raw) return undefined;
  const seconds = Number(raw);
  if (Number.isFinite(seconds) && seconds > 0) return seconds;
  const retryAt = Date.parse(raw);
  if (!Number.isFinite(retryAt)) return undefined;
  const remainingMs = retryAt - Date.now();
  return remainingMs > 0 ? Math.ceil(remainingMs / 1_000) : undefined;
};

async function openSSEStream(
  url: string,
  options: SSEOptions
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const res = await fetch(url, {
    method: options.method || 'GET',
    headers: options.headers,
    body: options.body,
    credentials: options.credentials ?? 'include',
    signal: options.signal,
  });
  captureSessionIdFromHeaders(res.headers);

  if (!res.ok || !res.body) {
    const { ApiError } = await import('@web/lib/api');
    throw new ApiError(`Stream error: ${res.status} ${res.statusText}`, {
      status: res.status,
      retryAfter: readRetryAfterSeconds(res.headers),
    });
  }

  return res.body.getReader();
}

export async function streamStructuredSSE(
  url: string,
  options: SSEOptions,
  onEvent: StructuredSSEHandler,
  onDone?: () => void,
  onOpen?: () => void
): Promise<void> {
  const reader = await openSSEStream(url, options);
  onOpen?.();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let doneSent = false;
  let eventType = 'message';
  let eventId: number | undefined;
  let dataLines: string[] = [];

  const flushFrame = () => {
    if (!eventType && dataLines.length === 0 && eventId === undefined) {
      return;
    }
    const rawPayload = dataLines.join('\n');
    if (rawPayload === '[DONE]') {
      if (onDone && !doneSent) {
        onDone();
        doneSent = true;
      }
    } else {
      let payload: unknown = undefined;
      if (dataLines.length > 0) {
        try {
          payload = JSON.parse(rawPayload);
        } catch {
          payload = rawPayload;
        }
      }
      onEvent({
        event: eventType || 'message',
        id: eventId,
        payload,
      });
    }

    eventType = 'message';
    eventId = undefined;
    dataLines = [];
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const rawLine of lines) {
        const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine;
        if (line === '') {
          flushFrame();
          continue;
        }
        if (line.startsWith(':')) {
          continue;
        }
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim() || 'message';
          continue;
        }
        if (line.startsWith('id:')) {
          const parsed = Number.parseInt(line.slice(3).trim(), 10);
          eventId = Number.isFinite(parsed) ? parsed : undefined;
          continue;
        }
        if (line.startsWith('data:')) {
          dataLines.push(line.slice(5).trim());
        }
      }
    }
  } finally {
    if (buffer.trim().length > 0) {
      dataLines.push(buffer.trim());
    }
    if (dataLines.length > 0 || eventId !== undefined || eventType !== 'message') {
      flushFrame();
    }
    if (onDone && !doneSent) {
      onDone();
      doneSent = true;
    }
  }
}

// Minimal chat-compatible SSE reader using fetch + ReadableStream decoding
export async function streamSSE(
  url: string,
  options: SSEOptions,
  onDelta: SSEMessageHandler,
  onJSON?: SSEJSONHandler,
  onDone?: () => void
): Promise<void> {
  await streamStructuredSSE(
    url,
    options,
    (event) => {
      const payload = event.payload;
      if (payload === undefined || payload === null) {
        return;
      }

      if (typeof payload === 'string') {
        if (payload.length > 0) {
          onDelta(payload);
        }
        return;
      }

      if (typeof payload === 'object') {
        const json = payload as {
          error?: { message?: string };
          choices?: Array<{
            delta?: { content?: string };
            message?: { content?: string };
          }>;
        };
        if (json.error?.message) {
          throw new Error(json.error.message);
        }
        if (onJSON) {
          onJSON(json);
        }
        const firstChoice = json.choices?.[0];
        const delta = firstChoice?.delta?.content;
        if (typeof delta === 'string' && delta.length > 0) {
          onDelta(delta);
          return;
        }
        const message = firstChoice?.message?.content;
        if (typeof message === 'string' && message.length > 0) {
          onDelta(message);
        }
      }
    },
    onDone
  );
}
