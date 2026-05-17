export interface Voice {
  voice_id: string;
  name: string;
}

export interface Model {
  model_id: string;
  name: string;
}

const BASE_URL = 'https://api.elevenlabs.io/v1';
const DEFAULT_ELEVENLABS_TIMEOUT_MS = 10_000;

type ElevenLabsRequestOptions = {
  timeoutMs?: number;
  signal?: AbortSignal;
  responseType?: 'json' | 'text' | 'arrayBuffer' | 'arraybuffer';
  includeBrowserTransportFailure?: boolean;
};

function createTimeoutSignal(timeoutMs: number): {
  signal: AbortSignal;
  cleanup: () => void;
  didTimeout: () => boolean;
} {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  return {
    signal: controller.signal,
    cleanup: () => clearTimeout(timeoutId),
    didTimeout: () => timedOut,
  };
}

function createRequestSignal(options?: ElevenLabsRequestOptions): {
  signal: AbortSignal;
  cleanup: () => void;
  didTimeout: () => boolean;
  didCallerAbort: () => boolean;
} {
  const timeout = createTimeoutSignal(
    options?.timeoutMs ?? DEFAULT_ELEVENLABS_TIMEOUT_MS
  );
  const callerSignal = options?.signal;
  if (!callerSignal) {
    return {
      ...timeout,
      didCallerAbort: () => false,
    };
  }

  const controller = new AbortController();
  let callerAborted = callerSignal.aborted;

  const abortFromTimeout = () => {
    controller.abort(timeout.signal.reason);
  };
  const abortFromCaller = () => {
    if (timeout.didTimeout()) return;
    callerAborted = true;
    timeout.cleanup();
    controller.abort(callerSignal.reason);
  };

  timeout.signal.addEventListener('abort', abortFromTimeout, { once: true });
  if (callerSignal.aborted) {
    abortFromCaller();
  } else {
    callerSignal.addEventListener('abort', abortFromCaller, { once: true });
  }

  return {
    signal: controller.signal,
    cleanup: () => {
      timeout.cleanup();
      timeout.signal.removeEventListener('abort', abortFromTimeout);
      callerSignal.removeEventListener('abort', abortFromCaller);
    },
    didTimeout: timeout.didTimeout,
    didCallerAbort: () => callerAborted,
  };
}

function isTimeoutLikeFetchFailure(
  error: unknown,
  didTimeout: boolean,
  includeBrowserTransportFailure = false
): boolean {
  if (error instanceof DOMException) {
    return (
      (error.name === 'AbortError' && didTimeout) ||
      error.name === 'TimeoutError'
    );
  }

  if (error instanceof Error) {
    const message = `${error.name} ${error.message}`.toLowerCase();
    return (
      message.includes('timeout') ||
      message.includes('timed out') ||
      message.includes('err_timed_out') ||
      (includeBrowserTransportFailure && message.includes('failed to fetch'))
    );
  }

  return false;
}

async function fetchElevenLabs<T>(
  path: string,
  apiKey: string,
  init: RequestInit = {},
  options?: ElevenLabsRequestOptions
): Promise<T> {
  const requestSignal = createRequestSignal(options);
  const headers = new Headers(init.headers);
  headers.set('xi-api-key', apiKey);

  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers,
      signal: requestSignal.signal,
    });

    if (!response.ok) {
      throw new Error(`ElevenLabs request failed with status ${response.status}`);
    }

    switch (options?.responseType ?? 'json') {
      case 'arrayBuffer':
      case 'arraybuffer':
        return (await response.arrayBuffer()) as T;
      case 'text':
        return (await response.text()) as T;
      case 'json':
      default:
        return (await response.json()) as T;
    }
  } catch (error) {
    const callerAbort = requestSignal.didCallerAbort();
    if (
      !callerAbort &&
      isTimeoutLikeFetchFailure(
        error,
        requestSignal.didTimeout(),
        options?.includeBrowserTransportFailure ?? true
      )
    ) {
      throw new Error('ElevenLabs request timed out');
    }
    throw error;
  } finally {
    requestSignal.cleanup();
  }
}

export const getVoices = async (
  apiKey: string,
  options?: ElevenLabsRequestOptions
): Promise<Voice[]> => {
  const response = await fetchElevenLabs<{ voices: Voice[] }>(
    '/voices',
    apiKey,
    { method: 'GET' },
    options
  );
  return response.voices;
};

export const getModels = async (
  apiKey: string,
  options?: ElevenLabsRequestOptions
): Promise<Model[]> => {
  return fetchElevenLabs<Model[]>(
    '/models',
    apiKey,
    { method: 'GET' },
    options
  );
};

export const generateSpeech = async (
  apiKey: string,
  text: string,
  voiceId: string,
  modelId: string,
  speed?: number,
  options?: ElevenLabsRequestOptions
): Promise<ArrayBuffer> => {
  const payload: Record<string, unknown> = {
    text,
    model_id: modelId
  }

  if (speed != null) {
    payload.voice_settings = { speed }
  }

  return fetchElevenLabs<ArrayBuffer>(
    `/text-to-speech/${voiceId}`,
    apiKey,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    },
    {
      ...options,
      responseType: 'arrayBuffer',
      includeBrowserTransportFailure: false,
    }
  );
};
