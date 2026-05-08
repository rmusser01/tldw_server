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
  const timeout = createTimeoutSignal(
    options?.timeoutMs ?? DEFAULT_ELEVENLABS_TIMEOUT_MS
  );
  const headers = new Headers(init.headers);
  headers.set('xi-api-key', apiKey);

  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers,
      signal: timeout.signal,
    });
  } catch (error) {
    timeout.cleanup();
    if (isTimeoutLikeFetchFailure(error, timeout.didTimeout(), true)) {
      throw new Error('ElevenLabs request timed out');
    }
    throw error;
  }
  timeout.cleanup();

  if (!response.ok) {
    throw new Error(`ElevenLabs request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
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
  speed?: number
): Promise<ArrayBuffer> => {
  const payload: Record<string, unknown> = {
    text,
    model_id: modelId
  }

  if (speed != null) {
    payload.voice_settings = { speed }
  }

  const timeout = createTimeoutSignal(DEFAULT_ELEVENLABS_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}/text-to-speech/${voiceId}`, {
      method: 'POST',
      headers: {
        'xi-api-key': apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal: timeout.signal,
    });
  } catch (error) {
    timeout.cleanup();
    if (isTimeoutLikeFetchFailure(error, timeout.didTimeout())) {
      throw new Error('ElevenLabs request timed out');
    }
    throw error;
  }
  timeout.cleanup();

  if (!response.ok) {
    throw new Error(`ElevenLabs request failed with status ${response.status}`);
  }

  return response.arrayBuffer();
};
