import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../api';
import { clearRequestHistory, getRequestHistory } from '../history';

vi.hoisted(() => {
  vi.stubEnv('NEXT_PUBLIC_API_URL', 'https://assets.example.test');
});

const previewPath = '/vn/vn-assets/packs/7/items/11/preview';
const baseURL = 'https://assets.example.test/api/v1';
const headers = { 'Content-Type': 'image/png' };

describe('web preview response lifecycle', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    clearRequestHistory();
  });

  it('cancels reading the response body after headers have arrived', async () => {
    const cancel = vi.fn();
    const response = new Response(new ReadableStream({ cancel }), { headers });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    const controller = new AbortController();
    const preview = apiClient.get<Blob>(previewPath, { baseURL, responseType: 'blob', signal: controller.signal });
    await vi.waitFor(() => expect(response.body?.locked).toBe(true));
    const rejected = expect(preview).rejects.toMatchObject({ name: 'AbortError' });

    controller.abort();

    await rejected;
    expect(cancel).toHaveBeenCalledOnce();
  });

  it('keeps image bytes out of the request history', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('preview', { headers })));

    const preview = await apiClient.get<Blob>(previewPath, { baseURL, responseType: 'blob' });

    expect(preview.size).toBe(7);
    expect(getRequestHistory().find((entry) => entry.url === previewPath)?.responseBody).toEqual({
      size: 7,
      type: 'image/png',
    });
  });
});
