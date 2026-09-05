import type { Page, Request, Response } from '@playwright/test';
import { describe, expect, it, vi } from 'vitest';
import { captureAllApiCalls } from '../e2e/utils/api-assertions';

describe('captureAllApiCalls', () => {
  it('waits for in-flight response parsing before returning captured calls', async () => {
    let requestHandler: ((request: Request) => void) | null = null;
    let resolveResponseBody: ((value: unknown) => void) | null = null;

    const response = {
      status: () => 200,
      headers: () => ({ 'content-type': 'application/json' }),
      json: () =>
        new Promise((resolve) => {
          resolveResponseBody = resolve;
        }),
    } as unknown as Response;
    const request = {
      url: () => 'http://127.0.0.1:8000/api/v1/chat/completions',
      method: () => 'POST',
      response: async () => response,
      postDataJSON: () => ({ model: 'local-uat-chat' }),
    } as unknown as Request;
    const page = {
      on: (_event: string, handler: (request: Request) => void) => {
        requestHandler = handler;
      },
      removeListener: vi.fn(),
    } as unknown as Page;

    const capture = captureAllApiCalls(page);
    requestHandler?.(request);
    await Promise.resolve();

    const stopping = capture.stop();
    let settled = false;
    void stopping.then(() => {
      settled = true;
    });
    await Promise.resolve();
    expect(settled).toBe(false);

    resolveResponseBody?.({ ok: true });
    await expect(stopping).resolves.toEqual([
      expect.objectContaining({
        method: 'POST',
        requestBody: { model: 'local-uat-chat' },
        responseBody: { ok: true },
        status: 200,
      }),
    ]);
  });

  it('does not wait for event-stream response bodies', async () => {
    let requestHandler: ((request: Request) => void) | null = null;
    const parseResponseBody = vi.fn(() => new Promise(() => {}));

    const response = {
      status: () => 200,
      headers: () => ({ 'content-type': 'text/event-stream; charset=utf-8' }),
      json: parseResponseBody,
    } as unknown as Response;
    const request = {
      url: () => 'http://127.0.0.1:8000/api/v1/chats/42/complete-v2',
      method: () => 'POST',
      response: async () => response,
      postDataJSON: () => ({ stream: true }),
    } as unknown as Request;
    const page = {
      on: (_event: string, handler: (request: Request) => void) => {
        requestHandler = handler;
      },
      removeListener: vi.fn(),
    } as unknown as Page;

    const capture = captureAllApiCalls(page);
    requestHandler?.(request);

    await expect(capture.stop()).resolves.toEqual([
      expect.objectContaining({
        method: 'POST',
        requestBody: { stream: true },
        responseBody: null,
        status: 200,
      }),
    ]);
    expect(parseResponseBody).not.toHaveBeenCalled();
  });
});
