/* @vitest-environment jsdom */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  IdempotentCommandStateError,
  createIdempotentCommand,
} from '../idempotent-command';

const BODY = {
  url: 'https://receiver.example/hooks/private',
  event_types: ['incident.created'],
  description: 'Incident receiver',
};

describe('idempotent command lifecycle', () => {
  let nextByte: number;

  beforeEach(() => {
    nextByte = 0;
    vi.stubGlobal('crypto', {
      getRandomValues: vi.fn((target: Uint8Array) => {
        for (let index = 0; index < target.length; index += 1) {
          target[index] = nextByte % 256;
          nextByte += 1;
        }
        return target;
      }),
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('uses 16 random bytes as 32 lowercase hexadecimal characters', () => {
    const command = createIdempotentCommand('create', BODY, vi.fn());

    expect(crypto.getRandomValues).toHaveBeenCalledWith(expect.any(Uint8Array));
    const randomBytes = vi.mocked(crypto.getRandomValues).mock.calls[0]?.[0];
    expect(randomBytes).toHaveLength(16);
    expect(command.idempotencyKey).toBe('000102030405060708090a0b0c0d0e0f');
  });

  it('reuses one key and one normalized body only for the same retryable command', async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('network'))
      .mockResolvedValueOnce({ ok: true });
    const source = structuredClone(BODY);
    const command = createIdempotentCommand('create', source, request);
    source.description = 'changed after command creation';
    source.event_types.push('user.created');

    await expect(command.run()).rejects.toThrow('network');
    expect(command.canRetry).toBe(true);
    await expect(command.retry()).resolves.toEqual({ ok: true });

    expect(request).toHaveBeenCalledTimes(2);
    expect(request.mock.calls[0]?.[0]).toBe(request.mock.calls[1]?.[0]);
    expect(request.mock.calls[0]?.[0]).toEqual({
      operation: 'create',
      body: BODY,
      idempotencyKey: command.idempotencyKey,
    });
    expect(Object.isFrozen(request.mock.calls[0]?.[0])).toBe(true);
    expect(Object.isFrozen(request.mock.calls[0]?.[0].body)).toBe(true);
    expect(Object.isFrozen(request.mock.calls[0]?.[0].body.event_types)).toBe(true);

    const nextCommand = createIdempotentCommand('create', BODY, request);
    expect(nextCommand.idempotencyKey).not.toBe(command.idempotencyKey);
  });

  it('does not make non-transport failures retryable or retry automatically', async () => {
    const request = vi.fn().mockRejectedValue(new Error('HTTP 409'));
    const command = createIdempotentCommand('create', BODY, request);

    await expect(command.run()).rejects.toThrow('HTTP 409');

    expect(request).toHaveBeenCalledTimes(1);
    expect(command.canRetry).toBe(false);
    await expect(command.retry()).rejects.toBeInstanceOf(IdempotentCommandStateError);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it('allows an explicit same-command retry after the proxy reports transport loss', async () => {
    const transportError = new Error('Webhook backend is unavailable');
    transportError.name = 'WebhookTransportError';
    const request = vi.fn()
      .mockRejectedValueOnce(transportError)
      .mockResolvedValueOnce({ ok: true });
    const command = createIdempotentCommand('create', BODY, request);

    await expect(command.run()).rejects.toBe(transportError);
    expect(command.canRetry).toBe(true);
    await expect(command.retry()).resolves.toEqual({ ok: true });
    expect(request.mock.calls[0]?.[0].idempotencyKey).toBe(
      request.mock.calls[1]?.[0].idempotencyKey,
    );
  });

  it('clears retry eligibility after completion', async () => {
    const request = vi.fn().mockResolvedValue({ ok: true });
    const command = createIdempotentCommand('rotate', { webhookId: 41 }, request);

    await command.run();

    expect(command.canRetry).toBe(false);
    await expect(command.retry()).rejects.toBeInstanceOf(IdempotentCommandStateError);
    await expect(command.run()).rejects.toBeInstanceOf(IdempotentCommandStateError);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it('keeps idempotency material out of browser storage and URLs', async () => {
    const localStorageSpy = vi.spyOn(Storage.prototype, 'setItem');
    const request = vi.fn().mockResolvedValue({ ok: true });
    const command = createIdempotentCommand('create', BODY, request);

    await command.run();

    expect(localStorageSpy).not.toHaveBeenCalled();
    expect(window.location.href).not.toContain(command.idempotencyKey);
    expect(JSON.stringify(request.mock.calls[0]?.[0])).toContain(command.idempotencyKey);
    expect(Object.keys(request.mock.calls[0]?.[0] ?? {}).sort()).toEqual([
      'body',
      'idempotencyKey',
      'operation',
    ]);
  });
});
