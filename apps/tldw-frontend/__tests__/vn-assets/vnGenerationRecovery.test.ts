import { beforeEach, describe, expect, it, vi } from 'vitest';
import { clearVNCommands, createVNCommandScope, readVNCommands, writeVNCommands } from '@web/lib/vnGenerationRecovery';

const key = 'tldw:vn-generation:pending:v1';
const scope = { server: 'https://vn.example/api/v1', principal: '1' };
const command = { packId: 7, slotId: 12, request: { idempotency_key: 'vn-generation-test-key', source_batch_id: 41 } };

describe('VN generation recovery journal', () => {
  beforeEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

  it('roundtrips the original request without credentials or prompts', () => {
    writeVNCommands(scope, [command]);
    expect(JSON.parse(sessionStorage.getItem(key)!)).toEqual({ version: 1, scope, commands: [command] });
    expect(readVNCommands(scope)).toEqual([command]);
    writeVNCommands(scope, []);
    expect(sessionStorage.getItem(key)).toBeNull();
  });

  it.each([
    { server: 'https://other.example/api/v1', principal: '1' },
    { server: 'https://vn.example/other/api/v1', principal: '1' },
    { server: 'https://vn.example/api/v1', principal: '2' },
  ])('never restores a different authority: %j', (other) => {
    writeVNCommands(scope, [command]);
    expect(readVNCommands(other)).toEqual([]);
    expect(sessionStorage.getItem(key)).toBeNull();
  });

  it.each([
    '',
    '{',
    JSON.stringify({ version: 2, scope, commands: [command] }),
    JSON.stringify({ version: 1, scope, commands: [{ ...command, packId: 0 }] }),
    JSON.stringify({ version: 1, scope, commands: [{ ...command, slotId: 1.5 }] }),
    JSON.stringify({ version: 1, scope, commands: [{ ...command, request: { ...command.request, source_batch_id: true } }] }),
    JSON.stringify({ version: 1, scope, commands: [{ ...command, request: { ...command.request, prompt: 'secret' } }] }),
    JSON.stringify({ version: 1, scope, commands: [command, command] }),
  ])('rejects an unreadable journal without silently losing its ambiguity', (raw) => {
    sessionStorage.setItem(key, raw);
    expect(() => readVNCommands(scope)).toThrow(/could not be read/);
    expect(sessionStorage.getItem(key)).toBe(raw);
  });

  it('permits generation when the journal key is absent', () => {
    expect(sessionStorage.getItem(key)).toBeNull();
    expect(readVNCommands(scope)).toEqual([]);
  });

  it.each([
    { ...scope, server: 'http://[' },
    { ...scope, server: 'not-a-url' },
    { ...scope, principal: 'x' },
    { ...scope, server: `${scope.server}/` },
    { ...scope, server: 'https://VN.example/api/v1' },
    { ...scope, server: `${scope.server}?token=secret` },
    { ...scope, principal: '01' },
    { ...scope, principal: '1.0' },
  ])('retains an invalid or noncanonical saved scope behind warned discard: %j', (savedScope) => {
    const raw = JSON.stringify({ version: 1, scope: savedScope, commands: [command] });
    sessionStorage.setItem(key, raw);
    expect(() => readVNCommands(scope)).toThrow(/could not be read/);
    expect(sessionStorage.getItem(key)).toBe(raw);
  });

  it('surfaces a quota failure when writing a command', () => {
    const prototype = Object.getPrototypeOf(window.sessionStorage);
    vi.spyOn(prototype, 'setItem').mockImplementation(() => { throw new Error('quota'); });
    expect(() => writeVNCommands(scope, [command])).toThrow(/unavailable/);
  });

  it.each(['capacity', 'invalid', 'duplicate'])('keeps readable commands intact after %s write validation fails', (failure) => {
    const saved = Array.from({ length: 64 }, (_, index) => ({ ...command, packId: index + 20 }));
    writeVNCommands(scope, saved);
    expect(readVNCommands(scope)).toEqual(saved);
    const raw = sessionStorage.getItem(key);
    const next = failure === 'capacity' ? [...saved, command]
      : failure === 'invalid' ? [{ ...command, request: { idempotency_key: 'invalid' } }]
        : [command, command];
    expect(() => writeVNCommands(scope, next)).toThrow('The generation request could not be saved for recovery. No request was sent.');
    expect(sessionStorage.getItem(key)).toBe(raw);
    expect(readVNCommands(scope)).toEqual(saved);
  });

  it('surfaces a denied read of saved commands', () => {
    const prototype = Object.getPrototypeOf(window.sessionStorage);
    vi.spyOn(prototype, 'getItem').mockImplementation(() => { throw new Error('denied'); });
    expect(() => readVNCommands(scope)).toThrow(/unavailable/);
  });

  it('surfaces a denied cleanup of saved commands', () => {
    const prototype = Object.getPrototypeOf(window.sessionStorage);
    vi.spyOn(prototype, 'removeItem').mockImplementation(() => { throw new Error('denied'); });
    expect(() => clearVNCommands()).toThrow(/unavailable/);
  });

  it('normalizes the API base but refuses credential-bearing URLs or unverified IDs', () => {
    expect(createVNCommandScope('https://VN.example/api/v1/', 1)).toEqual(scope);
    for (const server of ['https://secret@vn.example', 'https://vn.example?token=secret', 'file:///api']) {
      expect(() => createVNCommandScope(server, 1)).toThrow();
    }
    expect(() => createVNCommandScope(scope.server, 'cached-user')).toThrow();
  });

  it('writes a canonical scope that remains readable after repeated trailing slashes', () => {
    const normalized = createVNCommandScope(`${scope.server}///`, 1);
    expect(normalized).toEqual(scope);
    writeVNCommands(normalized, [command]);
    expect(readVNCommands(normalized)).toEqual([command]);
  });
});
