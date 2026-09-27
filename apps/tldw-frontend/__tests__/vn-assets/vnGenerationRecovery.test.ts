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

  it('surfaces quota, read and cleanup failures', () => {
    const prototype = Object.getPrototypeOf(window.sessionStorage);
    const spy = vi.spyOn(prototype, 'setItem').mockImplementation(() => { throw new Error('quota'); });
    expect(() => writeVNCommands(scope, [command])).toThrow(/unavailable/);
    spy.mockRestore();
    vi.spyOn(prototype, 'getItem').mockImplementation(() => { throw new Error('denied'); });
    expect(() => readVNCommands(scope)).toThrow(/unavailable/);
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
});
