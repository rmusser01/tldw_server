import { afterEach, describe, expect, it, vi } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import type { APIRequestContext } from '@playwright/test';
import { withOwnedNote, withOwnedExtensionProfile } from '../utils/notes-owned';

const roots: string[] = [];
afterEach(() => roots.splice(0).forEach((root) => rmSync(root, { recursive: true, force: true })));
function request() {
  return {
    post: vi.fn(async () => ({ ok: () => true, json: async () => ({ id: 'owned' }) })),
    get: vi.fn(async () => ({ ok: () => true, json: async () => ({ version: 3 }) })),
    delete: vi.fn(async () => ({ ok: () => true })),
  };
}
const data = { title: 'Synthetic', content: '{}', content_language: 'json' };
describe('committed Notes test resource lifecycle', () => {
  it('creates a missing Playwright output parent before writing evidence', async () => {
    const root = mkdtempSync(path.join(tmpdir(), 'tldw-notes-output-'));
    roots.push(root);
    const api = request();
    const output = path.join(root, 'not-created', 'nested', 'owned-note.json');
    await withOwnedNote(
      api as unknown as APIRequestContext,
      { outputPath: () => output },
      data,
      async () => {
        expect(JSON.parse(readFileSync(output, 'utf8')).id).toBe('owned');
      }
    );
    expect(api.delete).toHaveBeenCalledWith(
      expect.stringContaining('/notes/owned'),
      expect.objectContaining({ headers: expect.objectContaining({ 'expected-version': '3' }) })
    );
  });
  it.each(['evidence', 'body'])(
    'retires the allocated record after %s setup fails',
    async (failure) => {
      const api = request();
      const root = mkdtempSync(path.join(tmpdir(), 'tldw-notes-output-'));
      roots.push(root);
      await expect(
        withOwnedNote(
          api as unknown as APIRequestContext,
          {
            outputPath: () => {
              if (failure === 'evidence') throw Error('Owned output failure');
              return path.join(root, 'note.json');
            },
          },
          data,
          async () => {
            throw Error('Owned body failure');
          }
        )
      ).rejects.toThrow(failure === 'evidence' ? 'Owned output failure' : 'Owned body failure');
      expect(api.get).toHaveBeenCalledOnce();
      expect(api.delete).toHaveBeenCalledOnce();
    }
  );
  it('allocates the OS-temp profile and removes it after launch/body failure', async () => {
    let allocated = '';
    await expect(
      withOwnedExtensionProfile(async (profile) => {
        allocated = profile;
        expect(profile.startsWith(tmpdir())).toBe(true);
        expect(existsSync(profile)).toBe(true);
        throw Error('Owned launch failure');
      })
    ).rejects.toThrow('Owned launch failure');
    expect(existsSync(allocated)).toBe(false);
  });
});
