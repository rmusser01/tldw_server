import { expect, type APIRequestContext, type TestInfo } from '@playwright/test';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { TEST_CONFIG } from './helpers';

/** Only synthetic Notes allocated by the calling test are retired here. */
export async function withOwnedNote<T>(
  request: APIRequestContext,
  info: Pick<TestInfo, 'outputPath'>,
  data: { title: string; content: string; content_language: string | null },
  run: (note: { id: string; title: string }) => Promise<T>
): Promise<T> {
  const headers = { 'X-API-KEY': TEST_CONFIG.apiKey };
  const response = await request.post(`${TEST_CONFIG.serverUrl}/api/v1/notes/`, {
    headers,
    data,
  });
  expect(response.ok()).toBeTruthy();
  const allocated = await response.json();
  const note = { id: String(allocated.id), title: data.title };
  // Evidence, browser/profile setup and the test body are all inside cleanup.
  try {
    const evidence = info.outputPath(`owned-note-${encodeURIComponent(note.id)}.json`);
    mkdirSync(path.dirname(evidence), { recursive: true });
    writeFileSync(evidence, JSON.stringify({ ...note, createdAt: Date.now() }) + '\n');
    return await run(note);
  } finally {
    const current = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
      headers,
    });
    if (current.ok()) {
      const saved = await current.json();
      expect(
        (
          await request.delete(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {
            headers: { ...headers, 'expected-version': String(saved.version) },
          })
        ).ok()
      ).toBeTruthy();
    } else {
      expect(current.status(), 'Owned Note cleanup lookup').toBe(404);
    }
  }
}

/** An actually allocated OS-temp profile is cleaned even if launch fails. */
export async function withOwnedExtensionProfile<T>(run: (profile: string) => Promise<T>) {
  const profile = mkdtempSync(path.join(tmpdir(), 'tldw-notes-extension-'));
  try {
    return await run(profile);
  } finally {
    rmSync(profile, { recursive: true, force: true });
  }
}
