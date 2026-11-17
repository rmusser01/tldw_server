import { beforeEach, expect, it, vi } from 'vitest';
import { Storage } from '../extension/shims/plasmo-storage';

beforeEach(() => {
  localStorage.clear();
  vi.resetModules();
});
const record = (clipId: string) => ({
  ownerScope: 'alice',
  workspaceId: 'workspace',
  sourceId: `web-clipper:${clipId}`,
  body: {
    clip_id: clipId,
    clip_type: 'article',
    source_url: 'https://example.org',
    source_title: 'Article',
    workspace: { workspace_id: 'workspace' },
    content: { full_extract: 'Immutable' },
  },
});

it('keeps independent context captures with the actual WebUI shim and parsed getAll', async () => {
  const first = await import('@/utils/research-workspace-prefill');
  vi.resetModules();
  const second = await import('@/utils/research-workspace-prefill');
  await Promise.all([
    first.saveResearchWebCapture(record('one')),
    second.saveResearchWebCapture(record('two')),
  ]);
  expect(
    (await second.readResearchWebCaptures('alice', 'workspace'))
      .map((item) => item.body.clip_id)
      .sort()
  ).toEqual(['one', 'two']);
  const storage = new Storage({ area: 'local' });
  expect(Object.values(await storage.getAll()).every((value) => typeof value === 'object')).toBe(
    true
  );
});

it('rejects silently dropped native writes before reporting a durable WebUI checkpoint', async () => {
  const owning = await import('@/utils/research-workspace-prefill');
  const nativeSet = localStorage.setItem.bind(localStorage);
  const spy = vi
    .spyOn(Object.getPrototypeOf(localStorage), 'setItem')
    .mockImplementation((key: string, value: string) => {
      if (!key.includes('web-captures')) nativeSet(key, value);
    });
  try {
    await expect(owning.saveResearchWebCapture(record('lost'))).rejects.toThrow('Could not retain');
  } finally {
    spy.mockRestore();
  }
  expect(await owning.readResearchWebCaptures('alice', 'workspace')).toEqual([]);
});
