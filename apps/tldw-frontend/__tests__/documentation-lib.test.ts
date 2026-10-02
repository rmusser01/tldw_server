import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { describe, expect, it, vi } from 'vitest';

import { listDocumentationManifest, readDocumentationContent } from '@web/lib/documentation';

describe('documentation library', () => {
  it('loads real documentation from a nested standalone runtime', async () => {
    const root = mkdtempSync(path.join(tmpdir(), 'email-docs-standalone-'));
    const runtime = path.join(root, 'apps/tldw-frontend/.next/standalone/apps/tldw-frontend');
    mkdirSync(runtime, { recursive: true });
    mkdirSync(path.join(root, 'Docs/Published'), { recursive: true });
    writeFileSync(path.join(root, 'Docs/Published/guide.md'), '# Runtime guide\n');
    const cwd = vi.spyOn(process, 'cwd').mockReturnValue(runtime);
    try {
      expect((await listDocumentationManifest()).server.map((doc) => doc.relativePath)).toEqual([
        'guide.md',
      ]);
      expect(await readDocumentationContent('server', 'guide.md')).toBe('# Runtime guide\n');
      await expect(readDocumentationContent('server', '../outside.md')).rejects.toThrow(
        'Invalid documentation path.'
      );
      await expect(readDocumentationContent('server', 'guide.txt')).rejects.toThrow(
        'Unsupported documentation file type.'
      );
    } finally {
      cwd.mockRestore();
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('discovers published server documentation from the repository', async () => {
    const manifest = await listDocumentationManifest();

    expect(
      manifest.server.some((doc) => doc.relativePath === 'API-related/AuthNZ-API-Guide.md')
    ).toBe(true);
  });

  it('reads published server documentation content', async () => {
    const content = await readDocumentationContent('server', 'API-related/AuthNZ-API-Guide.md');

    expect(content).toContain('# AuthNZ API Guide');
  });

  it('rejects path traversal outside the documentation roots', async () => {
    await expect(readDocumentationContent('server', '../README.md')).rejects.toThrow(
      'Invalid documentation path.'
    );
  });
});
