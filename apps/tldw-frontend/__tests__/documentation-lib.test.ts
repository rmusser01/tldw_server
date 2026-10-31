import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { listDocumentationManifest, readDocumentationContent } from '@web/lib/documentation';

describe('documentation library in a standalone runtime', () => {
  let host: string;
  let bundle: string;

  beforeEach(async () => {
    host = await mkdtemp(path.join(tmpdir(), 'email-docs-standalone-'));
    bundle = path.join(host, 'standalone');
    const runtime = path.join(bundle, 'apps/tldw-frontend');
    await mkdir(runtime, { recursive: true });
    await mkdir(path.join(bundle, 'Docs/Published'), { recursive: true });
    await writeFile(path.join(bundle, 'Docs/Published/guide.md'), '# Runtime guide\n');
    vi.spyOn(process, 'cwd').mockReturnValue(runtime);
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await rm(host, { recursive: true, force: true });
  });

  it('discovers documentation from the standalone bundle', async () => {
    expect((await listDocumentationManifest()).server.map((doc) => doc.relativePath)).toEqual([
      'guide.md',
    ]);
  });

  it('reads documentation from the standalone bundle', async () => {
    expect(await readDocumentationContent('server', 'guide.md')).toBe('# Runtime guide\n');
  });

  it('rejects traversal outside the standalone documentation root', async () => {
    await expect(readDocumentationContent('server', '../outside.md')).rejects.toThrow(
      'Invalid documentation path.'
    );
  });

  it('rejects unsupported file types in the standalone bundle', async () => {
    await expect(readDocumentationContent('server', 'guide.txt')).rejects.toThrow(
      'Unsupported documentation file type.'
    );
  });

  it('fails closed when bundle documentation is missing despite host documentation', async () => {
    await rm(path.join(bundle, 'Docs'), { recursive: true });
    await mkdir(path.join(host, 'Docs/Published'), { recursive: true });
    await writeFile(path.join(host, 'Docs/Published/guide.md'), '# Unrelated host guide\n');

    await expect(listDocumentationManifest()).rejects.toThrow(
      'Unable to resolve repository root for documentation sources.'
    );
  });

  it('does not read host content when bundle documentation is missing', async () => {
    await rm(path.join(bundle, 'Docs'), { recursive: true });
    await mkdir(path.join(host, 'Docs/Published'), { recursive: true });
    await writeFile(path.join(host, 'Docs/Published/guide.md'), '# Unrelated host guide\n');

    await expect(readDocumentationContent('server', 'guide.md')).rejects.toThrow(
      'Unable to resolve repository root for documentation sources.'
    );
  });
});

describe('documentation library in the repository', () => {
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
