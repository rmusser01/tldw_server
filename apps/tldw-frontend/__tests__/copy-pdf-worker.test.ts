import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { expect, it } from 'vitest';

const script = path.resolve(__dirname, '../scripts/copy-pdf-worker.mjs');

const copyFromFixture = (worker: 'legacy' | 'modern' | null) => {
  const root = mkdtempSync(path.join(tmpdir(), 'pdf-worker-resolution-'));
  const put = (name: string, text: string) => {
    const target = path.join(root, name);
    mkdirSync(path.dirname(target), { recursive: true });
    writeFileSync(target, text);
  };
  try {
    put('package.json', '{}');
    put('node_modules/pdfjs-dist/package.json', '{"name":"pdfjs-dist"}');
    put('node_modules/pdfjs-dist/legacy/build/pdf.worker.min.mjs', 'wrong-hoisted-worker');
    put('node_modules/pdfjs-dist/build/pdf.worker.min.mjs', 'wrong-hoisted-worker');
    put('node_modules/react-pdf/package.json', '{"name":"react-pdf"}');
    put('node_modules/@tldw/ui/package.json', '{"name":"@tldw/ui"}');
    const dependency = 'node_modules/@tldw/ui/node_modules/react-pdf';
    put(dependency + '/package.json', '{"name":"react-pdf"}');
    put(dependency + '/node_modules/pdfjs-dist/package.json', '{"name":"pdfjs-dist"}');
    if (worker !== null) {
      put(
        dependency +
          '/node_modules/pdfjs-dist/' +
          (worker === 'legacy' ? 'legacy/' : '') +
          'build/pdf.worker.min.mjs',
        'shared-viewer-worker'
      );
    }
    mkdirSync(path.join(root, 'scripts'));
    copyFileSync(script, path.join(root, 'scripts/copy-pdf-worker.mjs'));
    const result = spawnSync(process.execPath, [path.join(root, 'scripts/copy-pdf-worker.mjs')], {
      cwd: root,
      encoding: 'utf8',
      timeout: 5000,
    });
    const output = path.join(root, 'public/pdf.worker.min.mjs');
    return {
      status: result.status,
      error: result.error,
      copied: existsSync(output) ? readFileSync(output, 'utf8') : null,
    };
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
};

it('copies the shared viewer dependency instead of a conflicting hoisted worker', () => {
  const result = copyFromFixture('legacy');
  expect(result.error).toBeUndefined();
  expect(result.status).toBe(0);
  expect(result.copied).toBe('shared-viewer-worker');
});

it('accepts the modern worker from that same resolved dependency', () => {
  const result = copyFromFixture('modern');
  expect(result.status).toBe(0);
  expect(result.copied).toBe('shared-viewer-worker');
});

it('refuses a missing viewer worker instead of copying a different package', () => {
  const result = copyFromFixture(null);
  expect(result.error).toBeUndefined();
  expect(result.status).not.toBe(0);
  expect(result.copied).toBeNull();
});
