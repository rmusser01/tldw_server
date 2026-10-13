/** Test-only local assets. Run from any cwd; --clean removes only these copies. */
import { cpSync, existsSync, rmSync, readdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const require = createRequire(resolve(web, 'package.json'));
const monacoRequire = createRequire(require.resolve('@monaco-editor/react'));
const source = resolve(dirname(monacoRequire.resolve('monaco-editor/package.json')), 'min/vs');
for (const target of [
  resolve(web, 'public/i1-monaco/vs'),
  resolve(web, '../packages/ui/src/public/i1-monaco/vs'),
]) {
  if (process.argv.includes('--clean')) {
    if (existsSync(target)) rmSync(resolve(target, '..'), { recursive: true });
  } else {
    cpSync(source, target, { recursive: true });
    for (const label of ['editor', 'json', 'css', 'html', 'ts']) {
      const name = readdirSync(resolve(source, 'assets')).find(
        (file) => file.startsWith(`${label}.worker-`) && file.endsWith('.js')
      );
      if (!name) throw new Error(`Missing installed ${label} worker`);
      writeFileSync(
        resolve(target, '..', `${label}-worker.js`),
        `importScripts('./vs/assets/${name}');\nglobalThis.postMessage({type:'vscode-worker-ready'});\n`
      );
    }
  }
  console.log(`${process.argv.includes('--clean') ? 'Removed' : 'Copied'} ${target}`);
}
