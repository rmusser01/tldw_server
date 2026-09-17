// Read-only source/static validation. Does not import or invoke either launcher.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
const out = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(out, '../../..');
const require = createRequire(path.join(repo, 'apps/tldw-frontend/package.json'));
const { ESLint } = require('eslint');
const config = (await import(pathToFileURL(path.join(repo, 'apps/tldw-frontend/eslint.config.mjs')))).default;
const linter = new ESLint({ cwd: repo, ignore: false, overrideConfigFile: true, overrideConfig: [...config, { settings: { next: { rootDir: path.join(repo, 'apps/tldw-frontend') }, react: { version: require('react/package.json').version } } }] });
const files = ['.tmp/uat-next-matrix-20260916/matrix-launcher.mjs', '.tmp/uat-next-matrix-20260916/matrix-upgrade.mjs', '.tmp/uat-repairs-231-246/native-upgrade-preparation/matrix-upgrade.test.mjs'];
const baseline = fs.readFileSync(path.join(out, '../native-upgrade-preparation/baseline/matrix-launcher.mjs'), 'utf8');
const normalized = fs.readFileSync(path.join(repo, files[0]), 'utf8').replace('export function requireInitialized(', 'function requireInitialized(').replace('export function frontendEnv(', 'function frontendEnv(').replace('export async function runtimeEnv(', 'async function runtimeEnv(');
const reports = [];
for (const file of files) {
  const result = (await linter.lintText(fs.readFileSync(path.join(repo, file), 'utf8'), { filePath: path.join(repo, file) }))[0];
  reports.push({ path: file, errors: result.errorCount, warnings: result.warningCount, messages: result.messages, ignored: result.messages.some(message => /ignored|outside of base path/i.test(message.message)) });
}
const old = (await linter.lintText(baseline, { filePath: path.join(repo, files[0]) }))[0];
const result = { exactThreeExportOnly: normalized === baseline, reports, baseline: { errors: old.errorCount, warnings: old.warningCount, messages: old.messages }, javascriptSecurityLimit: 'Bandit cannot parse these three JS files; no JS security certification claimed.' };
fs.writeFileSync(path.join(out, 'static-final.json'), JSON.stringify(result, null, 2) + '\n', { mode: 0o600 });
console.log(JSON.stringify(result));
if (!result.exactThreeExportOnly || reports.some(report => report.errors || report.ignored)) process.exitCode = 1;
