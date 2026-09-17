import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const repo = process.cwd();
const packet = path.join(repo, '.tmp/uat230-settings-20260917');
const frontend = path.join(repo, 'apps/tldw-frontend');
const require = createRequire(path.join(frontend, 'package.json'));
const { ESLint } = require('eslint');
const ts = require('typescript');
const files = [
  'apps/packages/ui/src/components/Option/Settings/tldw.tsx',
  'apps/packages/ui/src/services/tldw/TldwApiClient.ts',
  'apps/packages/ui/src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx',
  'apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx'
];
const write = (name, value) => fs.writeFileSync(path.join(packet, name), JSON.stringify(value, null, 2) + '\n');
const eslint = new ESLint({ cwd: frontend });
for (const version of ['baseline', 'current']) {
  const results = [];
  for (const file of files) {
    const physical = path.join(version === 'baseline' ? path.join(packet, 'baseline') : repo, file);
    results.push(...await eslint.lintText(fs.readFileSync(physical, 'utf8'), { filePath: path.join(repo, file) }));
  }
  write(`eslint-${version}.json`, results);
}
const configPath = path.join(frontend, 'tsconfig.json');
const config = ts.readConfigFile(configPath, ts.sys.readFile);
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, frontend);
const options = { ...parsed.options, noEmit: true, incremental: false };
const normalized = {};
for (const version of ['baseline', 'current']) {
  const host = ts.createCompilerHost(options, true);
  const original = host.readFile;
  host.readFile = filename => {
    const relative = path.relative(repo, filename).split(path.sep).join('/');
    return version === 'baseline' && files.includes(relative)
      ? fs.readFileSync(path.join(packet, 'baseline', relative), 'utf8')
      : original(filename);
  };
  const program = ts.createProgram(parsed.fileNames, options, host);
  const diagnostics = ts.getPreEmitDiagnostics(program).map(item => ({
    file: item.file ? path.relative(repo, item.file.fileName).split(path.sep).join('/') : null,
    code: item.code,
    message: ts.flattenDiagnosticMessageText(item.messageText, '\n'),
    location: item.file && item.start != null ? item.file.getLineAndCharacterOfPosition(item.start) : null
  }));
  write(`tsc-${version}.json`, diagnostics);
  normalized[version] = diagnostics.map(({location, ...item}) => JSON.stringify(item)).sort();
}
write('tsc-comparison.json', {
  baseline: normalized.baseline.length, current: normalized.current.length,
  added: normalized.current.filter(item => !normalized.baseline.includes(item)).map(JSON.parse),
  removed: normalized.baseline.filter(item => !normalized.current.includes(item)).map(JSON.parse),
  exactSemanticMatch: JSON.stringify(normalized.baseline) === JSON.stringify(normalized.current)
});
console.log(fs.readFileSync(path.join(packet, 'tsc-comparison.json'), 'utf8'));
