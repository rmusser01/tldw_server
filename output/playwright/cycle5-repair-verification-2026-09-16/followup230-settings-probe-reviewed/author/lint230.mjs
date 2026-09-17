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
const configModule = await import(path.join(frontend, 'eslint.config.mjs'));
const eslint = new ESLint({ cwd: repo, overrideConfigFile: true, overrideConfig: [...configModule.default, { settings: { next: { rootDir: frontend } } }] });
for (const version of ['baseline', 'current']) {
  const results = [];
  for (const file of files) {
    const physical = path.join(version === 'baseline' ? path.join(packet, 'baseline') : repo, file);
    results.push(...await eslint.lintText(fs.readFileSync(physical, 'utf8'), { filePath: path.join(repo, file) }));
  }
  write(`eslint-${version}.json`, results);
}

const normalizedLint = {};
for (const version of ['baseline','current']) {
 const results=JSON.parse(fs.readFileSync(path.join(packet, 'eslint-'+version+'.json'),'utf8'));
 normalizedLint[version]=results.flatMap(r=>r.messages.map(m=>JSON.stringify({file:path.relative(repo,r.filePath),ruleId:m.ruleId,severity:m.severity,message:m.message}))).sort();
}
write('eslint-comparison.json',{baseline:normalizedLint.baseline.length,current:normalizedLint.current.length,added:normalizedLint.current.filter(x=>!normalizedLint.baseline.includes(x)),removed:normalizedLint.baseline.filter(x=>!normalizedLint.current.includes(x)),exactMatch:JSON.stringify(normalizedLint.baseline)===JSON.stringify(normalizedLint.current)});
console.log(fs.readFileSync(path.join(packet,'eslint-comparison.json'),'utf8'));
