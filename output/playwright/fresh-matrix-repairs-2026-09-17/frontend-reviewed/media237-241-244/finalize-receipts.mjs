import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
const dir = path.dirname(fileURLToPath(import.meta.url));
const read = name => JSON.parse(fs.readFileSync(path.join(dir, name)));
const write = (name, value) => fs.writeFileSync(path.join(dir, name), JSON.stringify(value, null, 2) + '\n');
const signatureCounts = values => values.reduce((out, value) => { const key = JSON.stringify(value); out[key] = (out[key] || 0) + 1; return out; }, {});
const baseline = read('tsc-fresh-baseline.json'), current = read('tsc-fresh-current.json');
const a = signatureCounts(baseline), b = signatureCounts(current);
write('compiler-comparison.json', { baseline: baseline.length, current: current.length, added: Object.entries(b).filter(([k, n]) => n > (a[k] || 0)), removed: Object.entries(a).filter(([k, n]) => n > (b[k] || 0)) });
const before = read('eslint-baseline.json'), after = read('eslint-current.json'), introduced = [];
for (const file of after) {
  const old = before.find(value => value.filePath === file.filePath);
  const sig = value => value.ruleId + ' ' + value.message.replace(/line \d+/g, 'line N');
  const counts = new Map();
  for (const value of old?.messages || []) counts.set(sig(value), (counts.get(sig(value)) || 0) + 1);
  for (const value of file.messages) { const key = sig(value); if (counts.get(key)) counts.set(key, counts.get(key) - 1); else introduced.push({ file: file.filePath, line: value.line, message: value.message }); }
}
write('eslint-comparison.json', { baselineWarnings: before.reduce((s, x) => s + x.warningCount, 0), currentWarnings: after.reduce((s, x) => s + x.warningCount, 0), currentErrors: after.reduce((s, x) => s + x.errorCount, 0), introduced });
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const changed = read('source-freeze.json').files.filter(file => sha(fs.readFileSync(file.path)) !== file.sha256);
if (changed.length) throw Error('Source changed after freeze');
const patterns = { jwt: /eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/, privateKey: /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/, liveKey: /sk-(?:proj-|live-)[A-Za-z0-9_-]{20,}/, credentialUrl: /(?:postgres(?:ql)?|https?):\/\/[^\s:/]+:[^\s@/]+@/ };
const files = fs.readdirSync(dir).filter(name => name !== 'evidence-manifest.json' && fs.statSync(path.join(dir, name)).isFile());
const hits = [];
for (const file of files) for (const [rule, regex] of Object.entries(patterns)) if (regex.test(fs.readFileSync(path.join(dir, file), 'utf8'))) hits.push({ file, rule });
write('scan.json', { checkedAt: new Date().toISOString(), files: files.length, rules: Object.keys(patterns), hits });
write('evidence-manifest.json', { createdAt: new Date().toISOString(), files: files.map(file => ({ path: path.relative(process.cwd(), path.join(dir, file)), sha256: sha(fs.readFileSync(path.join(dir, file))) })) });
console.log(JSON.stringify({ compiler: read('compiler-comparison.json'), lint: read('eslint-comparison.json'), frozenHashes: 'match', scanHits: hits.length }));
