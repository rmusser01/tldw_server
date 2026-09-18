import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root=process.cwd();
const dir='.tmp/uat-repairs-231-246/capability258-review';
const hash=b=>crypto.createHash('sha256').update(b).digest('hex');
const files=[
 '.superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-258-brief.md',
 '.superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-258-report.md',
 '.superpowers/sdd/IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs/task-258-review.diff',
 'apps/packages/ui/src/services/tldw/server-capabilities.ts',
 'apps/packages/ui/src/services/__tests__/server-capabilities.test.ts',
 'apps/packages/ui/src/services/tldw/TldwApiClient.ts',
 'apps/packages/ui/src/services/tldw/runtime-auth-override.ts',
 'apps/packages/ui/src/services/tldw/deployment-mode.ts',
 'apps/packages/ui/src/services/tldw/browser-networking.ts',
 'apps/packages/ui/src/services/tldw/single-user-credential.ts',
 'apps/packages/ui/src/utils/api-key.ts',
 'apps/packages/ui/src/services/chat-surface-scope.ts',
 'apps/packages/ui/src/services/background-proxy.ts',
 'apps/packages/ui/src/hooks/useServerCapabilities.ts',
 'apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts',
 'apps/packages/ui/src/services/__tests__/chat-surface-scope.test.ts',
 'apps/packages/ui/src/services/tldw/__tests__/single-user-credential.test.ts',
 'apps/packages/ui/vitest.config.ts','apps/packages/ui/vitest.setup.ts',
 '.tmp/uat-repairs-231-246/capability258/compiler-compare.mjs',
 '.tmp/uat-repairs-231-246/capability258/compiler-comparison.json',
 '.tmp/uat-repairs-231-246/capability258/evidence.json',
 ...['cookie-guard.review.test.ts','vitest.config.mjs','focused.json','focused.log','cookie-guard.json','cookie-guard.log','audit.mjs'].map(f=>`${dir}/${f}`)
];
const inputs=files.map(p=>{const b=fs.readFileSync(path.join(root,p));return {path:p,bytes:b.length,sha256:hash(b)};});
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p)));
const focused=read(`${dir}/focused.json`),diagnostic=read(`${dir}/cookie-guard.json`);
const evidence=read('.tmp/uat-repairs-231-246/capability258/evidence.json');
const compiler=read('.tmp/uat-repairs-231-246/capability258/compiler-comparison.json');
const preserved259={
 'REVIEW.md':'c3b1a4e2c71115e943dc30634501f908a39df90ebeee93da668b1e726386d281',
 'audit.mjs':'1c437a6f8269ad2331440bb111ec41c67907b15e99699983987716ca6a9ccf1c',
 'audit.json':'7b9b7854c35dd16864c2a27d6486742aa62a0f7bcbd91d2eb202d4b57f4debf5'
};
const unchanged259=Object.entries(preserved259).map(([file,expected])=>({file,sha256:hash(fs.readFileSync(path.join(root,'.tmp/uat-repairs-231-246/trash259-diagnosis',file))),expected}));
const check=(name,pass)=>({name,pass});
const assertions=diagnostic.testResults.flatMap(s=>s.assertionResults||[]);
const checks=[
 check('Independent existing focused and adjacent tests: 117 pass, zero failures',focused.numTotalTests===117&&focused.numPassedTests===117&&focused.numFailedTests===0),
 check('Real client cookie diagnostic has two dispatch regressions and one valid control',diagnostic.numTotalTests===3&&diagnostic.numFailedTests===2&&diagnostic.numPassedTests===1&&assertions.filter(a=>a.status==='failed').every(a=>a.failureMessages.some(m=>m.includes('to have a length of +0 but got 1')))),
 check('Actual reviewed product and author test bytes match author evidence',Object.entries(evidence.currentSha256).every(([p,h])=>inputs.find(i=>i.path===p)?.sha256===h)),
 check('Author compiler comparison reports identical 90 diagnostics',compiler.baselineCount===90&&compiler.currentCount===90&&compiler.added.length===0&&compiler.removed.length===0),
 check('UAT259 artifacts remain byte-identical',unchanged259.every(i=>i.sha256===i.expected))
];
const result={task:'UAT258 / TASK13260.200',at:new Date().toISOString(),verdict:'CHANGES_REQUESTED',findings:[{priority:'P2',file:'apps/packages/ui/src/services/tldw/server-capabilities.ts',start:743,end:744,title:'Use the active cookie transport predicate before probing',cause:'getConfig can return a stored foreign-origin or wrong-auth-mode cookie marker. The new guard accepts the marker alone, while actual ensureConfigForRequest rejects it. Protected capability dispatch is therefore attempted without a supported authenticated transport.',evidence:'Review-only tests execute actual initialize/getConfig/ensureConfigForRequest. Both unsupported cookie cases fail only the final zero-dispatch assertion; the valid exact-origin single-user control passes.'}],checks,tests:{focused:{command:'bunx vitest run src/services/__tests__/server-capabilities.test.ts src/services/__tests__/tldw-api-client.quickstart-auth.test.ts src/services/__tests__/chat-surface-scope.test.ts src/services/tldw/__tests__/single-user-credential.test.ts --silent --reporter=json --outputFile=<review>/focused.json',cwd:'apps/packages/ui',exitCode:0,passed:117},diagnostic:{command:'apps/packages/ui/node_modules/.bin/vitest run --config <review>/vitest.config.mjs --silent --reporter=json --outputFile=<review>/cookie-guard.json',cwd:'.',exitCode:1,passed:1,failed:2,cases:assertions.map(a=>({name:a.fullName,status:a.status}))}},authorStaticEvidence:evidence.static,unchanged259,inputs,limits:['No production or repository test files changed; diagnostic tests exist only under this review directory.','No browser/runtime/native services/model/DB/Git/Backlog mutation or native HTTP traffic.','Background network requests and storage are mocked; the real TldwApiClient normalization and authentication readiness are exercised.','Author compiler/lint/Bandit reports were inspected; the compiler comparison was not rerun. Bandit cannot parse the TypeScript scope, so it supplies no TypeScript security assurance.','Native fresh-context and authenticated positive verification remain required after the source finding is addressed.']};
fs.writeFileSync(path.join(root,dir,'audit.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({verdict:result.verdict,checks,inputs:inputs.length,auditSha256:hash(fs.readFileSync(path.join(root,dir,'audit.json')))}));
