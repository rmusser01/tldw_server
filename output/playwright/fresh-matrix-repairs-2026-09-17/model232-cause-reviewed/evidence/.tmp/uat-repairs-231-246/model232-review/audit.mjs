import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {execFileSync} from 'node:child_process';
const dir=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(dir,'../../..');
const base='5aea3524e69179ef6f48edce333a9c477a70d71c';
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const inputs=new Map();
function read(p){const relative=path.relative(root,path.resolve(root,p));const bytes=fs.readFileSync(path.resolve(root,p));inputs.set(relative,{path:relative,bytes:bytes.length,sha256:digest(bytes)});return bytes.toString('utf8');}
const local=n=>read(path.join(dir,n));
const source=['apps/packages/ui/src/utils/chat-error-message.ts','apps/packages/ui/src/utils/__tests__/chat-error-message.test.ts','apps/packages/ui/src/services/tldw/__tests__/TldwChat.model-error.integration.test.ts'];
const sourceHashes=source.map(p=>({path:p,sha256:digest(read(p))}));
const recorded=JSON.parse(local('source-after-static.json'));
const baseline=digest(execFileSync('git',['show',`${base}:${source[0]}`],{cwd:root}));
const comparisons=JSON.parse(local('comparison.json'));
const focused=local('focused-final.log'),adjacent=local('adjacent.log'),saved=local('adjacent-saved-normal.log'),causal=local('baseline-causal.log');
const bandit=JSON.parse(local('bandit.json'));
const checks=[
 {name:'final three source hashes match final static review bytes',passed:sourceHashes.every(e=>recorded.find(r=>r.path===e.path)?.sha256===e.sha256)},
 {name:'baseline overlay matches baseline committed formatter',passed:digest(local('baseline-chat-error-message.ts'))===baseline},
 {name:'focused final run passes20',passed:/Tests\s+20 passed \(20\)/.test(focused)},
 {name:'four frontend-config adjacent suites pass84',passed:/Tests\s+84 passed \(84\)/.test(adjacent)&&/4 passed \(5\)/.test(adjacent)},
 {name:'saved-normal separately passes99 under its existing package config',passed:/Tests\s+99 passed \(99\)/.test(saved)},
 {name:'independent baseline reproduces exactly2 failures with18 controls',passed:/Tests\s+2 failed \| 18 passed \(20\)/.test(causal)},
 {name:'final scoped lint is clean',passed:comparisons.eslint.current===0},
 {name:'no new or removed compiler diagnostics',passed:comparisons.typescript.baseline===90&&comparisons.typescript.current===90&&!comparisons.typescript.added.length&&!comparisons.typescript.removed.length},
 {name:'Bandit limitations correctly identified',passed:bandit.errors.length===3&&bandit.results.length===0}
];
const nativePath='.tmp/uat-repairs-231-246/native-targeted/pg-single/model232-error-captured.txt';
const native=JSON.parse(read(nativePath).split('\n').find(l=>l.startsWith('{')));
const nativeResponse=native.events.find(e=>e.status===400&&e.url?.includes('/api/v1/chat/completions'));
checks.push({name:'native diagnostic is structured ordinary400 plus generic Health UI',passed:nativeResponse?.body?.detail?.error_code==='model_not_available'&&native.ui.includes('Health & diagnostics')});
for(const p of ['apps/packages/ui/vitest.config.ts','apps/packages/ui/package.json','apps/packages/ui/tsconfig.json','apps/tldw-frontend/vitest.config.ts','apps/tldw-frontend/tsconfig.json','apps/tldw-frontend/eslint.config.mjs','apps/packages/ui/src/utils/format-error-message.ts','apps/packages/ui/src/utils/server-error-message.ts','apps/packages/ui/src/services/tldw/TldwChat.ts','apps/packages/ui/src/services/background-proxy.ts','apps/packages/ui/src/services/tldw/domains/chat-rag.ts','backlog/tasks/task-13260.174 - Show-actionable-unavailable-model-guidance-in-Chat.md'])read(p);
for(const p of ['model232-red.txt','model232-red-v2.txt','model232-green.txt','model232-adjacent.txt'])read(`.tmp/uat-repairs-231-246/${p}`);
for(const name of ['REVIEW.md','audit.mjs','static-checks.mjs','baseline.config.ts','source-before.json','focused.log','static.log','static-current.log','eslint-baseline.json','eslint-current.json','typescript-baseline.json','typescript-current.json','bandit.log'])local(name);
const commands=[
 {cwd:'apps/packages/ui',command:'node node_modules/vitest/vitest.mjs run --maxWorkers=1 --no-file-parallelism src/utils/__tests__/chat-error-message.test.ts src/services/tldw/__tests__/TldwChat.model-error.integration.test.ts',log:'focused-final.log',exit:0},
 {cwd:'apps/tldw-frontend',command:'node node_modules/vitest/vitest.mjs run --maxWorkers=1 --no-file-parallelism ../packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx ../packages/ui/src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/services/tldw/__tests__/TldwChat.abort.test.ts ../packages/ui/src/utils/__tests__/server-error-message.test.ts',log:'adjacent.log',exit:1,limit:'84tests pass, saved-normal collection fails missing pa-tesseract alias'},
 {cwd:'apps/packages/ui',command:'node node_modules/vitest/vitest.mjs run --maxWorkers=1 --no-file-parallelism src/hooks/chat/__tests__/useChatActions.saved-normal.integration.test.tsx',log:'adjacent-saved-normal.log',exit:0},
 {cwd:'apps/tldw-frontend',command:'node node_modules/vitest/vitest.mjs run --config ../../.tmp/uat-repairs-231-246/model232-review/baseline.config.ts --maxWorkers=1 --no-file-parallelism ../packages/ui/src/utils/__tests__/chat-error-message.test.ts ../packages/ui/src/services/tldw/__tests__/TldwChat.model-error.integration.test.ts',log:'baseline-causal.log',exit:1,limit:'expected2causal failures'},
 {cwd:'.',command:'node .tmp/uat-repairs-231-246/model232-review/static-checks.mjs',log:'static.log',exit:134,limit:'default heap exhausted after baseline compiler result'},
 {cwd:'.',command:'node --max-old-space-size=8192 .tmp/uat-repairs-231-246/model232-review/static-checks.mjs current',log:'static-current.log',exit:0},
 {cwd:'.',command:'source .venv/bin/activate && PYTHONDONTWRITEBYTECODE=1 python -m bandit apps/packages/ui/src/utils/chat-error-message.ts apps/packages/ui/src/utils/__tests__/chat-error-message.test.ts apps/packages/ui/src/services/tldw/__tests__/TldwChat.model-error.integration.test.ts -f json -o .tmp/uat-repairs-231-246/model232-review/bandit.json',exit:0,limit:'3unsupportedTypeScript parse errors'},
 {cwd:'.',command:'git diff --check -- apps/packages/ui/src/utils/chat-error-message.ts apps/packages/ui/src/utils/__tests__/chat-error-message.test.ts apps/packages/ui/src/services/tldw/__tests__/TldwChat.model-error.integration.test.ts',exit:0}
];
const failed=checks.filter(c=>!c.passed);
const audit={at:new Date().toISOString(),task:'TASK-13260.174',finding:'UAT232',verdict:failed.length?'REMAINING_GAP':'CLEAR_BOUNDED_IMPLEMENTATION_NATIVE_PENDING',baselineCommit:base,baselineProductionSha256:baseline,sourceHashes,checks,commands,comparisons,
 nativeDiagnostic:{at:native.at,status:nativeResponse?.status,path:nativeResponse?new URL(nativeResponse.url).pathname:null,errorCode:nativeResponse?.body?.detail?.error_code,genericHealthGuidance:true},
 verification:{focused:20,adjacentFrontend:84,adjacentSavedNormal:99,totalSuccessful:203,successfulFiles:7,skipped:0,baseline:{expectedFailures:2,passingControls:18},bandit:{parseErrors:3,meaningfulTypeScriptCoverage:false}},
 limitations:['Native visible recovery, real Retry and canonical reload remain pending on accepted source.','HTTP behavior is a deterministic server fixture; the real service, domain adapter, direct transport parser, SSE parser and formatter execute.','Initial configuration/collection failures and compiler OOM remain retained; final green totals combine suites run under appropriate existing configs.','Compiler result is differential90/90, not a clean-build claim.','No product/source/Git/task/browser/runtime/database edits by reviewer. Only review artifacts were written.'],
 reviewedFiles:[...inputs.values()].sort((a,b)=>a.path.localeCompare(b.path))};
fs.writeFileSync(path.join(dir,'audit.json'),JSON.stringify(audit,null,2)+'\n');
console.log(JSON.stringify({verdict:audit.verdict,checksPassed:checks.length-failed.length,failed,sourceHashes,reviewedFiles:inputs.size,auditSha256:digest(fs.readFileSync(path.join(dir,'audit.json')))},null,2));
if(failed.length)process.exitCode=1;
