import fs from 'node:fs';import path from 'node:path';import crypto from 'node:crypto';
const out='.tmp/uat-repairs-231-246/review-media',src='.tmp/uat-repairs-231-246/media237-241-244';
const read=p=>JSON.parse(fs.readFileSync(p)),sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const files=read(src+'/source-freeze.json').files.map(f=>({...f,verified:sha(fs.readFileSync(f.path))===f.sha256}));if(files.some(f=>!f.verified))throw Error('Source changed');
const delta=(before,after)=>{const c=new Map();for(const x of before)c.set(x,(c.get(x)||0)+1);return after.filter(x=>{const n=c.get(x)||0;if(n){c.set(x,n-1);return false}return true})};
const t0=read(out+'/tsc-fresh-baseline.json').map(x=>JSON.stringify(x)),t1=read(out+'/tsc-fresh-current.json').map(x=>JSON.stringify(x));
const lint=p=>read(p).flatMap(f=>f.messages.map(m=>JSON.stringify({file:f.filePath,ruleId:m.ruleId,severity:m.severity,message:m.message.replace(/\(at line \d+\)/g,'(at line <moved>)')})));
const l0=lint(out+'/eslint-baseline.json'),l1=lint(out+'/eslint-current.json');
const comparison={compiler:{baseline:t0.length,current:t1.length,added:delta(t0,t1),removed:delta(t1,t0)},lint:{baseline:l0.length,current:l1.length,added:delta(l0,l1),removed:delta(l1,l0),normalization:'Only embedded (at line N) positions normalized; signatures retain file, rule, severity, text and multiplicity.'},bandit:{findings:read(out+'/bandit.json').results.length,parseLimitations:read(out+'/bandit.json').errors.length}};
if(comparison.compiler.added.length||comparison.compiler.removed.length||comparison.lint.added.length)throw Error('New diagnostic');
fs.writeFileSync(out+'/source-after.json',JSON.stringify({at:new Date().toISOString(),files},null,2)+'\n');fs.writeFileSync(out+'/static-comparison.json',JSON.stringify(comparison,null,2)+'\n');
fs.writeFileSync(out+'/REVIEW.md',`# Independent Media237/241/244 review

**CLEAR for integration; original native acceptance remains pending.**

Independent final193 Media/Quick Ingest tests across15suites and37 actual Form/action consumer tests pass, zero skips. Twelve frozen source/test hashes match. Actual-root ESLint has0errors and151warnings versus154baseline, with zero new signatures/multiplicity after normalizing embedded moved-line references only. Fresh compiler programs have90 identical baseline/current diagnostics. Bandit ran on all12TS/TSX paths:0findings but12parse limitations, so it supplies no TypeScript security assurance.

Reviewed initial gate, real QueryClient lifetime/cache behavior, abort propagation through search/type/keyword/detail requests, current-operation ingest completion and retained filter/page state. Full-content handoff requires loaded content and current owner/selection; full-content and RAG producers reuse the existing owned payload contract and wait for storage before navigation. Existing normal/RAG semantics and real downstream consumer controls remain covered.

Review found an unfenced stale-selection interval. Two causal REDs established a late deletion warning after synchronous authority replacement and selection replacement after delayed deletion refresh. The correction uses the existing lifetime and current-selection checks before dispatch and after awaits; final193 includes both regressions and original deletion recovery positives. The intermediate overguard failure is retained.

No browser/profile/provider/database was changed by review. Component checks do not establish native workflow acceptance. Full project compiler/lint are not clean at baseline; no new diagnostics are introduced by this unit.
`);
const entries=fs.readdirSync(out).filter(n=>n!=='reviewer-manifest.json').map(p=>{const b=fs.readFileSync(path.join(out,p));return {path:p,bytes:b.length,sha256:sha(b)}});
fs.writeFileSync(out+'/reviewer-manifest.json',JSON.stringify({tasks:['TASK13260.179','TASK13260.183','TASK13260.186'],files:entries},null,2)+'\n');
console.log(JSON.stringify({sourceHashes:files.length,compiler:comparison.compiler,newLint:comparison.lint.added.length,reviewSha:sha(fs.readFileSync(out+'/REVIEW.md')),manifestSha:sha(fs.readFileSync(out+'/reviewer-manifest.json'))}));
