import fs from 'node:fs'
import path from 'node:path'
import {createRequire} from 'node:module'
const repo=process.cwd(), packet=path.join(repo,'.tmp/uat-repairs-231-246/chat231-232-236-243'), frontend=path.join(repo,'apps/tldw-frontend'), ui=path.join(repo,'apps/packages/ui/src')
const {production,tests}=JSON.parse(fs.readFileSync(path.join(packet,'owned-paths.json'),'utf8'))
const require=createRequire(path.join(frontend,'package.json')), {ESLint}=require('eslint'), ts=require('typescript')
const config=(await import(path.join(frontend,'eslint.config.mjs'))).default
const lint=new ESLint({cwd:repo,overrideConfigFile:true,overrideConfig:[...config,{settings:{next:{rootDir:frontend}}}]})
const write=(name,data)=>fs.writeFileSync(path.join(packet,name),JSON.stringify(data,null,2)+'\n')
const normalized={}
for(const version of ['baseline','current']) {
  const results=[]
  for(const file of [...production,...tests]) {
    const source=path.join(version==='baseline'?path.join(packet,'baseline'):ui,file)
    if(!fs.existsSync(source)) continue
    results.push(...await lint.lintText(fs.readFileSync(source,'utf8'),{filePath:path.join(ui,file)}))
  }
  write(`eslint-${version}.json`,results)
  normalized[version]=results.flatMap(r=>r.messages.map(m=>JSON.stringify({file:path.relative(ui,r.filePath),rule:m.ruleId,severity:m.severity,message:m.message}))).sort()
}
write('eslint-comparison.json',{baseline:normalized.baseline.length,current:normalized.current.length,added:normalized.current.filter(x=>!normalized.baseline.includes(x)).map(JSON.parse),removed:normalized.baseline.filter(x=>!normalized.current.includes(x)).map(JSON.parse)})
const raw=ts.readConfigFile(path.join(frontend,'tsconfig.json'),ts.sys.readFile), parsed=ts.parseJsonConfigFileContent(raw.config,ts.sys,frontend), options={...parsed.options,noEmit:true,incremental:false}
for(const version of ['baseline','current']) {
  const host=ts.createCompilerHost(options,true), read=host.readFile
  host.readFile=file=>version==='baseline'&&production.includes(path.relative(ui,file))?fs.readFileSync(path.join(packet,'baseline',path.relative(ui,file)),'utf8'):read(file)
  const program=ts.createProgram(parsed.fileNames,options,host)
  const diagnostics=ts.getPreEmitDiagnostics(program).map(d=>({file:d.file?path.relative(repo,d.file.fileName):null,code:d.code,message:ts.flattenDiagnosticMessageText(d.messageText,'\n'),location:d.file&&d.start!=null?d.file.getLineAndCharacterOfPosition(d.start):null}))
  write(`tsc-${version}.json`,diagnostics)
  normalized[version]=diagnostics.map(({location,...d})=>JSON.stringify(d)).sort()
}
write('tsc-comparison.json',{baseline:normalized.baseline.length,current:normalized.current.length,added:normalized.current.filter(x=>!normalized.baseline.includes(x)).map(JSON.parse),removed:normalized.baseline.filter(x=>!normalized.current.includes(x)).map(JSON.parse)})
console.log(fs.readFileSync(path.join(packet,'tsc-comparison.json'),'utf8'))
