import fs from 'node:fs';import {execFileSync} from 'node:child_process';
const phase=process.argv[2];if(!/^[a-z-]+$/.test(phase))throw new Error('Invalid phase');
const prefix='/private/tmp/cycle4-native124-final-'+phase;
const code=`async (page) => {
  await page.screenshot({path: ${JSON.stringify(prefix+'.png')},fullPage:true});
  return await page.locator('article').evaluateAll(async nodes => Promise.all(nodes.map(async node => ({
    label:node.getAttribute('aria-label'),id:node.getAttribute('data-message-id'),role:node.getAttribute('data-role'),text:node.innerText,
    images:await Promise.all([...node.querySelectorAll('img')].map(async image=>{
      const src=image.getAttribute('src')||'';const bytes=src.startsWith('data:')?Uint8Array.from(atob(src.split(',')[1]),c=>c.charCodeAt(0)):new Uint8Array();
      return {alt:image.alt,mime:src.split(';')[0],bytes:bytes.length,sha256:[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('')};
    }))
  }))));
}`;
for(const [command,args,suffix]of [['run-code',[code],'-identity.json'],['snapshot',[],'-snapshot.txt'],['requests',[],'-requests.txt']]){fs.writeFileSync(prefix+suffix,execFileSync('node',['/private/tmp/cycle4-targeted2-safe-browser.mjs','single',command,...args],{encoding:'utf8',maxBuffer:8*1024*1024}));}
console.log(JSON.stringify({phase,prefix}));
