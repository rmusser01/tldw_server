import fs from 'node:fs'
const dir='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/__tests__'
for(const suffix of ['follow-up-research.test.tsx','image-refine.integration.test.tsx','voice-visibility.integration.test.tsx']){
 const name='PlaygroundForm.'+suffix
 const original=fs.readFileSync('/private/tmp/cycle5-uat143-before/'+name,'utf8')
 const marker='vi.mock("@tanstack/react-query", () => ({'
 const stub='// Prompt Assist has its own lifecycle suites; this fixture exercises the Form flow.\nvi.mock("@/components/Chat/composer/PromptAssistComposerAction", () => ({\n  PromptAssistComposerAction: () => null\n}))\n\n'
 if(!original.includes(marker)) throw new Error(name)
 fs.writeFileSync(dir+'/'+name,original.replace(marker,stub+marker))
}
