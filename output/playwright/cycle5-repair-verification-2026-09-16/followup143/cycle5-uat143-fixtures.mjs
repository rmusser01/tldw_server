import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
const root='/Users/macbook-dev/Documents/GitHub/tldw_server2'
const dir=root+'/apps/packages/ui/src/components/Option/Playground/__tests__'
const names=['composer-options.guard.test.ts','llamacpp-controls.guard.test.ts','document-processing.test.tsx','follow-up-research.test.tsx','image-refine.integration.test.tsx','voice-visibility.integration.test.tsx'].map(n=>'PlaygroundForm.'+n)
const before='/private/tmp/cycle5-uat143-before'
fs.mkdirSync(before,{recursive:true})
const rows=names.map(n=>{const b=fs.readFileSync(dir+'/'+n);fs.writeFileSync(before+'/'+n,b);return {path:path.relative(root,dir+'/'+n),sha256:crypto.createHash('sha256').update(b).digest('hex')}})
fs.writeFileSync('/private/tmp/cycle5-uat143-before.json',JSON.stringify(rows,null,2)+'\n')
const edit=(name,old,next)=>{const f=dir+'/PlaygroundForm.'+name,s=fs.readFileSync(f,'utf8');if(!s.includes(old))throw new Error(name+': missing target');fs.writeFileSync(f,s.replace(old,next))}
edit('document-processing.test.tsx','  form: makeForm(),','  form: makeForm(),\n  beginPromptAssistReset: vi.fn(() => 1),\n  markPromptAssistAttemptSaved: vi.fn(),')
const query=`  useQuery: ({ queryKey }: { queryKey: readonly unknown[] }) => ({
    // Prompt capabilities are an object; list-query fixtures remain empty arrays.
    data: queryKey[0] === "promptCapabilities"
      ? {
          availability: "unavailable",
          prompt_improvement_v1: { supported: false, limits: null },
          single_text_recipe_v2: { supported: false }
        }
      : []
  }),`
for (const n of ['follow-up-research.test.tsx','image-refine.integration.test.tsx','voice-visibility.integration.test.tsx']) edit(n,'  useQuery: () => ({ data: [] }),',query)
edit('voice-visibility.integration.test.tsx','    typing: false,','    messageRevision: 0,\n    promptAssistMutation: { revision: 0, source: "owner" },\n    promptAssistSavedAttemptId: null,\n    beginPromptAssistReset: vi.fn(() => 1),\n    markPromptAssistAttemptSaved: vi.fn(),\n    typing: false,')
edit('composer-options.guard.test.ts','    expect(source).toContain("col-span-2 flex shrink-0 justify-end self-end")',`    const mobileSendClasses = source.match(
      /data-testid="composer-inline-send-control"\\s+className=\\{\\s*isMobileViewport\\s*\\? "([^"]+)"/
    )?.[1].split(/\\s+/)
    expect(mobileSendClasses).toEqual(
      expect.arrayContaining(["col-span-2", "flex", "shrink-0", "justify-end", "self-end"])
    )`)
