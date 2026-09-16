import fs from 'node:fs'
import base from '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/vitest.config'
const ui='/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui'
const target=ui+'/src/components/Flashcards/tabs/__tests__/ManageTab.empty-state.test.tsx'
const capture=fs.readFileSync('/private/tmp/uat-cycle4-single-create-button-semantics.json','utf8')
const nativeDom=JSON.parse(capture.split('\n').find(line=>line.startsWith('{'))!).html
export default {...base,plugins:[{name:'uat112-real-owner-and-native-dom',enforce:'pre',transform(code,id){
 if(id!==target)return
 const insert=code.lastIndexOf('\n})')
 if(insert<0)throw new Error('Expected fixture describe ending')
 const cases=`
 it("UAT112 actual Manage FAB has an accessible name", () => {
 const view=renderManageTab();
 try { expect(screen.getByTestId("flashcards-fab-create")).toHaveAccessibleName("Create card"); }
 finally {view.unmount();}
 });
 it("UAT112 native retained hidden icon does not rename completed Create", () => {
 const host=document.createElement("div"); host.innerHTML=${JSON.stringify(nativeDom)}; document.body.append(host);
 try { const button=host.querySelector("button")!;
 expect(button).toBeEnabled();
 expect(host.querySelector(".ant-btn-loading-icon")).toHaveStyle({opacity:"0",width:"0px"});
 expect(button).toHaveAccessibleName("Create");
 } finally {host.remove();}
 });
 `
 return{code:code.slice(0,insert)+cases+code.slice(insert),map:null}
 }}],test:{...base.test,setupFiles:[ui+'/vitest.setup.ts'],include:[target]}}
