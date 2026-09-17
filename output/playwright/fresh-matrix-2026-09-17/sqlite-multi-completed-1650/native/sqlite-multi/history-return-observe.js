async(page)=>{
 const states=[];
 await page.goBack();
 await page.getByRole('heading',{name:'Sign in to tldw',exact:true}).waitFor({timeout:10000});
 states.push({step:'first Back after normal Alice login',url:page.url(),body:await page.locator('body').innerText()});
 await page.goBack();
 await page.getByRole('button',{name:'Logout',exact:true}).waitFor({timeout:15000});
 states.push({step:'Back to settings',url:page.url(),body:await page.locator('body').innerText()});
 await page.goBack();
 await page.getByRole('textbox',{name:'New deck name',exact:true}).first().waitFor({timeout:15000});
 states.push({step:'Back to previous Bob generator under Alice',url:page.url(),body:await page.locator('body').innerText(),inputs:await page.locator('textarea,input:not([type=password])').evaluateAll(es=>es.map(e=>({label:e.getAttribute('aria-label'),placeholder:e.getAttribute('placeholder'),value:e.value})))});
 return {states,at:new Date().toISOString(),events:(page.__matrixEvents||[]).slice(-35)};
}
