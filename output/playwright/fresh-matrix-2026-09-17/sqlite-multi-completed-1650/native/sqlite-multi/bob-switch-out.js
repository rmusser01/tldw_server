async(page)=>{
 const draft={url:page.url(),inputs:await page.locator('textarea,input:not([type=password])').evaluateAll(es=>es.map(e=>({label:e.getAttribute('aria-label'),placeholder:e.getAttribute('placeholder'),value:e.value}))),body:await page.locator('body').innerText()};
 await page.goto('http://127.0.0.1:18681/settings/tldw');
 await page.getByRole('button',{name:'Logout',exact:true}).click();
 await page.getByRole('button',{name:'Logout',exact:true}).waitFor({state:'hidden',timeout:15000});
 await page.goto('http://127.0.0.1:18681/login');
 await page.getByRole('heading',{name:'Sign in to Tldw',exact:true}).waitFor({timeout:10000});
 return {draft,after:{url:page.url(),body:await page.locator('body').innerText()},at:new Date().toISOString()};
}
