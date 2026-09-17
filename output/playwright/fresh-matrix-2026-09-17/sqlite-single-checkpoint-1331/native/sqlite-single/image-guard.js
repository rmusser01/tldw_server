async page => {
 const before=(page.__matrixEvents||[]).filter(e=>e.event==='request'&&e.url.endsWith('/chat/completions')).length;
 await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).fill('Describe this attached public test icon in one sentence.');
 await page.getByRole('button',{name:'Send message',exact:true}).click();
 return {at:new Date().toISOString(),beforeCompletionRequests:before,body:await page.locator('body').innerText()};
}
