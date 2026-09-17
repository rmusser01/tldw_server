async page=>{
 await page.goto('http://127.0.0.1:18683/chat');await page.getByRole('button',{name:'Send message',exact:true}).waitFor({timeout:15000});
 const before=await page.locator('body').ariaSnapshot();page.__chatStart=new Date().toISOString();
 await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).fill('Remember the code ORBIT-742 for this conversation. Reply with only ORBIT-742.');await page.getByRole('button',{name:'Send message',exact:true}).click();
 await page.getByRole('article',{name:/Assistant message/}).first().getByText('ORBIT-742',{exact:true}).waitFor({timeout:20000});
 await page.getByRole('button',{name:'Stop generation',exact:true}).waitFor({state:'hidden',timeout:10000});
 const first=await page.locator('body').ariaSnapshot();await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).fill('What code did I ask you to remember? Reply with the code only.');await page.getByRole('button',{name:'Send message',exact:true}).click();
 await page.getByRole('article',{name:/Assistant message/}).nth(1).getByText('ORBIT-742',{exact:true}).waitFor({timeout:20000});
 await page.getByRole('button',{name:'Stop generation',exact:true}).waitFor({state:'hidden',timeout:10000});
 return {at:new Date().toISOString(),before,first,second:await page.locator('body').ariaSnapshot(),events:(page.__matrixEvents||[]).filter(e=>e.at>page.__chatStart&&/\/chat\/completions$|\/messages\?/.test(e.url||''))};
}
