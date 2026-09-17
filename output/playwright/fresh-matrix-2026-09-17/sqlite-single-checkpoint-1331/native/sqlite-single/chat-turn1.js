async page => {
 await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).fill('Remember this public UAT fact: the observatory code is ORBIT-742. Reply with the code only.');
 await page.getByRole('button',{name:'Send message',exact:true}).click();
 return {at:new Date().toISOString(),url:page.url(),body:await page.locator('body').innerText()};
}
