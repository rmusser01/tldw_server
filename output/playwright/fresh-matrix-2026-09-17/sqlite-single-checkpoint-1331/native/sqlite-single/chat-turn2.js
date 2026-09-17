async page => {
 await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).fill('What observatory code did I give you? Reply with that code only.');
 await page.getByRole('button',{name:'Send message',exact:true}).click();
 await page.getByRole('button',{name:'Send message',exact:true}).waitFor({state:'visible',timeout:45000});
 return {at:new Date().toISOString(),url:page.url(),body:await page.locator('body').innerText()};
}
