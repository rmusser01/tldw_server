async page => {
 await page.getByRole('button',{name:'Retry same model',exact:true}).click();
 await page.getByRole('button',{name:'Send message',exact:true}).waitFor({state:'visible',timeout:45000});
 return {at:new Date().toISOString(),fault:page.__controlledFailure,events:page.__matrixEvents,body:await page.locator('body').innerText()};
}
