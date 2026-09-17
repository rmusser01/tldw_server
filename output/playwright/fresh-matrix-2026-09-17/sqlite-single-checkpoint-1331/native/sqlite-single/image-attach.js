async page => {
 await page.getByRole('button',{name:'New saved chat',exact:true}).click();
 const chooserPromise=page.waitForEvent('filechooser');
 await page.getByRole('button',{name:'Attach image',exact:true}).click();
 const chooser=await chooserPromise;
 await chooser.setFiles('/Users/macbook-dev/Documents/GitHub/tldw_server2/.tmp/uat-next-matrix-20260916/sources/sqlite-single/apps/tldw-frontend/public/icon/128.png');
 return {at:new Date().toISOString(),body:await page.locator('body').innerText()};
}
