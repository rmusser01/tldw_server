async page => {
  let release, resolveHeld, resolveFinished;
  const held=new Promise(resolve=>{resolveHeld=resolve;});
  const finished=new Promise(resolve=>{resolveFinished=resolve;});
  const result={started:Date.now(),startUrl:page.url()};
  await page.route('**/api/v1/notifications/mark-read',async route=>{
    const response=await route.fetch();
    result.backendStatus=response.status();result.backendBody=await response.text();result.requestBody=route.request().postData();result.heldAt=Date.now();
    resolveHeld(); await new Promise(resolve=>{release=resolve;});
    try{await route.fulfill({response});result.delivery='fulfilled';}catch(error){result.delivery='browser-request-already-settled';result.deliveryError=String(error.message);}
    result.releasedAt=Date.now();resolveFinished();
  });
  const row=page.getByRole('listitem').filter({has:page.getByRole('heading',{name:'UAT142 delayed View ownership',exact:true})});
  await row.getByRole('button',{name:'View',exact:true}).click();
  await Promise.race([held,new Promise((_,reject)=>setTimeout(()=>reject(Error('No real response within3s')),3000))]);
  await page.getByRole('button',{name:'Notes',exact:true}).click();
  await page.waitForURL('**/notes');result.navigatedUrl=page.url();result.navigatedAt=Date.now();
  release();await finished;await page.waitForTimeout(500);result.finalUrl=page.url();result.finished=Date.now();
  await page.unroute('**/api/v1/notifications/mark-read');
  await page.screenshot({path:'.tmp/uat142-native-20260916/after-late-response.png'});
  return result;
}
