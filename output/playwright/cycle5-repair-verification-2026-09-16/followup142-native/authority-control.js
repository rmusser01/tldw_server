async page => {
  const settings=page.context().pages().find(p=>p.url()==='http://127.0.0.1:18580/settings/tldw');
  if(!settings)throw Error('Prepared real Settings tab is missing');
  let release,resolveHeld,resolveFinished;
  const held=new Promise(resolve=>{resolveHeld=resolve;});
  const finished=new Promise(resolve=>{resolveFinished=resolve;});
  const result={started:Date.now(),startUrl:page.url(),settingsUrl:settings.url()};
  await page.route('**/api/v1/notifications/mark-read',async route=>{
    const response=await route.fetch();
    result.backendStatus=response.status();result.backendBody=await response.text();result.requestBody=route.request().postData();result.heldAt=Date.now();
    resolveHeld();await new Promise(resolve=>{release=resolve;});
    try{await route.fulfill({response});result.delivery='fulfilled';}catch(error){result.delivery='browser-request-already-settled';result.deliveryError=String(error.message);}
    result.releasedAt=Date.now();resolveFinished();
  });
  await page.getByRole('listitem').filter({has:page.getByRole('heading',{name:'UAT142 disconnect during View',exact:true})}).getByRole('button',{name:'View',exact:true}).click();
  await held;
  await settings.bringToFront();await settings.getByRole('button',{name:'Disconnect',exact:true}).click();
  result.disconnectedAt=Date.now();
  await page.waitForTimeout(500);
  result.beforeReleaseUrl=page.url();result.beforeReleaseText=await page.locator('body').innerText();
  release();await finished;await page.waitForTimeout(500);
  result.finalUrl=page.url();result.finalText=await page.locator('body').innerText();result.finished=Date.now();
  await page.unroute('**/api/v1/notifications/mark-read');
  await page.bringToFront();await page.screenshot({path:'.tmp/uat142-native-20260916/authority-after-late-response.png'});
  return result;
}
