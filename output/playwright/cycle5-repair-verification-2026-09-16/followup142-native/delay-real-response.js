async page => {
  page.__uat142={stage:'installed',url:page.url()};
  await page.route('**/api/v1/notifications/mark-read',async route=>{
    const response=await route.fetch();
    page.__uat142={stage:'backend-response-held',status:response.status(),body:await response.text(),requestBody:route.request().postData(),originalUrl:page.url()};
    await new Promise(resolve=>{page.__uat142Release=resolve;});
    await route.fulfill({response});
    page.__uat142.stage='released';
  });
  return page.__uat142;
}
