async page => {
  const receipt={started:Date.now(),startUrl:page.url()};
  const responsePromise=page.waitForResponse(r=>r.url().endsWith('/notifications/mark-read')&&r.request().method()==='POST');
  await page.getByRole('listitem').filter({has:page.getByRole('heading',{name:'UAT142 current authority View',exact:true})}).getByRole('button',{name:'View',exact:true}).click();
  const response=await responsePromise;
  receipt.status=response.status();receipt.request=response.request().postData();receipt.body=await response.text();
  await page.waitForURL('**/prompts');receipt.finalUrl=page.url();receipt.finished=Date.now();
  return receipt;
}
