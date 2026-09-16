async page => {
  page.__abortedSourceRetrievals=[];
  await page.route('**/api/v1/rag/search*',async route=>{
    page.__abortedSourceRetrievals.push({url:route.request().url(),method:route.request().method(),body:route.request().postData(),at:Date.now()});
    await route.abort('failed');
  });
  await page.getByRole('button',{name:'Send message',exact:true}).click();
  return {sent:true,controlledFailure:'Only the selected-source RAG transport is aborted; no response body is fabricated.',at:Date.now()};
}
