async page => {
  const context=page.context(); context.__uat114=[];context.__uat114counter=0;context.__uat114ids=new WeakMap();
  context.on('request', request=>{if(!request.url().includes('/api/'))return;const id=++context.__uat114counter;context.__uat114ids.set(request,id);let owner;try{owner=context.pages().indexOf(request.frame().page());}catch{owner=-1;}context.__uat114.push({id,owner,event:'start',at:Date.now(),url:request.url(),method:request.method()});});
  context.on('response',response=>{const id=context.__uat114ids.get(response.request());if(id)context.__uat114.push({id,event:'response',at:Date.now(),status:response.status()});});
  context.on('requestfinished',request=>{const id=context.__uat114ids.get(request);if(id)context.__uat114.push({id,event:'finished',at:Date.now()});});
  context.on('requestfailed',request=>{const id=context.__uat114ids.get(request);if(id)context.__uat114.push({id,event:'failed',at:Date.now(),error:request.failure()?.errorText});});
  await page.reload(); return {monitorInstalled:true};
}
