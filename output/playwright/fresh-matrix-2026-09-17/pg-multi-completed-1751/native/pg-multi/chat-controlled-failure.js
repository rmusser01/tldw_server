async page => {
 page.__controlledFailure=[];
 await page.route('**/api/v1/chat/completions',async route=>{
  const req=route.request();const original=req.postDataJSON();
  const altered={...original,api_provider:'ollama',model:'uat-deliberately-unavailable-20260917'};
  page.__controlledFailure.push({at:new Date().toISOString(),url:req.url(),originalProvider:original.api_provider,originalModel:original.model,effectiveProvider:altered.api_provider,effectiveModel:altered.model,conversationId:original.conversation_id,clientMessageId:original.metadata?.tldw_client_message_id,mode:'one real backend request continued; no response fulfillment'});
  await route.continue({postData:JSON.stringify(altered)});
 },{times:1});
 await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).fill('Repeat the observatory code from our conversation. Reply with that code only.');
 await page.getByRole('button',{name:'Send message',exact:true}).click();
 await page.getByRole('button',{name:'Retry same model',exact:true}).waitFor({state:'visible',timeout:20000});
 return {at:new Date().toISOString(),fault:page.__controlledFailure,body:await page.locator('body').innerText()};
}
