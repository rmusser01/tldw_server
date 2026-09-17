async page => {
 page.__uat031Events=[];
 const keep=url=>/\/api\/v1\/(?:chats?|characters|notes|flashcards)(?:\/|\?|$)/.test(url);
 page.on('request',r=>{if(keep(r.url()))page.__uat031Events.push({event:'request',at:new Date().toISOString(),url:r.url(),method:r.method(),body:r.postData()})});
 page.on('response',async r=>{if(!keep(r.url()))return;let body;try{body=await r.text()}catch{}page.__uat031Events.push({event:'response',at:new Date().toISOString(),url:r.url(),status:r.status(),body})});
 page.on('pageerror',e=>page.__uat031Events.push({event:'pageerror',at:new Date().toISOString(),message:String(e)}));
 return {at:new Date().toISOString(),installed:true,url:page.url()};
}
