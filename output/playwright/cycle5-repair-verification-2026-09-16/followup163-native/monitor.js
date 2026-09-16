async page => {
  page.__uat163Events=[];
  let seq=0;const ids=new Map();
  page.on('request',r=>{if(!/\/api\/v1\/(chat|chats|rag|media)(\/|\?)/.test(r.url()))return;const id=++seq;ids.set(r,id);page.__uat163Events.push({event:'request',id,url:r.url(),method:r.method(),body:r.postData(),at:Date.now()});});
  page.on('response',async r=>{const id=ids.get(r.request());if(!id)return;page.__uat163Events.push({event:'response',id,status:r.status(),at:Date.now()});try{page.__uat163Events.push({event:'body',id,body:await r.text(),at:Date.now()});}catch(e){page.__uat163Events.push({event:'body-unavailable',id,error:String(e),at:Date.now()});}});
  page.on('requestfinished',r=>{const id=ids.get(r);if(id)page.__uat163Events.push({event:'finished',id,at:Date.now()});});
  page.on('requestfailed',r=>{const id=ids.get(r);if(id)page.__uat163Events.push({event:'failed',id,failure:r.failure(),at:Date.now()});});
  return {installed:true,url:page.url()};
}
