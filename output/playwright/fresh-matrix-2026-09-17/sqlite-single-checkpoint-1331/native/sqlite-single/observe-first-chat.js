async page => {
  page.__firstSetupChat=[];
  page.on('request',r=>{if(r.url().includes('/api/v1/setup/first-run/first-chat')){let body;try{const v=r.postDataJSON();body={prompt:v.prompt};}catch{}page.__firstSetupChat.push({at:new Date().toISOString(),kind:'request',url:r.url(),method:r.method(),body});}});
  page.on('response',async r=>{if(r.url().includes('/api/v1/setup/first-run/first-chat')){const at=new Date().toISOString();let body;try{const v=await r.json();body=Object.fromEntries(['status','provider','model','response_text','elapsed_ms','latency_ms','error','message'].filter(k=>k in v).map(k=>[k,v[k]]));}catch{}page.__firstSetupChat.push({at,readAt:new Date().toISOString(),kind:'response',status:r.status(),url:r.url(),body});}});
  return {at:new Date().toISOString(),observing:'wizard first-chat only; selected noncredential body fields'};
}
