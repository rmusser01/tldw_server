async page => {
 page.__uat141Events=[];page.__uat141Phase='cold-aborted';
 page.__uat141Route=async route=>{page.__uat141Events.push({kind:'aborted',phase:page.__uat141Phase,url:route.request().url(),at:Date.now()});await route.abort('failed')};
 page.on('response',async r=>{if(!r.url().includes('/api/v1/notifications'))return;let body;try{if(!r.url().includes('/stream'))body=await r.text()}catch{}page.__uat141Events.push({kind:'response',phase:page.__uat141Phase,url:r.url(),status:r.status(),body,at:Date.now()})});
 await page.route('**/api/v1/notifications**',page.__uat141Route);
 await page.goto('http://127.0.0.1:18580/notifications');return {phase:page.__uat141Phase,url:page.url()};
}
