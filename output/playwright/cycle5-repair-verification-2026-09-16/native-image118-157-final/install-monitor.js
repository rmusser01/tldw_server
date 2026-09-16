async (page) => {
  page.__finalImageEvents = [];
  let seq = 0;
  const ids = new Map();
  const matches = r => /\/api\/v1\/(chat|chats)(\/|\?)/.test(r.url());
  page.on('request', r => {
    if (!matches(r)) return;
    const id = ++seq;
    ids.set(r,id);
    page.__finalImageEvents.push({event:'request',id,method:r.method(),url:r.url(),body:r.postData(),at:Date.now()});
  });
  page.on('response', async r => {
    const id = ids.get(r.request());
    if (!id) return;
    page.__finalImageEvents.push({event:'response',id,status:r.status(),at:Date.now()});
    try {
      const body = await r.text();
      page.__finalImageEvents.push({event:'body',id,body,at:Date.now()});
    } catch (e) {
      page.__finalImageEvents.push({event:'body-unavailable',id,error:String(e),at:Date.now()});
    }
  });
  page.on('requestfinished',r=>{const id=ids.get(r);if(id)page.__finalImageEvents.push({event:'finished',id,at:Date.now()});});
  page.on('requestfailed',r=>{const id=ids.get(r);if(id)page.__finalImageEvents.push({event:'failed',id,failure:r.failure(),at:Date.now()});});
  return {monitorInstalled:true,url:page.url()};
}
