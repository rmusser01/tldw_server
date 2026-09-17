async page => {
  if (page.__uatReviewedObservers) {
    for (const [event, listener] of page.__uatReviewedObservers) page.off(event, listener);
  }
  page.__uatReviewedEvents = [];
  const keep = url => /\/api\/v1\/(?:persona|visual-identities|chats?|characters|notes|flashcards|study[_-]packs|buddy|keywords|collections|workspaces)(?:\/|\?|$)/.test(url);
  const request = r => {
    if (keep(r.url())) page.__uatReviewedEvents.push({event:'request', at:new Date().toISOString(), url:r.url(), method:r.method(), body:r.postData()});
  };
  const response = async r => {
    const at = new Date().toISOString();
    if (keep(r.url())) {
      let body; try { body = await r.text(); } catch {}
      page.__uatReviewedEvents.push({event:'response', at, bodyReadAt:new Date().toISOString(), url:r.url(), status:r.status(), body});
    } else if (/\/api\/v1\/auth\/me(?:\?|$)/.test(r.url())) {
      let identity; try { const v = await r.json(); identity = {id:v.id, username:v.username}; } catch {}
      page.__uatReviewedEvents.push({event:'identity', at, status:r.status(), identity});
    }
  };
  const pageerror = e => page.__uatReviewedEvents.push({event:'pageerror', at:new Date().toISOString(), message:String(e)});
  page.__uatReviewedObservers = [['request',request],['response',response],['pageerror',pageerror]];
  for (const [event,listener] of page.__uatReviewedObservers) page.on(event,listener);
  return {at:new Date().toISOString(), installed:true, url:page.url()};
}
