async (page) => {
  page.__uat230Events = [];
  if (!page.__uat230Watching) {
    page.__uat230Watching = true;
    const relevant = (url) => /\/openapi\.json(?:\?|$)|\/billing\/(?:plans|subscription|usage|invoices)(?:\?|$)/.test(url);
    page.on('request', request => { if (relevant(request.url())) page.__uat230Events.push({at:new Date().toISOString(),kind:'request',method:request.method(),url:request.url()}); });
    page.on('response', async response => {
      if (relevant(response.url())) page.__uat230Events.push({at:new Date().toISOString(),kind:'response',status:response.status(),url:response.url()});
      if (/\/auth\/me(?:\?|$)/.test(response.url()) && response.status() === 200) {
        try { const body=await response.json(); page.__uat230Events.push({at:new Date().toISOString(),kind:'identity',id:body.id,username:body.username}); } catch {}
      }
    });
    page.on('pageerror', error => page.__uat230Events.push({at:new Date().toISOString(),kind:'pageerror',message:error.message}));
  }
  return {at:new Date().toISOString(),url:page.url(),observing:'OpenAPI and Billing URL/status only; auth/me ID/username; page errors'};
}
