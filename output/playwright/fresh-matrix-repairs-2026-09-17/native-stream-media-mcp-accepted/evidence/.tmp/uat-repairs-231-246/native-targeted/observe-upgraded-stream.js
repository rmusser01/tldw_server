async page=>{
  const origin=page.url().match(/^http:\/\/127\.0\.0\.1:1878[23](?=\/|$)/)?.[0];
  if(!origin)throw Error('Expected targeted acceptance origin');
  page.__repairStreamHeaders=[];
  page.on('response',async r=>{
    if(!r.url().startsWith(origin+'/api/v1/chats/')||!r.url().split('?')[0].endsWith('/complete-v2'))return;
    page.__repairStreamHeaders.push({at:new Date().toISOString(),status:r.status(),contentType:await r.headerValue('content-type'),contentEncoding:await r.headerValue('content-encoding'),transferEncoding:await r.headerValue('transfer-encoding'),vary:await r.headerValue('vary'),cacheControl:await r.headerValue('cache-control')});
  });
  return {installed:true,origin,fields:['status','content-type','content-encoding','transfer-encoding','vary','cache-control'],noAuthHeadersOrBody:true};
}
