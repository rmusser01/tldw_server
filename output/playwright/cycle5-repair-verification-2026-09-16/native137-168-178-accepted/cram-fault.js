async page => {
  if (page.__uatCramFault) throw new Error('Cram fault already installed');
  const match = url => url.origin === 'http://127.0.0.1:18583'
    && url.pathname.replace(/\/$/, '') === '/api/v1/flashcards'
    && url.searchParams.get('deck_id') === '1'
    && url.searchParams.get('due_status') === 'all'
    && url.searchParams.get('order_by') === 'due_at'
    && url.searchParams.get('limit') === '200'
    && url.searchParams.get('offset') === '0';
  const handler = route => route.request().method() === 'GET'
    ? route.abort('failed') : route.continue();
  await page.route(match, handler);
  page.__uatCramFault = {match, handler};
  return {installed: true, at: new Date().toISOString(), target: 'Only deck1 first Cram queue page; actual observed same-origin API proxy'};
}
