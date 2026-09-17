async page => {
  if (page.__repairCharacterCdp) await page.__repairCharacterCdp.detach();
  const origin = page.url().match(/^http:\/\/127\.0\.0\.1:1878[23](?=\/|$)/)?.[0] || '';
  if (!/^http:\/\/127\.0\.0\.1:1878[23]$/.test(origin)) throw new Error('Expected owned targeted acceptance origin');
  const session = await page.context().newCDPSession(page);
  const ids = new Set();
  page.__repairCharacterTiming = [];
  const record = value => page.__repairCharacterTiming.push({ observedAt: new Date().toISOString(), ...value });
  session.on('Network.requestWillBeSent', event => {
    if (!event.request.url.startsWith(origin + '/')) return;
    const pathname = event.request.url.slice(origin.length).split(/[?#]/)[0];
    if (!/\/complete-v2\/?$/.test(pathname) || event.request.method !== 'POST') return;
    ids.add(event.requestId);
    record({ event: 'request', requestId: event.requestId, path: pathname, timestamp: event.timestamp, wallTime: event.wallTime });
  });
  session.on('Network.responseReceived', event => {
    if (ids.has(event.requestId)) record({ event: 'headers', requestId: event.requestId, timestamp: event.timestamp, status: event.response.status });
  });
  session.on('Network.dataReceived', event => {
    if (ids.has(event.requestId)) record({ event: 'body-bytes', requestId: event.requestId, timestamp: event.timestamp, dataLength: event.dataLength, encodedDataLength: event.encodedDataLength });
  });
  session.on('Network.loadingFinished', event => {
    if (ids.has(event.requestId)) record({ event: 'finished', requestId: event.requestId, timestamp: event.timestamp, encodedDataLength: event.encodedDataLength });
  });
  session.on('Network.loadingFailed', event => {
    if (ids.has(event.requestId)) record({ event: 'failed', requestId: event.requestId, timestamp: event.timestamp, errorText: event.errorText, canceled: event.canceled });
  });
  await session.send('Network.enable');
  page.__repairCharacterCdp = session;
  return { installed: true, at: new Date().toISOString(), origin, recordsHeadersOrTokens: false, interceptsOrReplays: false };
}
