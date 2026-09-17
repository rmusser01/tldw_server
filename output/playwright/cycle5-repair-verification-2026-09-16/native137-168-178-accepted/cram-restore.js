async page => {
  if (!page.__uatCramFault) throw new Error('No owned Cram fault installed');
  await page.unroute(page.__uatCramFault.match, page.__uatCramFault.handler);
  delete page.__uatCramFault;
  return {restored: true, at: new Date().toISOString()};
}
