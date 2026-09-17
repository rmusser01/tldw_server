async page=>({at:new Date().toISOString(),body:await page.locator('body').ariaSnapshot(),events:page.__matrixEvents.filter(e=>e.at>page.__wikiAt&&/ingest|process/.test(e.url||''))})
