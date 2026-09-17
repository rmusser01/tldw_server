async page=>({at:new Date().toISOString(),body:await page.locator('body').ariaSnapshot(),events:page.__matrixEvents.filter(e=>/\/flashcards\//.test(e.url||''))})
