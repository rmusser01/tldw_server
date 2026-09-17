async page => ({at:new Date().toISOString(),url:page.url(),events:page.__matrixEvents || [],body:await page.locator('body').innerText()})
