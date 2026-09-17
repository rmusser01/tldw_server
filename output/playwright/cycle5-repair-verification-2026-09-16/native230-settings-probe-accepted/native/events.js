async (page) => ({at:new Date().toISOString(),url:page.url(),events:page.__uat230Events || [],body:await page.locator('body').innerText()})
