async page => ({at:new Date().toISOString(),url:page.url(),firstChat:page.__firstSetupChat || [],body:await page.locator('body').innerText()})
