async page => ({at:new Date().toISOString(),url:page.url(),ui:await page.locator('body').ariaSnapshot(),events:page.__matrixEvents})
