async page => ({at:new Date().toISOString(),url:page.url(),events:page.__uatReviewedEvents??[],body:await page.locator('body').innerText()})
