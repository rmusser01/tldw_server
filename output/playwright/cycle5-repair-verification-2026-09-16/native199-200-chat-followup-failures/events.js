async page=>({at:new Date().toISOString(),url:page.url(),events:page.__uat031Events??[],faults:page.__uat031Faults??[],visible:await page.locator('body').innerText()})
