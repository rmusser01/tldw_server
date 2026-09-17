async page=>({at:new Date().toISOString(),url:page.url(),events:page.__uat172Events??[],faults:page.__uat172Faults??[],visible:await page.locator('body').innerText()})
