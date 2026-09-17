async page=>{
 const p=page.waitForResponse(r=>r.url().includes('/flashcards/decks')&&r.request().method()==='GET',{timeout:15000});
 await page.goto('http://127.0.0.1:18681/flashcards?tab=cards');const r=await p;
 await page.getByText('What is the Birch Workshop emblem?',{exact:true}).waitFor({timeout:15000});
 await page.getByRole('tabpanel',{name:'Manage',exact:true}).getByRole('combobox').first().click();
 await page.getByRole('option',{name:'Bob Private Deck BIRCH913',exact:true}).waitFor({timeout:10000});
 const snapshot=await page.locator('body').ariaSnapshot();await page.keyboard.press('Escape');
 return {at:new Date().toISOString(),decks:{status:r.status(),body:await r.json()},snapshot};
}
