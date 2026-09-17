async page=>{
 const d=page.getByRole('dialog',{name:'Create Flashcard',exact:true});
 await d.getByRole('textbox',{name:'New deck name',exact:true}).fill('Bob Private Deck BIRCH913');
 const dr=page.waitForResponse(r=>r.url().endsWith('/flashcards/decks')&&r.request().method()==='POST',{timeout:10000});
 await d.getByRole('button',{name:'Create',exact:true}).first().click();const deck=await dr;
 await d.getByRole('textbox',{name:'New deck name',exact:true}).waitFor({state:'hidden',timeout:10000});
 await d.getByPlaceholder('Question or prompt...').fill('What is the Birch Workshop emblem?');
 await d.getByPlaceholder('Answer...',{exact:true}).fill('A silver kite. BIRCH-913.');
 const cr=page.waitForResponse(r=>/\/flashcards\/?$/.test(r.url())&&r.request().method()==='POST',{timeout:10000});
 await d.getByRole('button',{name:'Create',exact:true}).click();const card=await cr;
 await d.waitFor({state:'hidden',timeout:10000});
 return {at:new Date().toISOString(),deck:{status:deck.status(),body:await deck.json()},card:{status:card.status(),body:await card.json()},snapshot:await page.locator('body').ariaSnapshot()};
}
