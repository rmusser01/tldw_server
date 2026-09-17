async page=>{await page.getByRole('button',{name:'Close the ingest wizard',exact:true}).click();
 const list=page.waitForResponse(r=>r.url().includes('/flashcards/decks')&&r.request().method()==='GET',{timeout:15000});
 await page.goto('http://127.0.0.1:18683/flashcards?tab=cards');const before=await list;
 await page.getByRole('button',{name:'Create Flashcard',exact:true}).click();const d=page.getByRole('dialog',{name:'Create Flashcard',exact:true});
 const initial=await d.ariaSnapshot();
 await d.getByRole('button',{name:'or create a new deck',exact:true}).click();
 await d.getByRole('textbox',{name:'New deck name',exact:true}).fill('Alice Private Deck ORBIT742');
 const dr=page.waitForResponse(r=>r.url().endsWith('/flashcards/decks')&&r.request().method()==='POST',{timeout:10000});
 await d.getByRole('button',{name:'Create',exact:true}).first().click();const deck=await dr;
 await d.getByRole('textbox',{name:'New deck name',exact:true}).waitFor({state:'hidden',timeout:10000});
 await d.getByPlaceholder('Question or prompt...').fill('What is Rowan Observatory public booking code?');
 await d.getByPlaceholder('Answer...',{exact:true}).fill('ORBIT-742.');
 const cr=page.waitForResponse(r=>/\/flashcards\/?$/.test(r.url())&&r.request().method()==='POST',{timeout:10000});
 await d.getByRole('button',{name:'Create',exact:true}).click();const card=await cr;await d.waitFor({state:'hidden',timeout:10000});
 return {at:new Date().toISOString(),before:{status:before.status(),body:await before.json()},initial,deck:{status:deck.status(),body:await deck.json()},card:{status:card.status(),body:await card.json()},snapshot:await page.locator('body').ariaSnapshot()};
}

