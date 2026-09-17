async(page)=>{
 const states=[];
 await page.goto('http://127.0.0.1:18681/media?id=1');
 await page.getByRole('region',{name:'Media content',exact:true}).getByText('Rowan Observatory — public synthetic UAT handbook',{exact:true}).waitFor({timeout:15000});
 states.push({surface:'Alice own Media1 restored',snapshot:await page.locator('body').ariaSnapshot()});
 await page.goto('http://127.0.0.1:18681/notes?source_ref_id=4b06d0b4-51a5-4857-8d83-4082df846767');
 await page.getByRole('textbox',{name:'Note content',exact:true}).waitFor({timeout:15000});
 states.push({surface:'Alice own Biology Note restored',snapshot:await page.locator('body').ariaSnapshot()});
 await page.getByRole('region',{name:'Note editor'}).getByRole('button',{name:'More actions',exact:true}).click();
 await page.getByRole('menuitem',{name:'Generate flashcards',exact:true}).click();
 await page.getByRole('textbox',{name:'New deck name',exact:true}).first().fill('ALICE SECOND DECK DRAFT ORBIT742');
 states.push({surface:'Alice generator before switch',snapshot:await page.locator('body').ariaSnapshot()});
 await page.goto('http://127.0.0.1:18681/settings/tldw');
 await page.getByRole('button',{name:'Logout',exact:true}).click();
 await page.getByRole('button',{name:'Logout',exact:true}).waitFor({state:'hidden',timeout:10000});
 await page.goto('http://127.0.0.1:18681/login');
 await page.getByRole('heading',{name:'Sign in to tldw',exact:true}).waitFor({timeout:10000});
 return {states,url:page.url(),at:new Date().toISOString()};
}
