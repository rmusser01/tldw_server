async(page)=>{
 const states=[];
 await page.getByRole('button',{name:'Knowledge QA',exact:true}).click();
 await page.getByRole('textbox',{name:'Search your knowledge base',exact:true}).waitFor({timeout:15000});
 states.push({surface:'Alice Knowledge QA after Bob',url:page.url(),snapshot:await page.locator('body').ariaSnapshot()});
 await page.getByRole('button',{name:'Chat',exact:true}).click();
 await page.getByRole('textbox',{name:'Type a message... (/ commands, @ mentions)',exact:true}).waitFor({timeout:15000});
 states.push({surface:'Alice Chat after Bob',url:page.url(),snapshot:await page.locator('body').ariaSnapshot()});
 const response=page.waitForResponse(r=>r.url().includes('/notes/b6ed11ef-8ac1-4b92-82c2-c5fd1e3bab4a')&&r.request().method()==='GET',{timeout:15000});
 await page.goto('http://127.0.0.1:18681/notes?source_ref_id=b6ed11ef-8ac1-4b92-82c2-c5fd1e3bab4a');
 const denied=await response;
 await page.getByRole('list',{name:'Notes',exact:true}).waitFor({timeout:10000});
 states.push({surface:'Alice foreign Bob Note',status:denied.status(),url:page.url(),snapshot:await page.locator('body').ariaSnapshot()});
 return {states,events:(page.__matrixEvents||[]).slice(-40),at:new Date().toISOString()};
}
