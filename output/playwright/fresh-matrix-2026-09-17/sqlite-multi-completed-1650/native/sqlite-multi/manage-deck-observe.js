async page=>{await page.getByRole('tab',{name:'Manage',exact:true}).click();return {url:page.url(),snapshot:await page.locator('body').ariaSnapshot()};}
