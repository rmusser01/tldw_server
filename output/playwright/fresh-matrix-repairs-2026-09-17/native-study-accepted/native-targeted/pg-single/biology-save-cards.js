async page=>{await page.getByRole('button',{name:'Save generated cards',exact:true}).click();return {at:new Date().toISOString(),body:await page.locator('body').ariaSnapshot()};}
