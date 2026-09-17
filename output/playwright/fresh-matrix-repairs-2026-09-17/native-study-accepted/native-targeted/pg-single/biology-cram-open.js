async page=>{await page.getByText('Cram',{exact:true}).click();return {at:new Date().toISOString(),body:await page.getByRole('tabpanel',{name:'Study',exact:true}).ariaSnapshot()};}
