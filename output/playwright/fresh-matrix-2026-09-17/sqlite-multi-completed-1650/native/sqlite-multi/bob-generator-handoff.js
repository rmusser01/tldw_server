async (page) => {
  await page.getByRole('region',{name:'Note editor'}).getByRole('button',{name:'More actions',exact:true}).click();
  await page.getByRole('menuitem',{name:'Generate flashcards',exact:true}).click();
  await page.getByRole('textbox',{name:'New deck name',exact:true}).first().waitFor({state:'visible',timeout:20000});
  await page.getByRole('textbox',{name:'New deck name',exact:true}).first().fill('BOB ONLY DECK DRAFT BIRCH913');
  return {url:page.url(),body:await page.locator('body').innerText()};
}
