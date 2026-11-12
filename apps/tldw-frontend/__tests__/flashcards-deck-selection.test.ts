import {describe,expect,it,vi} from 'vitest';
import type {Locator,Page} from '@playwright/test';
import {FlashcardsPage} from '../e2e/utils/page-objects/FlashcardsPage';
vi.mock('@playwright/test',()=>({expect:(locator:{isVisible:()=>Promise<boolean>})=>({toBeVisible:async()=>{if(!await locator.isVisible())throw new Error('Required deck is not visible');}})}));
vi.mock('../e2e/utils/helpers',()=>({}));
function setup(initial:string,available=true){
 let selected=initial,open=false;
 const click=vi.fn(async()=>{open=true;});
 const optionClick=vi.fn(async()=>{selected='owned-deck';open=false;});
 const selector={click,getByText:(name:string)=>({isVisible:async()=>selected===name})};
 const option={click:optionClick,isVisible:async()=>open&&available};
 const page=new FlashcardsPage({} as Page);
 vi.spyOn(page,'reviewDeckSelect','get').mockReturnValue(selector as unknown as Locator);
 vi.spyOn(page as unknown as {getActiveSelectOption:(name:string)=>Locator},'getActiveSelectOption').mockReturnValue(option as unknown as Locator);
 return {page,click,optionClick,selected:()=>selected};
}
describe('Study deck selection',()=>{
 it('leaves an already selected exact deck alone',async()=>{const s=setup('owned-deck');await s.page.selectReviewDeckByName('owned-deck');expect(s.click).not.toHaveBeenCalled();expect(s.optionClick).not.toHaveBeenCalled();});
 it('selects a different deck using normal interaction',async()=>{const s=setup('another-deck');await s.page.selectReviewDeckByName('owned-deck');expect(s.click).toHaveBeenCalledWith();expect(s.optionClick).toHaveBeenCalledWith();expect(s.selected()).toBe('owned-deck');});
 it('fails when the requested option is unavailable',async()=>{const s=setup('another-deck',false);await expect(s.page.selectReviewDeckByName('owned-deck')).rejects.toThrow('Required deck is not visible');expect(s.optionClick).not.toHaveBeenCalled();});
});
