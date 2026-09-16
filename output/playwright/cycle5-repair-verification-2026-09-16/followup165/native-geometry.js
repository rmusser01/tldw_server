async page => {
 const value=page.getByText('tldw:../../../Working/Language_Models/gemma-4-26B-A4B/gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf',{exact:true});
 return {at:new Date().toISOString(),url:page.url(),viewport:page.viewportSize(),geometry:await value.evaluate(el=>{const tile=el.parentElement;const range=document.createRange();range.selectNodeContents(el);const t=tile.getBoundingClientRect();return {text:el.textContent,wordBreak:getComputedStyle(el).wordBreak,valueRect:el.getBoundingClientRect().toJSON(),tileRect:t.toJSON(),textRects:[...range.getClientRects()].map(r=>r.toJSON()),textContained:[...range.getClientRects()].every(r=>r.left>=t.left-1&&r.right<=t.right+1),documentWidth:document.documentElement.scrollWidth,innerWidth:window.innerWidth}})};
}
