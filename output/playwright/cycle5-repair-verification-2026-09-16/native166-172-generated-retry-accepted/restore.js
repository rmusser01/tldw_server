async page => { page.__uat172FailDeckReads=false;await page.unroute('**/api/v1/flashcards/decks*',page.__uat172Route);return {at:new Date().toISOString(),faults:page.__uat172Faults,restored:true}; }
