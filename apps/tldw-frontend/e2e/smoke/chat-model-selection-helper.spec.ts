import { expect, test } from "@playwright/test"
import { ChatPage } from "../utils/page-objects/ChatPage"

// Isolated browser checks for the page object; no application API or server.
for (const role of ["option", "menuitem"]) {
  test(`waits for a delayed ${role} before accepting model selection`, async ({ page }) => {
    await page.setContent(`<button data-testid="model-selector">gpt-4o</button>
      <script>
        const trigger = document.querySelector('button');
        trigger.onclick = () => setTimeout(() => {
          const choice = document.createElement('button');
          choice.setAttribute('role', '${role}');
          choice.innerHTML = '<span>gpt-4.1-mini</span>';
          choice.dataset.testid = 'model-selector-option';
          choice.dataset.modelId = 'gpt-4.1-mini';
          choice.onclick = () => { trigger.textContent = choice.textContent; choice.remove(); };
          document.body.append(choice);
        }, 300);
      </script>`)
    await new ChatPage(page).selectModel("gpt-4.1-mini")
    await expect(page.getByTestId("model-selector")).toHaveText("gpt-4.1-mini")
  })
}

test("fails explicitly when the requested model is absent", async ({ page }) => {
  await page.setContent('<button data-testid="model-selector">gpt-4o</button>')
  await expect(new ChatPage(page).selectModel("missing-model")).rejects.toThrow()
  await expect(page.getByTestId("model-selector")).toHaveText("gpt-4o")
})

test("treats punctuation in model IDs literally", async ({ page }) => {
  await page.setContent(`<button data-testid="model-selector">gpt-4o</button>
    <button role="option" data-testid="model-selector-option" data-model-id="gpt-4X1-mini"><span>gpt-4X1-mini</span></button><button role="option" data-testid="model-selector-option" data-model-id="gpt-4.1-mini"><span>gpt-4.1-mini</span></button>
    <script>document.querySelectorAll('[role=option]').forEach(choice => {
      choice.onclick = () => { document.querySelector('[data-testid=model-selector]').textContent = choice.textContent; };
    });</script>`)
  await new ChatPage(page).selectModel("gpt-4.1-mini")
  await expect(page.getByTestId("model-selector")).toHaveText("gpt-4.1-mini")
})

for (const label of ["gpt-4.1-mini", "My quick model"]) {
  test(`selects the exact model identity with display label ${label}`, async ({ page }) => {
    await page.setContent(`<button data-testid="model-selector">gpt-4o</button>
      <button role="menuitem" data-testid="model-selector-option" data-model-id="gpt-4.1-mini-preview"><span>gpt-4.1-mini-preview</span></button>
      <button role="menuitem" data-testid="model-selector-option" data-model-id="gpt-4.1-mini"><span>${label}</span></button>
      <script>document.querySelectorAll('[role=menuitem]').forEach(choice => {
        choice.onclick = () => {
          const trigger = document.querySelector('[data-testid=model-selector]');
          trigger.textContent = choice.textContent;
          trigger.dataset.selectedModel = choice.dataset.modelId;
        };
      });</script>`)
    await new ChatPage(page).selectModel("gpt-4.1-mini")
    await expect(page.getByTestId("model-selector")).toHaveAttribute("data-selected-model", "gpt-4.1-mini")
    await expect(page.getByTestId("model-selector")).toHaveText(label)
  })
}
