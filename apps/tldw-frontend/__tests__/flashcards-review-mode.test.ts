import type { Locator, Page } from '@playwright/test';
import { describe, expect, it, vi } from 'vitest';
import { FlashcardsPage } from '../e2e/utils/page-objects/FlashcardsPage';

vi.mock('@playwright/test', () => ({
  expect: (radio: { isChecked: () => Promise<boolean> }) => ({
    toBeChecked: async () => {
      if (!(await radio.isChecked())) throw new Error('Requested study mode is not selected');
    },
  }),
}));
vi.mock('../e2e/utils/helpers', () => ({}));

function setup(initialMode: 'Due only' | 'Cram') {
  let selectedMode = initialMode;
  const selectDueOnly = vi.fn(async () => {
    selectedMode = 'Due only';
  });
  const selectCram = vi.fn(async () => {
    selectedMode = 'Cram';
  });
  const flashcards = new FlashcardsPage({} as Page);
  const modeToggle = {
    getByRole: (_role: string, { name }: { name: string }) => ({
      isChecked: async () => selectedMode === name,
    }),
    getByText: (label: string) => ({
      click: label === 'Due only' ? selectDueOnly : selectCram,
    }),
  };
  vi.spyOn(flashcards, 'reviewModeToggle', 'get').mockReturnValue(modeToggle as unknown as Locator);
  return { flashcards, selectDueOnly, selectCram };
}

describe('Flashcards study mode selection', () => {
  it('leaves the already selected Due only control alone', async () => {
    const { flashcards, selectDueOnly } = setup('Due only');
    await flashcards.setReviewMode('due');
    expect(selectDueOnly).not.toHaveBeenCalled();
  });

  it('switches to Cram using a normal click', async () => {
    const { flashcards, selectCram } = setup('Due only');
    await flashcards.setReviewMode('cram');
    expect(selectCram).toHaveBeenCalledExactlyOnceWith();
  });

  it('switches back to Due only using a normal click', async () => {
    const { flashcards, selectDueOnly } = setup('Cram');
    await flashcards.setReviewMode('due');
    expect(selectDueOnly).toHaveBeenCalledExactlyOnceWith();
  });
});
