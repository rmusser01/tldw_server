import { describe, expect, it, vi } from 'vitest';
import { ContentReviewPage } from '../ContentReviewPage';

vi.mock('../../helpers', () => ({
  waitForAppShell: async () => {},
  waitForConnection: async () => {},
}));

describe('ContentReviewPage readiness', () => {
  it('rejects when neither the review heading nor the empty state renders', async () => {
    const missing = {
      waitFor: async () => {
        throw new Error('review UI missing');
      },
      first() {
        return this;
      },
      filter() {
        return this;
      },
    };
    const page = { getByText: () => missing, locator: () => missing };
    await expect(new ContentReviewPage(page as never).assertPageReady()).rejects.toThrow(
      'review UI missing'
    );
  });
});
