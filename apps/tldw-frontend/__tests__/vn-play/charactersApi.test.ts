import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  apiClient: {
    get: vi.fn(),
  },
}));

vi.mock('@web/lib/api', () => ({
  apiClient: mocks.apiClient,
}));

import { listCharacters } from '@web/lib/api/characters';

describe('characters api client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('loads characters through the paginated query endpoint', async () => {
    mocks.apiClient.get
      .mockResolvedValueOnce({
        items: [{ id: 1, name: 'Alpha' }],
        has_more: true,
        next_offset: 100,
        page: 1,
        page_size: 100,
        total: 2,
      })
      .mockResolvedValueOnce({
        items: [{ id: 2, name: 'Beta' }],
        has_more: false,
        next_offset: null,
        page: 2,
        page_size: 100,
        total: 2,
      });

    const characters = await listCharacters();

    expect(characters.map((character) => character.id)).toEqual([1, 2]);
    expect(mocks.apiClient.get).toHaveBeenNthCalledWith(1, '/characters/query', {
      params: {
        include_image_base64: false,
        page: 1,
        page_size: 100,
      },
    });
    expect(mocks.apiClient.get).toHaveBeenNthCalledWith(2, '/characters/query', {
      params: {
        include_image_base64: false,
        page: 2,
        page_size: 100,
      },
    });
  });

  it('passes search text through to the character query endpoint', async () => {
    mocks.apiClient.get.mockResolvedValueOnce({
      items: [{ id: 7, name: 'Mira Vale' }],
      has_more: false,
      next_offset: null,
      page: 1,
      page_size: 100,
      total: 1,
    });

    await listCharacters({ query: 'mira' });

    expect(mocks.apiClient.get).toHaveBeenCalledWith('/characters/query', {
      params: {
        include_image_base64: false,
        page: 1,
        page_size: 100,
        query: 'mira',
      },
    });
  });
});
