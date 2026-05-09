import { apiClient } from '@web/lib/api';
import type { CharacterListQueryResponse, CharacterListResponse, CharacterSummary } from '@web/types/characters';

const CHARACTERS_BASE = '/characters';
const CHARACTER_QUERY_PAGE_SIZE = 100;

export interface ListCharactersOptions {
  query?: string;
  pageSize?: number;
}

function normalizeCharacterList(response: CharacterListResponse): CharacterSummary[] {
  if (Array.isArray(response)) {
    return response;
  }

  return Array.isArray(response.items) ? response.items : [];
}

function normalizePageSize(pageSize?: number): number {
  if (!Number.isFinite(pageSize)) return CHARACTER_QUERY_PAGE_SIZE;
  return Math.min(Math.max(Math.trunc(pageSize ?? CHARACTER_QUERY_PAGE_SIZE), 1), CHARACTER_QUERY_PAGE_SIZE);
}

export async function listCharacters(options: ListCharactersOptions = {}): Promise<CharacterSummary[]> {
  const pageSize = normalizePageSize(options.pageSize);
  const query = options.query?.trim();
  const characters: CharacterSummary[] = [];
  let page = 1;

  while (true) {
    const response = await apiClient.get<CharacterListQueryResponse>(`${CHARACTERS_BASE}/query`, {
      params: {
        include_image_base64: false,
        page,
        page_size: pageSize,
        ...(query ? { query } : {}),
      },
    });
    const pageItems = normalizeCharacterList(response);
    characters.push(...pageItems);

    if (!response.has_more || pageItems.length === 0) {
      break;
    }
    page += 1;
  }

  return characters;
}
