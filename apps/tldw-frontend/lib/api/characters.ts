import { apiClient } from '@web/lib/api';
import type { CharacterListResponse, CharacterSummary } from '@web/types/characters';

const CHARACTERS_BASE = '/characters';

function normalizeCharacterList(response: CharacterListResponse): CharacterSummary[] {
  if (Array.isArray(response)) {
    return response;
  }

  return Array.isArray(response.items) ? response.items : [];
}

export async function listCharacters(): Promise<CharacterSummary[]> {
  const response = await apiClient.get<CharacterListResponse>(`${CHARACTERS_BASE}/`, {
    params: {
      limit: 1000,
      offset: 0,
    },
  });
  return normalizeCharacterList(response);
}
