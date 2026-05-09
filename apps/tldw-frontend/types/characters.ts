export interface CharacterSummary {
  id: number;
  version?: number;
  name?: string | null;
  description?: string | null;
  tags?: string[] | string | null;
  creator?: string | null;
  image_present?: boolean;
}

export interface CharacterListQueryResponse {
  items?: CharacterSummary[];
  total?: number;
  page?: number;
  page_size?: number;
  has_more?: boolean;
  next_offset?: number | null;
}

export type CharacterListResponse = CharacterSummary[] | CharacterListQueryResponse;
