import {
  createDocumentRequest,
  type DocumentRequest,
  type RequestKey,
} from "../types";

let requestId = 0;

export function requestFor(
  text: string,
  language: string | null = "json",
  overrides: Partial<RequestKey & Omit<DocumentRequest, "key">> = {},
): DocumentRequest {
  const {
    authority_id = "test",
    document_id = "n1",
    session_generation = 1,
    draft_revision = 1,
    language_revision = 1,
    request_id = ++requestId,
    ...fields
  } = overrides;
  return createDocumentRequest({
    key: {
      authority_id,
      document_id,
      session_generation,
      draft_revision,
      language_revision,
      request_id,
    },
    text,
    language,
    ...fields,
  });
}
