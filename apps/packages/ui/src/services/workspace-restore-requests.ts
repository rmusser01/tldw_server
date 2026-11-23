import { bgRequest } from "@/services/background-proxy";
import {
  requestScopeFields,
  type ServicePromptRequestScope,
} from "@/services/tldw/domains/service-prompts";
import type {
  WorkspaceArtifactApiResponse,
  WorkspaceContextResponse,
  WorkspaceListApiResponse,
  WorkspaceNoteApiResponse,
} from "@/services/tldw/domains/workspace-api";
import type { ClientPath, ClientPathRuntimeWithQuery } from "@/services/tldw/openapi-guard";
import type { KnowledgeNoteHead } from "@/utils/knowledge-note-provenance";

type RestoreRequest = {
  requestScope: ServicePromptRequestScope;
  abortSignal: AbortSignal;
};

type RestoreNote = KnowledgeNoteHead & {
  id: string;
  title: string;
  content: string;
  keywords?: unknown[];
  version: number;
};

const NOTES_SEARCH_PATH = "/api/v1/notes/search/" satisfies ClientPath;

const scopedGet = <T>(
  path: ClientPathRuntimeWithQuery,
  { requestScope, abortSignal }: RestoreRequest,
): Promise<T> => bgRequest<T, ClientPathRuntimeWithQuery>({
  method: "GET",
  abortSignal,
  ...requestScopeFields(requestScope),
  path,
});

export const workspaceRestoreRequests = {
  searchNotes(workspaceId: string, request: RestoreRequest) {
    const token = encodeURIComponent(`workspace:${workspaceId}`);
    return scopedGet<{ notes?: RestoreNote[] }>(
      `${NOTES_SEARCH_PATH}?tokens=${token}&limit=100&include_keywords=true`,
      request,
    );
  },
  getNote(id: string, request: RestoreRequest) {
    return scopedGet<RestoreNote>(`/api/v1/notes/${encodeURIComponent(id)}`, request);
  },
  listWorkspaces(request: RestoreRequest) {
    return scopedGet<WorkspaceListApiResponse>("/api/v1/workspaces/", request);
  },
  getContext(workspaceId: string, request: RestoreRequest) {
    return scopedGet<WorkspaceContextResponse>(
      `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/context`,
      request,
    );
  },
  getContents(workspaceId: string, request: RestoreRequest) {
    const path = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}` as const;
    return Promise.all([
      scopedGet<WorkspaceArtifactApiResponse[]>(`${path}/artifacts`, request),
      scopedGet<WorkspaceNoteApiResponse[]>(`${path}/notes`, request),
    ]);
  },
};
