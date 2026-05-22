import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type {
  PrototypeCollaboratorSessionCreateInput,
  PrototypeWorkspaceDetail,
  PrototypePromotionCreateInput,
  PrototypePromotionRequest,
  PrototypePromotionReviewInput,
  PrototypePromotionReviewResult,
  PrototypeSessionJob,
  PrototypeWorkspace,
  PrototypeWorkspaceCreateInput,
  PrototypeWorkspaceSessionCreateInput
} from "@/types/prototype-workspace"
import {
  createPrototypeCollaboratorBranchSessionRequest,
  createPrototypeOwnerBranchSessionRequest,
  createPrototypePromotionRequestRequest,
  createPrototypeWorkspaceRequest,
  getPrototypeWorkspaceRequest,
  reviewPrototypePromotionRequestRequest
} from "@/services/tldw/domains/prototype-workspaces"

export const prototypeWorkspaceQueryKeys = {
  all: () => ["prototype-workspaces"] as const,
  workspaces: () => [...prototypeWorkspaceQueryKeys.all(), "workspaces"] as const,
  workspace: (prototypeWorkspaceId: string | null | undefined) =>
    [
      ...prototypeWorkspaceQueryKeys.workspaces(),
      "detail",
      prototypeWorkspaceId ?? null
    ] as const,
  sessions: (prototypeWorkspaceId: string) =>
    [
      ...prototypeWorkspaceQueryKeys.all(),
      "sessions",
      String(prototypeWorkspaceId)
    ] as const,
  promotions: (prototypeWorkspaceId: string) =>
    [
      ...prototypeWorkspaceQueryKeys.all(),
      "promotions",
      String(prototypeWorkspaceId)
    ] as const
}

export const useCreatePrototypeWorkspace = () => {
  const queryClient = useQueryClient()
  return useMutation<PrototypeWorkspace, Error, PrototypeWorkspaceCreateInput>({
    mutationFn: createPrototypeWorkspaceRequest,
    onSuccess: async (workspace) => {
      queryClient.setQueryData(
        prototypeWorkspaceQueryKeys.workspace(workspace.id),
        workspace
      )
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.workspaces()
      })
    }
  })
}

export const usePrototypeWorkspace = (prototypeWorkspaceId: string | null | undefined) => {
  const queryWorkspaceId = prototypeWorkspaceId ?? null
  return useQuery<PrototypeWorkspaceDetail, Error>({
    queryKey: prototypeWorkspaceQueryKeys.workspace(queryWorkspaceId),
    queryFn: () => getPrototypeWorkspaceRequest(String(prototypeWorkspaceId)),
    enabled: Boolean(prototypeWorkspaceId)
  })
}

export const useCreateOwnerBranchSession = (prototypeWorkspaceId: string) => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypeSessionJob,
    Error,
    PrototypeWorkspaceSessionCreateInput
  >({
    mutationFn: async (body) => {
      if (!prototypeWorkspaceId) {
        throw new Error("prototype_workspace_id is required")
      }
      return createPrototypeOwnerBranchSessionRequest(prototypeWorkspaceId, body)
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.sessions(prototypeWorkspaceId)
      })
    }
  })
}

export const useCreateCollaboratorBranchSession = () => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypeSessionJob,
    Error,
    PrototypeCollaboratorSessionCreateInput
  >({
    mutationFn: createPrototypeCollaboratorBranchSessionRequest,
    onSuccess: async (sessionJob) => {
      await queryClient.invalidateQueries({
        queryKey: prototypeWorkspaceQueryKeys.sessions(
          sessionJob.prototype_workspace_id
        )
      })
    }
  })
}

export const useCreatePromotionRequest = () => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypePromotionRequest,
    Error,
    PrototypePromotionCreateInput
  >({
    mutationFn: createPrototypePromotionRequestRequest,
    onSuccess: async (_promotion, variables) => {
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: prototypeWorkspaceQueryKeys.workspace(
            variables.prototype_workspace_id
          )
        }),
        queryClient.invalidateQueries({
          queryKey: prototypeWorkspaceQueryKeys.promotions(
            variables.prototype_workspace_id
          )
        })
      ])
    }
  })
}

export const useReviewPrototypePromotionRequest = () => {
  const queryClient = useQueryClient()
  return useMutation<
    PrototypePromotionReviewResult,
    Error,
    PrototypePromotionReviewInput
  >({
    mutationFn: reviewPrototypePromotionRequestRequest,
    onSuccess: async (reviewResult, variables) => {
      const workspaceId =
        reviewResult.prototype_workspace_id || variables.prototype_workspace_id
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: prototypeWorkspaceQueryKeys.workspace(workspaceId)
        }),
        queryClient.invalidateQueries({
          queryKey: prototypeWorkspaceQueryKeys.promotions(workspaceId)
        })
      ])
    }
  })
}
