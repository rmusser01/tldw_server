import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { explainerApi } from "./explainerApi"

export const EXPLAINER_QUERY_KEYS = {
  sessions: ["explainer", "sessions"] as const,
  session: (sessionId: string | null | undefined) =>
    ["explainer", "session", sessionId ?? "none"] as const,
  job: (jobId: string | null | undefined) =>
    ["explainer", "job", jobId ?? "none"] as const
}

export const useExplainerSessions = () =>
  useQuery({
    queryKey: EXPLAINER_QUERY_KEYS.sessions,
    queryFn: () => explainerApi.listSessions({ limit: 50, offset: 0 })
  })

export const useExplainerSession = (sessionId: string | null | undefined) =>
  useQuery({
    queryKey: EXPLAINER_QUERY_KEYS.session(sessionId),
    queryFn: () => explainerApi.getSession(sessionId as string),
    enabled: Boolean(sessionId)
  })

export const useExplainerJob = (
  jobId: string | null | undefined,
  sessionId: string | null | undefined
) => {
  const queryClient = useQueryClient()
  return useQuery({
    queryKey: EXPLAINER_QUERY_KEYS.job(jobId),
    queryFn: async () => {
      const job = await explainerApi.getJob(jobId as string)
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        void queryClient.invalidateQueries({
          queryKey: EXPLAINER_QUERY_KEYS.session(sessionId)
        })
      }
      return job
    },
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (!status) return 1500
      return ["completed", "failed", "cancelled"].includes(status) ? false : 1500
    }
  })
}

export const useExplainerMutations = () => {
  const queryClient = useQueryClient()
  const refreshSessions = () =>
    queryClient.invalidateQueries({ queryKey: EXPLAINER_QUERY_KEYS.sessions })
  const refreshSession = (sessionId: string) =>
    queryClient.invalidateQueries({ queryKey: EXPLAINER_QUERY_KEYS.session(sessionId) })

  return {
    createSession: useMutation({
      mutationFn: (payload: Parameters<typeof explainerApi.createSession>[0]) =>
        explainerApi.createSession(payload),
      onSuccess: (session) => {
        refreshSessions()
        if (session?.id) refreshSession(session.id)
      }
    }),
    updateSession: useMutation({
      mutationFn: ({
        sessionId,
        payload
      }: {
        sessionId: string
        payload: Parameters<typeof explainerApi.updateSession>[1]
      }) => explainerApi.updateSession(sessionId, payload),
      onSuccess: (session) => {
        if (session?.id) refreshSession(session.id)
      }
    }),
    expandNode: useMutation({
      mutationFn: ({
        sessionId,
        nodeId,
        payload
      }: {
        sessionId: string
        nodeId: string
        payload?: Parameters<typeof explainerApi.expandNode>[2]
      }) => explainerApi.expandNode(sessionId, nodeId, payload),
      onSuccess: (_job, variables) => refreshSession(variables.sessionId)
    }),
    answerQuestion: useMutation({
      mutationFn: ({
        sessionId,
        nodeId,
        payload
      }: {
        sessionId: string
        nodeId: string
        payload: Parameters<typeof explainerApi.answerQuestion>[2]
      }) => explainerApi.answerQuestion(sessionId, nodeId, payload),
      onSuccess: (_node, variables) => refreshSession(variables.sessionId)
    }),
    exportChatbook: useMutation({
      mutationFn: ({
        sessionId,
        payload
      }: {
        sessionId: string
        payload: Parameters<typeof explainerApi.exportChatbook>[1]
      }) => explainerApi.exportChatbook(sessionId, payload)
    })
  }
}
