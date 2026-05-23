import { useEffect } from "react"
import { useLocation, useSearchParams } from "react-router-dom"
import { usePrototypeWorkspace } from "@/hooks/usePrototypeWorkspaces"
import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import { PrototypeWorkspaceOwnerView } from "./PrototypeWorkspaceOwnerView"
import { PrototypeWorkspaceSessionView } from "./PrototypeWorkspaceSessionView"

interface PrototypeWorkspaceLocationState {
  prototypeSharePassword?: unknown
}

export const PrototypeWorkspacePage = () => {
  const [searchParams] = useSearchParams()
  const location = useLocation()

  const activeWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.activeWorkspaceId
  )
  const collaboratorSessionId = usePrototypeWorkspaceStore(
    (state) => state.collaboratorSessionId
  )
  const collaboratorSessionToken = usePrototypeWorkspaceStore(
    (state) => state.collaboratorSessionToken
  )
  const setActiveWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.setActiveWorkspaceId
  )
  const setCollaboratorEntry = usePrototypeWorkspaceStore(
    (state) => state.setCollaboratorEntry
  )

  const workspaceId = searchParams.get("workspace")
  const sessionToken = searchParams.get("session_token")
  const shareToken = searchParams.get("share_token")
  const navigationState = location.state as PrototypeWorkspaceLocationState | null
  const initialSharePassword =
    typeof navigationState?.prototypeSharePassword === "string"
      ? navigationState.prototypeSharePassword
      : null

  useEffect(() => {
    if (workspaceId) {
      setActiveWorkspaceId(workspaceId)
    }
  }, [workspaceId, setActiveWorkspaceId])

  useEffect(() => {
    if (sessionToken || shareToken) {
      setCollaboratorEntry({
        collaboratorSessionToken: sessionToken,
        collaboratorShareToken: shareToken
      })
    }
  }, [sessionToken, shareToken, setCollaboratorEntry])

  const hasStoredCollaboratorEntry = Boolean(
    workspaceId && (collaboratorSessionId || collaboratorSessionToken)
  )
  const isCollaboratorEntry = Boolean(
    sessionToken || shareToken || hasStoredCollaboratorEntry
  )
  const resolvedWorkspaceId = isCollaboratorEntry
    ? workspaceId ?? null
    : workspaceId ?? activeWorkspaceId
  const workspaceDetail = usePrototypeWorkspace(
    isCollaboratorEntry ? null : resolvedWorkspaceId
  )
  const viewerRole = workspaceDetail.data?.viewer_role ?? "owner"

  if (isCollaboratorEntry || viewerRole !== "owner") {
    return (
      <PrototypeWorkspaceSessionView
        prototypeWorkspaceId={resolvedWorkspaceId}
        sessionToken={sessionToken}
        shareToken={shareToken}
        initialPassword={initialSharePassword}
        workspace={workspaceDetail.data}
      />
    )
  }

  return (
    <PrototypeWorkspaceOwnerView
      prototypeWorkspaceId={resolvedWorkspaceId}
      workspace={workspaceDetail.data}
    />
  )
}
