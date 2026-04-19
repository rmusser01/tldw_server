import { useEffect } from "react"
import { useSearchParams } from "react-router-dom"
import { usePrototypeWorkspace } from "@/hooks/usePrototypeWorkspaces"
import { usePrototypeWorkspaceStore } from "@/store/prototype-workspace"
import { PrototypeWorkspaceOwnerView } from "./PrototypeWorkspaceOwnerView"
import { PrototypeWorkspaceSessionView } from "./PrototypeWorkspaceSessionView"

export const PrototypeWorkspacePage = () => {
  const [searchParams] = useSearchParams()

  const activeWorkspaceId = usePrototypeWorkspaceStore(
    (state) => state.activeWorkspaceId
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

  const resolvedWorkspaceId = workspaceId ?? activeWorkspaceId
  const isCollaboratorEntry = Boolean(sessionToken || shareToken)
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
