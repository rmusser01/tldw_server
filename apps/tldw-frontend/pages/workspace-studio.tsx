import { RouteRedirect } from "@web/components/navigation/RouteRedirect"

export default function WorkspaceStudioRedirect() {
  return (
    <RouteRedirect
      to="/research-studio"
      title="Research Studio has moved"
      description="Workspace Studio links now open Research Studio."
    />
  )
}
