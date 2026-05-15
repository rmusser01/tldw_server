import { RouteRedirect } from "@web/components/navigation/RouteRedirect"

export default function WorkspacePlaygroundRedirect() {
  return (
    <RouteRedirect
      to="/research-studio"
      title="Research Studio has moved"
      description="Legacy workspace links now open Research Studio."
    />
  )
}
