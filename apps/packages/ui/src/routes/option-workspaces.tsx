import OptionLayout from "~/components/Layouts/Layout"
import { WorkspacesManagerPage } from "@/components/Option/Workspaces/WorkspacesManagerPage"

const OptionWorkspaces = () => {
  return (
    <OptionLayout>
      <div
        data-testid="workspaces-route-shell"
        className="flex h-full min-h-0 w-full flex-1 overflow-hidden"
      >
        <WorkspacesManagerPage />
      </div>
    </OptionLayout>
  )
}

export default OptionWorkspaces
