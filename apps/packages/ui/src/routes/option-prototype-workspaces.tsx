import OptionLayout from "~/components/Layouts/Layout"
import { PrototypeWorkspacePage } from "@/components/Option/PrototypeWorkspace"

const OptionPrototypeWorkspaces = () => {
  return (
    <OptionLayout>
      <div
        data-testid="prototype-workspaces-route-shell"
        className="flex flex-1 min-h-0 w-full overflow-hidden"
      >
        <PrototypeWorkspacePage />
      </div>
    </OptionLayout>
  )
}

export default OptionPrototypeWorkspaces
