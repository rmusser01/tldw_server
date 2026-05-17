import OptionLayout from "~/components/Layouts/Layout"
import { WorkspacePlayground } from "@/components/Option/WorkspacePlayground"

const OptionWorkspacePlayground = () => {
  return (
    <OptionLayout>
      <div className="flex min-h-0 w-full flex-1">
        <WorkspacePlayground />
      </div>
    </OptionLayout>
  )
}

export default OptionWorkspacePlayground
