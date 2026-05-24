import OptionLayout from "~/components/Layouts/Layout"
import { ResearchWorkspace } from "@/components/Option/ResearchWorkspace"

const OptionResearchWorkspace = () => {
  return (
    <OptionLayout>
      <div className="flex h-full min-h-0 w-full flex-1 overflow-hidden">
        <ResearchWorkspace />
      </div>
    </OptionLayout>
  )
}

export default OptionResearchWorkspace
