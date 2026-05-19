import OptionLayout from "~/components/Layouts/Layout"
import { ChatWorkspacePage } from "@/components/Option/ChatWorkspace"

const OptionChatWorkspace = () => {
  return (
    <OptionLayout>
      <div className="h-full min-h-0 w-full overflow-hidden">
        <ChatWorkspacePage />
      </div>
    </OptionLayout>
  )
}

export default OptionChatWorkspace
