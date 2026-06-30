import { Playground } from "~/components/Option/Playground/Playground"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionChat = () => {
  return (
    <RouteErrorBoundary routeId="chat" routeLabel="Chat">
      <h1 className="sr-only">Chat</h1>
      <Playground />
    </RouteErrorBoundary>
  )
}

export default OptionChat
