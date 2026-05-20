import OptionLayout from "~/components/Layouts/Layout"
import { AudiobookStudioPage } from "~/components/Option/AudiobookStudio/AudiobookStudioPage"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"

const OptionAudiobookStudio = () => {
  return (
    <RouteErrorBoundary routeId="audiobook-studio" routeLabel="Audiobook Studio">
      <OptionLayout>
        <AudiobookStudioPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAudiobookStudio
