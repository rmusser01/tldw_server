import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { AudioStudioPage } from "@/components/Option/AudioStudio/AudioStudioPage"

const OptionAudioStudio = () => {
  return (
    <RouteErrorBoundary routeId="audio-studio" routeLabel="Audio Studio">
      <OptionLayout>
        <AudioStudioPage />
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionAudioStudio
