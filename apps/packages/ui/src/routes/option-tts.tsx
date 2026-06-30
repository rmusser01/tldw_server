import { lazy, Suspense } from "react"
import OptionLayout from "~/components/Layouts/Layout"
import { RouteErrorBoundary } from "@/components/Common/RouteErrorBoundary"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { useTranslation } from "react-i18next"
import { HostedAudioFeatureMessage } from "./HostedAudioFeatureMessage"

const SpeechPlaygroundPage = lazy(() =>
  import("@/components/Option/Speech/SpeechPlaygroundPage")
)

const OptionTts = () => {
  const { t } = useTranslation("option")

  return (
    <RouteErrorBoundary routeId="tts" routeLabel="TTS Playground">
      <OptionLayout>
        {isHostedTldwDeployment() ? (
          <HostedAudioFeatureMessage
            featureName={t("tts.playground", "TTS Playground")}
          />
        ) : (
          <Suspense
            fallback={
              <div className="sr-only" role="status">
                {t("hostedAudio.loadingTts", "Loading TTS playground.")}
              </div>
            }
          >
            <SpeechPlaygroundPage lockedMode="listen" hideModeSwitcher />
          </Suspense>
        )}
      </OptionLayout>
    </RouteErrorBoundary>
  )
}

export default OptionTts
