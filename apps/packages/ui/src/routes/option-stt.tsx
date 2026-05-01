import { lazy, Suspense } from "react"
import OptionLayout from "~/components/Layouts/Layout"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { useTranslation } from "react-i18next"
import { HostedAudioFeatureMessage } from "./HostedAudioFeatureMessage"

const SttPlaygroundPage = lazy(() =>
  import("@/components/Option/STT/SttPlaygroundPage")
)

const OptionStt = () => {
  const { t } = useTranslation("option")

  return (
    <OptionLayout>
      {isHostedTldwDeployment() ? (
        <HostedAudioFeatureMessage
          featureName={t("header.modeStt", "STT Playground")}
        />
      ) : (
        <Suspense
          fallback={
            <div className="sr-only" role="status">
              {t("hostedAudio.loadingStt", "Loading STT playground.")}
            </div>
          }
        >
          <SttPlaygroundPage />
        </Suspense>
      )}
    </OptionLayout>
  )
}

export default OptionStt
