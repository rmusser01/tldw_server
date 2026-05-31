import { ExternalLink } from "lucide-react"
import { useTranslation } from "react-i18next"

type HostedAudioFeatureMessageProps = {
  featureName: string
}

export const HostedAudioFeatureMessage = ({
  featureName
}: HostedAudioFeatureMessageProps) => {
  const { t } = useTranslation("option")

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-10">
      <section
        aria-labelledby="hosted-audio-feature-title"
        className="rounded-lg border border-border bg-surface px-5 py-6 shadow-sm"
      >
        <p className="text-xs font-semibold uppercase tracking-wide text-text-subtle">
          {t("hostedAudio.badge", "Hosted mode")}
        </p>
        <h1
          id="hosted-audio-feature-title"
          className="mt-2 text-2xl font-semibold text-text"
        >
          {t(
            "hostedAudio.title",
            "Audio features require a self-hosted tldw server"
          )}
        </h1>
        <p className="mt-3 text-sm leading-6 text-text-muted">
          {t(
            "hostedAudio.description",
            "{{featureName}} depends on server-side audio runtimes, local model dependencies, host audio tooling, or provider keys that are only exposed in self-hosted deployments. The hosted product keeps these routes visible so you can find the feature boundary without exposing controls that cannot run here.",
            { featureName }
          )}
        </p>
        <a
          href="https://github.com/rmusser01/tldw_server#quickstart"
          target="_blank"
          rel="noopener noreferrer"
          className="mt-5 inline-flex items-center gap-2 rounded-md border border-border bg-surface2 px-3 py-2 text-sm font-medium text-text transition-colors hover:bg-surface3"
        >
          {t("hostedAudio.quickstart", "Open self-hosting quickstart")}
          <span className="sr-only">
            {" "}
            {t("hostedAudio.opensInNewTab", "(opens in new tab)")}
          </span>
          <ExternalLink className="size-4" aria-hidden="true" />
        </a>
      </section>
    </div>
  )
}
