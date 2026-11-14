import type { ChangeEvent } from "react"
import { useTranslation } from "react-i18next"

type ProviderKind = "github" | "local" | null

type ProviderSelectorProps = {
  provider: ProviderKind
  githubUrl: string
  busy: boolean
  onSelectProvider: (provider: Exclude<ProviderKind, null>) => void
  onGithubUrlChange: (value: string) => void
  onLoadGithub: () => void | Promise<void>
  onLocalFilesSelected: (files: FileList) => void | Promise<void>
}

const handleFileSelection = (
  event: ChangeEvent<HTMLInputElement>,
  callback: (files: FileList) => void | Promise<void>
) => {
  const files = event.target.files
  if (!files || files.length === 0) {
    event.target.value = ""
    return
  }
  void Promise.resolve(callback(files)).finally(() => {
    event.target.value = ""
  })
}

const directoryPickerAttributes = {
  webkitdirectory: "",
  directory: ""
} as Record<string, string>

export function ProviderSelector({
  provider,
  githubUrl,
  busy,
  onSelectProvider,
  onGithubUrlChange,
  onLoadGithub,
  onLocalFilesSelected
}: ProviderSelectorProps) {
  const { t } = useTranslation(["option"])

  return (
    <section className="space-y-3">
      <header>
        <h2 className="text-sm font-semibold">
          {t("option:repo2txt.providerTitle", {
            defaultValue: "Source Provider"
          })}
        </h2>
      </header>

      <div className="flex items-center gap-2">
        <button
          type="button"
          className="rounded border px-3 py-1.5 text-sm"
          disabled={busy}
          aria-pressed={provider === "github"}
          onClick={() => onSelectProvider("github")}
        >
          {t("option:repo2txt.providerGithub", { defaultValue: "GitHub" })}
        </button>
        <button
          type="button"
          className="rounded border px-3 py-1.5 text-sm"
          disabled={busy}
          aria-pressed={provider === "local"}
          onClick={() => onSelectProvider("local")}
        >
          {t("option:repo2txt.providerLocal", { defaultValue: "Local" })}
        </button>
      </div>

      {provider === "github" && (
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="url"
            disabled={busy}
            value={githubUrl}
            onChange={(event) => onGithubUrlChange(event.target.value)}
            placeholder={t("option:repo2txt.githubPlaceholder", {
              defaultValue: "https://github.com/owner/repo"
            })}
            className="min-w-[260px] flex-1 rounded border px-3 py-1.5 text-sm"
          />
          <button
            type="button"
            className="rounded border px-3 py-1.5 text-sm"
            onClick={() => void onLoadGithub()}
            disabled={busy || githubUrl.trim().length === 0}
          >
            {t("option:repo2txt.loadSource", { defaultValue: "Load Source" })}
          </button>
        </div>
      )}

      {provider === "local" && (
        <div className="flex flex-wrap items-center gap-2">
          <label className="inline-flex cursor-pointer items-center gap-2 rounded border px-3 py-1.5 text-sm">
            <span>
              {t("option:repo2txt.chooseDirectory", {
                defaultValue: "Choose Directory"
              })}
            </span>
            <input
              type="file"
              disabled={busy}
              data-testid="repo2txt-local-directory-input"
              className="hidden"
              onChange={(event) =>
                handleFileSelection(event, onLocalFilesSelected)
              }
              multiple
              {...directoryPickerAttributes}
            />
          </label>
          <label className="inline-flex cursor-pointer items-center gap-2 rounded border px-3 py-1.5 text-sm">
            <span>
              {t("option:repo2txt.chooseZip", { defaultValue: "Choose Zip" })}
            </span>
            <input
              type="file"
              disabled={busy}
              data-testid="repo2txt-local-zip-input"
              className="hidden"
              onChange={(event) =>
                handleFileSelection(event, onLocalFilesSelected)
              }
              accept=".zip,application/zip"
            />
          </label>
        </div>
      )}
    </section>
  )
}
