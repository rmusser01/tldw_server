import { SaveButton } from "@/components/Common/SaveButton"
import { getSearchSettings, setSearchSettings } from "@/services/search"
import { ALL_GOOGLE_DOMAINS } from "@/utils/google-domains"
import { SUPPORTED_SERACH_PROVIDERS } from "@/utils/search-provider"
import { useQuery } from "@tanstack/react-query"
import { Select, Skeleton, Switch, InputNumber, Input } from "antd"
import { useTranslation } from "react-i18next"
import { useSimpleForm } from "@/hooks/useSimpleForm"

export const SearchModeSettings = () => {
  const { t } = useTranslation("settings")

  const form = useSimpleForm({
    initialValues: {
      isSimpleInternetSearch: false,
      searchProvider: "",
      totalSearchResults: 0,
      visitSpecificWebsite: false,
      searxngURL: "",
      searxngJSONMode: false,
      braveApiKey: "",
      tavilyApiKey: "",
      googleDomain: "",
      defaultInternetSearchOn: false,
      exaAPIKey: "",
      firecrawlAPIKey: ""
    }
  })

  const { status } = useQuery({
    queryKey: ["fetchSearchSettings"],
    queryFn: async () => {
      const data = await getSearchSettings()
      form.setValues(data)
      form.resetDirty(data)
      return data
    }
  })

  const sectionHeader = (
    <div className="mb-5">
      <div className="flex items-center gap-2">
        <h2 className="text-base font-semibold leading-7 text-text">
          {t("generalSettings.webSearch.heading")}
        </h2>
        {form.isDirty() && (
          <span className="text-xs px-2 py-0.5 rounded bg-warn/10 text-warn">
            {t("generalSettings.webSearch.unsavedChanges", "Unsaved changes")}
          </span>
        )}
      </div>
      <div className="border-b border-border mt-3"></div>
    </div>
  )

  return (
    <div>
      {sectionHeader}
      {status === "pending" && <Skeleton active />}
      {status === "error" && (
        <p className="rounded-md border border-border bg-surface p-3 text-sm text-text-muted">
          {t(
            "generalSettings.webSearch.unavailable",
            "Web search settings are unavailable until the server is reachable."
          )}
        </p>
      )}
      {status === "success" && (
      <form
        onSubmit={form.onSubmit(async (values) => {
          await setSearchSettings(values)
          form.resetDirty(values)
        })}
        className="space-y-4">
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <span className="text-text ">
            {t("generalSettings.webSearch.provider.label")}
          </span>
          <div>
            <Select
              aria-label={t("generalSettings.webSearch.provider.label", "Web Search Provider")}
              placeholder={t("generalSettings.webSearch.provider.placeholder")}
              showSearch
              className="w-full mt-4 sm:mt-0 sm:w-[200px]"
              options={SUPPORTED_SERACH_PROVIDERS}
              filterOption={(input, option) =>
                option!.label.toLowerCase().indexOf(input.toLowerCase()) >= 0 ||
                option!.value.toLowerCase().indexOf(input.toLowerCase()) >= 0
              }
              {...form.getInputProps("searchProvider")}
            />
          </div>
        </div>
        {form.values.searchProvider === "searxng" && (
          <div className="transition-all duration-200">
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                {t("generalSettings.webSearch.searxng.url.label")} <span className="text-danger">*</span>
              </span>
              <div>
                <Input
                  aria-label={t("generalSettings.webSearch.searxng.url.label", "SearXNG URL")}
                  placeholder="https://searxng.example.com"
                  className="w-full mt-4 sm:mt-0 sm:w-[200px]"
                  required
                  {...form.getInputProps("searxngURL")}
                />
              </div>
            </div>
          </div>
        )}
        {form.values.searchProvider === "google" && (
          <div className="transition-all duration-200">
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                {t("generalSettings.webSearch.googleDomain.label")}
              </span>
              <div>
                <Select
                  aria-label={t("generalSettings.webSearch.googleDomain.label", "Google Domain")}
                  showSearch
                  className="w-full mt-4 sm:mt-0 sm:w-[200px]"
                  options={ALL_GOOGLE_DOMAINS.map((e) => ({
                    label: e,
                    value: e
                  }))}
                  filterOption={(input, option) =>
                    option!.label.toLowerCase().indexOf(input.toLowerCase()) >=
                      0 ||
                    option!.value.toLowerCase().indexOf(input.toLowerCase()) >=
                      0
                  }
                  {...form.getInputProps("googleDomain")}
                />
              </div>
            </div>
          </div>
        )}
        {form.values.searchProvider === "brave-api" && (
          <div className="transition-all duration-200">
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                {t("generalSettings.webSearch.braveApi.label")} <span className="text-danger">*</span>
              </span>
              <div>
                <Input.Password
                  aria-label={t("generalSettings.webSearch.braveApi.label", "Brave API Key")}
                  placeholder={t(
                    "generalSettings.webSearch.braveApi.placeholder"
                  )}
                  required
                  className="w-full mt-4 sm:mt-0 sm:w-[200px]"
                  {...form.getInputProps("braveApiKey")}
                />
              </div>
            </div>
          </div>
        )}
        {form.values.searchProvider === "tavily-api" && (
          <div className="transition-all duration-200">
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                {t("generalSettings.webSearch.tavilyApi.label")} <span className="text-danger">*</span>
              </span>
              <div>
                <Input.Password
                  aria-label={t("generalSettings.webSearch.tavilyApi.label", "Tavily API Key")}
                  placeholder={t(
                    "generalSettings.webSearch.tavilyApi.placeholder"
                  )}
                  required
                  className="w-full mt-4 sm:mt-0 sm:w-[200px]"
                  {...form.getInputProps("tavilyApiKey")}
                />
              </div>
            </div>
          </div>
        )}

        {form.values.searchProvider === "exa" && (
          <div className="transition-all duration-200">
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                {t("generalSettings.webSearch.exa.label")} <span className="text-danger">*</span>
              </span>
              <div>
                <Input.Password
                  aria-label={t("generalSettings.webSearch.exa.label", "Exa API Key")}
                  placeholder={t("generalSettings.webSearch.exa.placeholder")}
                  required
                  className="w-full mt-4 sm:mt-0 sm:w-[200px]"
                  {...form.getInputProps("exaAPIKey")}
                />
              </div>
            </div>
          </div>
        )}

        {form.values.searchProvider === "firecrawl" && (
          <div className="transition-all duration-200">
            <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
              <span className="text-text">
                {t("generalSettings.webSearch.firecrawlAPIKey.label")} <span className="text-danger">*</span>
              </span>
              <div>
                <Input.Password
                  aria-label={t("generalSettings.webSearch.firecrawlAPIKey.label", "Firecrawl API Key")}
                  placeholder={t(
                    "generalSettings.webSearch.firecrawlAPIKey.placeholder"
                  )}
                  required
                  className="w-full mt-4 sm:mt-0 sm:w-[200px]"
                  {...form.getInputProps("firecrawlAPIKey")}
                />
              </div>
            </div>
          </div>
        )}

        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <span className="text-text ">
            {t("generalSettings.webSearch.searchMode.label")}
          </span>
          <div>
            <Switch
              className="mt-4 sm:mt-0"
              aria-label={t("generalSettings.webSearch.searchMode.label", "Search Mode")}
              {...form.getInputProps("isSimpleInternetSearch", {
                type: "checkbox"
              })}
            />
          </div>
        </div>
        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <span className="text-text ">
            {t("generalSettings.webSearch.totalSearchResults.label")}
          </span>
          <div>
            <InputNumber
              aria-label={t("generalSettings.webSearch.totalSearchResults.label", "Total Search Results")}
              placeholder={t(
                "generalSettings.webSearch.totalSearchResults.placeholder"
              )}
              {...form.getInputProps("totalSearchResults")}
              className="!w-full mt-4 sm:mt-0 sm:w-[200px]"
            />
          </div>
        </div>

        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <span className="text-text ">
            {t("generalSettings.webSearch.visitSpecificWebsite.label")}
          </span>
          <div>
            <Switch
              className="mt-4 sm:mt-0"
              aria-label={t("generalSettings.webSearch.visitSpecificWebsite.label", "Visit Specific Website")}
              {...form.getInputProps("visitSpecificWebsite", {
                type: "checkbox"
              })}
            />
          </div>
        </div>

        <div className="flex sm:flex-row flex-col space-y-4 sm:space-y-0 sm:justify-between">
          <span className="text-text ">
            {t("generalSettings.webSearch.searchOnByDefault.label")}
          </span>
          <div>
            <Switch
              className="mt-4 sm:mt-0"
              aria-label={t("generalSettings.webSearch.searchOnByDefault.label", "Search On By Default")}
              {...form.getInputProps("defaultInternetSearchOn", {
                type: "checkbox"
              })}
            />
          </div>
        </div>
        <div className="flex justify-end">
          <SaveButton btnType="submit" />
        </div>
      </form>
      )}
    </div>
  )
}
