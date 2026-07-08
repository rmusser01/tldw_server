import dynamic from "next/dynamic"
import Head from "next/head"

export default dynamic(async () => {
  const { SettingsRoute } = await import("@/routes/settings-route")
  const mod = await import("@/components/Option/Settings/data-management")
  const Component = mod.DataManagementSettings
  const Page = () => (
    <>
      <Head>
        <title>Data Management | Settings | tldw</title>
      </Head>
      <SettingsRoute>
        <Component />
      </SettingsRoute>
    </>
  )
  return { default: Page }
}, { ssr: false })
