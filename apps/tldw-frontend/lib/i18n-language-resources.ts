// Keep the locale lookup map out of the shared app shell.
export const loadLanguageResources = (
  language: string,
  namespaces: readonly string[]
) =>
  Promise.all(
    namespaces.map((namespace) =>
      import(`@tldw/ui/assets/locale/${language}/${namespace}.json`).catch(
        () => null
      )
    )
  )
