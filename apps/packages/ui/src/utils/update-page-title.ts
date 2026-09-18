export const updatePageTitle = (title: string = 'tldw Assistant') => {
  if (typeof document === "undefined") return
  // Next pages own their titles through Head (Chat observes its active history).
  // An asynchronous Chat callback must not override the current route's title.
  if (typeof window !== "undefined" && "__NEXT_DATA__" in window) return
  document.title = title
}
