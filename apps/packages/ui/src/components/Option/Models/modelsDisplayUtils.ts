const LAST_REFRESHED_TIME_FORMATTER = new Intl.DateTimeFormat("en-GB", {
  hour: "2-digit",
  hourCycle: "h23",
  minute: "2-digit"
})

export const formatModelsLastRefreshedTime = (timestamp: number): string =>
  LAST_REFRESHED_TIME_FORMATTER.format(new Date(timestamp))
