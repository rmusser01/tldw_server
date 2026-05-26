export const getWorkspaceSourcesLocationLabel = (
  isMobile: boolean
): string => (isMobile ? "Sources tab" : "Sources pane")

export const getWorkspaceChatNoSourcesHint = (
  isMobile: boolean
): string =>
  `Select sources from the ${getWorkspaceSourcesLocationLabel(isMobile)} for grounded answers, or type a message for general chat without sources`

export const getWorkspaceChatSourcesExplainer = (
  isMobile: boolean
): string =>
  `Sources are documents, PDFs, web pages, or other content you add. Add or select sources from the ${getWorkspaceSourcesLocationLabel(isMobile)} to ask grounded questions about them.`

export const getWorkspaceStudioNoSourcesHint = (
  isMobile: boolean
): string =>
  `Select sources from the ${getWorkspaceSourcesLocationLabel(isMobile)} to enable generation`
