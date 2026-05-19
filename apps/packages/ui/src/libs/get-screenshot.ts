import { isChromiumTarget } from "@/config/platform"

export const captureVisibleTabScreenshot = () => {
  const result = new Promise<string>((resolve, reject) => {
    if (isChromiumTarget) {
      chrome.tabs.query({ active: true, currentWindow: true }, () => {
        chrome.tabs.captureVisibleTab(null, { format: "png" }, (dataUrl) => {
          const errorMessage = chrome.runtime?.lastError?.message
          if (errorMessage) {
            reject(new Error(errorMessage))
            return
          }
          if (!dataUrl) {
            reject(new Error("Failed to capture screenshot"))
            return
          }
          resolve(dataUrl)
        })
      })
    } else {
      browser.tabs
        .query({ active: true, currentWindow: true })
        .then(async (tabs) => {
          const dataUrl = (await Promise.race([
            browser.tabs.captureVisibleTab(null, { format: "png" }),
            new Promise((_, reject) =>
              setTimeout(
                () => reject(new Error("Screenshot capture timed out")),
                10000
              )
            )
          ])) as string
          if (!dataUrl) {
            reject(new Error("Failed to capture screenshot"))
            return
          }
          resolve(dataUrl)
        })
        .catch(reject)
    }
  })
  return result
}

export const getScreenshotFromCurrentTab = async () => {
  try {
    const screenshotDataUrl = await captureVisibleTabScreenshot()
    return {
      success: true,
      screenshot: screenshotDataUrl,
      error: null
    }
  } catch (error) {
    return {
      success: false,
      screenshot: null,
      error:
        error instanceof Error ? error.message : "Failed to capture screenshot"
    }
  }
}
