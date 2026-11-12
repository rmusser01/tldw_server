import { isGoogleDocs, parseGoogleDocs } from "@/parser/google-docs"
import { cleanUnwantedUnicode } from "@/utils/clean"
import { isChromiumTarget, isFirefoxTarget } from "@/config/platform"

const _getHtml = () => {
  const url = window.location.href
  if (document.contentType === "application/pdf") {
    return { url, content: "", type: "pdf" }
  }

  return {
    content: document.documentElement.outerHTML,
    url,
    type: "html"
  }
}

type TabSnapshot = {
  url: string
  content: string
  type: string
}

export const getDataFromCurrentTab = async () => {
  const result = new Promise<TabSnapshot>((resolve) => {
    if (isChromiumTarget) {
      chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
        const tab = tabs[0]

        const data = await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: _getHtml
        })

        if (data.length > 0) {
          resolve(data[0].result as TabSnapshot)
        }
      })
    } else {
      browser.tabs
        .query({ active: true, currentWindow: true })
        .then(async (tabs) => {
          const tab = tabs[0]
          try {
            const data = await browser.scripting.executeScript({
              target: { tabId: tab.id },
              func: _getHtml
            })

            if (data.length > 0) {
              resolve(data[0].result as TabSnapshot)
            }
          } catch (e) {
            console.error("error", e)
            // this is a weird method but it works
            if (isFirefoxTarget) {
              // all I need is to get the pdf url but somehow
              // firefox won't allow extensions to run content scripts on pdf https://bugzilla.mozilla.org/show_bug.cgi?id=1454760
              // so I set up a weird method to fix this issue by asking tab to give the url
              // and then I can get the pdf url
              const result: TabSnapshot = {
                url: tab.url,
                content: "",
                type: "pdf"
              }
              resolve(result)
            }
          }
        })
    }
  })

  const { content, type, url } = await result

  if (type === "pdf") {
    return {
      url,
      content: "",
      pdf: [],
      type: "pdf"
    }
  }
  // Twitter parsing pulls cheerio into its module; load it only when the
  // URL could actually be a tweet/timeline so ordinary pages never pay for
  // it in the eagerly loaded sidepanel chunk.
  if (/twitter\.com|x\.com/.test(url)) {
    const { isTwitterTimeline, isTweet, parseTwitterTimeline, parseTweet } =
      await import("@/parser/twitter")
    if (isTwitterTimeline(url)) {
      const data = parseTwitterTimeline(content)
      return {
        url,
        content: data,
        type: "html",
        pdf: []
      }
    } else if (isTweet(url)) {
      const data = parseTweet(content)
      return {
        url,
        content: data,
        type: "html",
        pdf: []
      }
    }
  }
  if (isGoogleDocs(url)) {
    const data = await parseGoogleDocs()
    if (data) {
      return {
        url,
        content: cleanUnwantedUnicode(data),
        type: "html",
        pdf: []
      }
    }
  }
  // Loaded lazily so cheerio/Readability/Turndown stay out of the eagerly
  // loaded sidepanel chunk (real split chunk in ESM builds).
  const { defaultExtractContent } = await import("@/parser/default")
  const data = defaultExtractContent(content)
  return { url, content: data, type, pdf: [] }
}

export const getContentFromCurrentTab = async (isUsingVS: boolean) => {
  const data = await getDataFromCurrentTab()

  if (isUsingVS) {
    return data
  }

  return data
}
