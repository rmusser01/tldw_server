import { ChatDocuments } from "@/models/ChatTypes"
import { getMaxContextSize } from "@/services/kb"

const formatDocumentHeader = (title: string, url: string) => {
    const hostname = new URL(url).hostname
    return `# ${title} (${hostname}) \n\n`
}

const truncateContent = (content: string, maxLength: number): string => {
    if (content.length <= maxLength) {
        return content
    }

    // Try to truncate at word boundary
    const truncated = content.substring(0, maxLength)
    const lastSpaceIndex = truncated.lastIndexOf(' ')

    if (lastSpaceIndex > maxLength * 0.8) {
        return truncated.substring(0, lastSpaceIndex) + '...'
    }

    return truncated + '...'
}

export const getTabContents = async (documents: ChatDocuments) => {
    const result: string[] = []
    const maxContextSize = await getMaxContextSize()

    // Calculate available space per document
    const contextPerDocument = Math.floor(maxContextSize / documents.length)
    let remainingContext = maxContextSize

    for (const doc of documents) {
        try {
            if (remainingContext <= 0) break

            const pageContent = await browser.scripting.executeScript({
                target: { tabId: doc.tabId! },
                func: () => ({
                    html: document.documentElement.outerHTML,
                    title: document.title,
                    url: window.location.href,
                    isPDF: document.contentType === 'application/pdf'
                })
            })
            const content = pageContent[0].result as {
                html: string
                title: string
                url: string
                isPDF: boolean
            }
            const header = formatDocumentHeader(doc.title, doc.url)
            let extractedContent = ""

            // The specialized parsers each drag cheerio into their module;
            // load them only when the URL could actually match so ordinary
            // documents never pull them into the eager sidepanel chunk.
            if (/wikipedia\.org\/wiki\//.test(doc.url)) {
                const { isWikipedia, parseWikipedia } = await import("@/parser/wiki")
                if (isWikipedia(doc.url)) {
                    extractedContent = parseWikipedia(content.html)
                }
            } else if (/amazon\.[a-z]{2,}/i.test(doc.url)) {
                const { isAmazonURL, parseAmazonWebsite } = await import("@/parser/amazon")
                if (isAmazonURL(doc.url)) {
                    extractedContent = parseAmazonWebsite(content.html)
                }
            } else if (/twitter\.com|x\.com/.test(doc.url)) {
                const {
                    isTwitterProfile,
                    isTwitterTimeline,
                    parseTweetProfile,
                    parseTwitterTimeline
                } = await import("@/parser/twitter")
                if (isTwitterProfile(doc.url)) {
                    extractedContent = parseTweetProfile(content.html)
                } else if (isTwitterTimeline(doc.url)) {
                    extractedContent = parseTwitterTimeline(content.html)
                }
            }

            if (!extractedContent) {
                // Loaded lazily so cheerio/Readability/Turndown stay out of
                // the eagerly loaded sidepanel chunk.
                const { defaultExtractContent } = await import("@/parser/default")
                extractedContent = defaultExtractContent(content.html)
            }

            // Calculate available space for this document's content
            const headerLength = header.length
            const availableSpace = Math.min(
                contextPerDocument - headerLength,
                remainingContext - headerLength
            )

            if (availableSpace > 0) {
                const truncatedContent = truncateContent(extractedContent, availableSpace)
                const documentContent = `${header}${truncatedContent}`

                result.push(documentContent)
                remainingContext -= documentContent.length
            }
        } catch (e) {
            console.error("Error processing document:", e)
            continue
        }
    }

    return result.join("\n\n")
}
