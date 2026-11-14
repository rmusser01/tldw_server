import { bgRequest } from '@/services/background-proxy'
import { db } from '@/db/dexie/schema'
import { generateID } from '@/db/dexie/helpers'
import type { ScopedRequestOptions } from './TldwApiClient'
import { requestScopeFields } from './domains/service-prompts'

/** Nonpersisting credential-free article extraction response. */
export interface PublicArticleExtractionResponse {
  status: string
  message: string
  results: {
    url: string
    title?: string | null
    content?: string | null
    ingested_at?: string | null
    extraction_successful?: boolean
    error?: string
    metadata?: Record<string, unknown>
  }[]
}

export interface ProcessOptions {
  storeLocal?: boolean
  metadata?: Record<string, any>
}

export const tldwMedia = {
  async extractPublicArticle(
    url: string,
    options?: ScopedRequestOptions
  ): Promise<PublicArticleExtractionResponse> {
    const scopeFields = requestScopeFields(options?.requestScope)
    return await bgRequest<PublicArticleExtractionResponse>({
      path: '/api/v1/media/ingest-web-content',
      method: 'POST',
      ...scopeFields,
      abortSignal: options?.signal,
      headers: { 'Content-Type': 'application/json', ...scopeFields.headers },
      body: {
        urls: [url],
        scrape_method: 'individual',
        credential_free: true,
        perform_analysis: false,
        perform_translation: false,
        perform_chunking: false,
        auto_chunking_use_llm: false,
        use_cookies: false,
        overwrite_existing: false,
        perform_rolling_summarization: false,
        perform_confabulation_check_of_analysis: false
      }
    })
  },
  async addUrl(url: string, metadata?: Record<string, any>) {
    return await bgRequest<any>({
      path: '/api/v1/media/add',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { url, ...(metadata || {}) }
    })
  },

  async processUrl(url: string, opts?: ProcessOptions) {
    // Process without storing on server
    const res = await bgRequest<any>({
      path: '/api/v1/media/process-web-scraping',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { url, ...(opts?.metadata || {}) }
    })
    if (opts?.storeLocal) {
      try {
        await db.processedMedia.add({
          id: generateID(),
          url,
          title: res?.title || res?.metadata?.title,
          content: res?.content || res?.text || '',
          metadata: res?.metadata || {},
          createdAt: Date.now()
        })
      } catch (e) {
        console.error('Failed to store processed media locally', e)
      }
    }
    return res
  }
}
