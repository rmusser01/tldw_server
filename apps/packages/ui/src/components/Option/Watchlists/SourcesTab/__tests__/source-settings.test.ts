import { describe, expect, it } from "vitest"
import {
  buildSourceSettingsPayload,
  sourceSettingsToFormValues
} from "../source-settings"

describe("source-settings", () => {
  it("preserves unknown keys while serializing website scrape rules", () => {
    const payload = buildSourceSettingsPayload(
      {
        retention: { days: 30 },
        custom_identity: "source-slug",
        scrape_rules: { title_selector: "css:.old-title" }
      },
      {
        scrape_list_url: "https://example.com/news",
        scrape_item_selector: "css:article",
        scrape_link_selector: ".//a/@href",
        scrape_title_selector: "css:h2",
        scrape_summary_selector: "css:.deck",
        scrape_limit: 20,
        source_top_n: 10,
        discover_method: "frontpage"
      }
    )

    expect(payload).toEqual({
      retention: { days: 30 },
      custom_identity: "source-slug",
      scrape_rules: {
        list_url: "https://example.com/news",
        item_selector: "css:article",
        link_xpath: ".//a/@href",
        title_selector: "css:h2",
        summary_selector: "css:.deck",
        limit: 20
      },
      top_n: 10,
      discover_method: "frontpage"
    })
  })

  it("does not create noisy settings when advanced fields are empty", () => {
    expect(
      buildSourceSettingsPayload(null, {
        scrape_list_url: " ",
        scrape_item_selector: "",
        scrape_link_selector: "",
        discover_method: "auto",
        source_top_n: null
      })
    ).toBeUndefined()
  })

  it("clears empty advanced rule fields without deleting unrelated settings", () => {
    expect(
      buildSourceSettingsPayload(
        {
          retention: { days: 7 },
          scrape_rules: { item_selector: "css:article" },
          top_n: 5,
          discover_method: "frontpage"
        },
        {
          scrape_item_selector: "",
          discover_method: "auto",
          source_top_n: null
        }
      )
    ).toEqual({
      retention: { days: 7 }
    })
  })

  it("maps existing settings back into editable form values", () => {
    expect(
      sourceSettingsToFormValues({
        scrape_rules: {
          list_url: "https://example.com/news",
          item_selector: "css:article",
          link_xpath: ".//a/@href",
          title_selector: "css:h2",
          limit: 15
        },
        top_n: 15,
        discover_method: "frontpage"
      })
    ).toEqual({
      scrape_list_url: "https://example.com/news",
      scrape_item_selector: "css:article",
      scrape_link_selector: ".//a/@href",
      scrape_title_selector: "css:h2",
      scrape_summary_selector: "",
      scrape_content_selector: "",
      scrape_date_selector: "",
      scrape_guid_selector: "",
      scrape_limit: 15,
      source_top_n: 15,
      discover_method: "frontpage",
      skip_article_fetch: false
    })
  })
})
