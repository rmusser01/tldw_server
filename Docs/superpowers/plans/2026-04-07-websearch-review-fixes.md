# WebSearch Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the backend WebSearch issues identified in review, tighten the endpoint failure contract, and add regression coverage for the new behavior.

**Architecture:** Keep the public `/api/v1/research/websearch` surface stable for successful and partial-success requests, but convert fatal provider failures with no results into explicit HTTP failures. Improve the WebSearch aggregation pipeline in place by preserving review IDs, falling back safely when scraping fails, and scoping the relevance circuit breaker per provider.

**Tech Stack:** FastAPI, pytest, Pydantic, loguru, existing WebSearch/Web_Scraping helpers

---

## Stage 1: Endpoint Failure Contract
**Goal:** Ensure provider/config/network failures with no results do not return `200`.
**Success Criteria:** Raw and aggregate endpoint paths return `502` on fatal phase-1 provider failure; partial-success payloads with warnings still return `200`.
**Tests:** `tldw_Server_API/tests/WebSearch/integration/test_websearch_endpoint.py`
**Status:** Not Started

## Stage 2: Relevance and Review Pipeline Safety
**Goal:** Preserve result identity through `user_review` and keep relevant results when scraping/full-page extraction fails.
**Success Criteria:** `user_review` does not renumber relevant results; relevance evaluation can use snippet/title/url fallback content and summarization falls back safely when scraping fails.
**Tests:** `tldw_Server_API/tests/WebSearch/test_websearch_core.py`, `tldw_Server_API/tests/WebScraping/test_review_selector.py`
**Status:** Not Started

## Stage 3: Provider Error Surfacing and Breaker Isolation
**Goal:** Make DuckDuckGo policy denial explicit and isolate the relevance breaker by provider.
**Success Criteria:** DuckDuckGo policy failure raises a provider error instead of silent empty results; breaker state for one provider does not suppress another provider.
**Tests:** `tldw_Server_API/tests/WebSearch/test_websearch_core.py`, `tldw_Server_API/tests/WebScraping/test_config_cache_and_limits.py`
**Status:** Not Started

## Stage 4: Verification and Security
**Goal:** Prove the touched WebSearch scope passes tests and basic security checks.
**Success Criteria:** Focused pytest slice passes; Bandit reports no new findings in touched paths.
**Tests:** Focused WebSearch pytest targets plus Bandit on touched files.
**Status:** Not Started
