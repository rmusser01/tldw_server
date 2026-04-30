from __future__ import annotations

from typing import Any

try:
    from playwright.sync_api import Page, sync_playwright
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
except ImportError:  # pragma: no cover - optional dependency guard
    Page = Any  # type: ignore[misc,assignment]
    PlaywrightTimeoutError = TimeoutError  # type: ignore[misc,assignment]
    sync_playwright = None

CAPTCHA_FINGERPRINTS: dict[str, list[str]] = {
    "reCAPTCHA": ["google.com/recaptcha", "recaptcha/api.js", "g-recaptcha"],
    "hCaptcha": ["hcaptcha.com", "hcaptcha-box", "h-captcha"],
    "Cloudflare Turnstile": ["challenges.cloudflare.com/turnstile", "cf-turnstile"],
}


def _scan_for_captcha_fingerprints(page: Page, network_requests: list[str]) -> str | None:
    """
    Scan the page's HTML and network requests for known CAPTCHA signatures.

    Returns the name of the detected CAPTCHA provider or ``None``.
    """
    html_content = page.content().lower()
    all_evidence = network_requests + [html_content]

    for provider, patterns in CAPTCHA_FINGERPRINTS.items():
        for pattern in patterns:
            for evidence in all_evidence:
                if pattern in evidence:
                    return provider
    return None


def detect_captcha(url: str) -> dict[str, Any]:
    """
    Analyze the URL to detect the presence and type of CAPTCHA.
    """
    if sync_playwright is None:
        return {
            "status": "error",
            "message": "playwright is not installed; install the 'scrape-analyzers[browser]' extra.",
            "error_code": "missing_dependency",
        }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                captured_requests: list[str] = []

                def capture_request(request) -> None:
                    captured_requests.append(request.url.lower())

                page.on("request", capture_request)

                try:
                    page.goto(url, wait_until="load", timeout=30_000)
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except Exception as navigation_error:
                    _ = navigation_error  # continue detector heuristics with partial page state

                page.wait_for_timeout(2_000)

                captcha_on_load = _scan_for_captcha_fingerprints(page, captured_requests)
                if captcha_on_load:
                    return {
                        "status": "success",
                        "captcha_detected": True,
                        "captcha_type": captcha_on_load,
                        "trigger_condition": "on page load",
                    }

                captured_requests.clear()
                for _ in range(10):
                    page.reload(wait_until="domcontentloaded")

                captcha_after_burst = _scan_for_captcha_fingerprints(page, captured_requests)
                if captcha_after_burst:
                    return {
                        "status": "success",
                        "captcha_detected": True,
                        "captcha_type": captcha_after_burst,
                        "trigger_condition": "after burst of requests",
                    }

                return {"status": "success", "captcha_detected": False}
            finally:
                browser.close()
    except PlaywrightTimeoutError:
        return {"status": "error", "message": "Page load timed out.", "error_code": "timeout"}
    except Exception as exc:  # pragma: no cover - defensive catch
        return {"status": "error", "message": "Captcha detection failed."}
