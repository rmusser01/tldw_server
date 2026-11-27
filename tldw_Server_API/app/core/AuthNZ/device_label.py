"""Turn a User-Agent header into a short label such as "Chrome on macOS".

Only the browser and OS family are reported: never versions and never the raw
agent string, so the label is safe to store, log and show in an alert.
"""

from __future__ import annotations

_MAX_AGENT_CHARS = 512

# First match wins; order matters (Edge and Opera also say "Chrome/").
_BROWSERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("Edg/", "EdgA/", "EdgiOS/"), "Edge"),
    (("OPR/",), "Opera"),
    (("FxiOS/", "Firefox/"), "Firefox"),
    (("CriOS/", "Chrome/"), "Chrome"),
)

# iPhone/iPad and Android are checked before "Mac OS X" / "Linux", which they also contain.
_SYSTEMS: tuple[tuple[str, str], ...] = (
    ("iPhone", "iOS"),
    ("iPad", "iOS"),
    ("Android", "Android"),
    ("Windows NT", "Windows"),
    ("CrOS", "ChromeOS"),
    ("Mac OS X", "macOS"),
    ("Linux", "Linux"),
)


def device_label(user_agent: str | None) -> str:
    """Return "<Browser> on <OS>", "<Browser>" if the OS is unknown, else "Unknown browser"."""
    agent = (user_agent or "")[:_MAX_AGENT_CHARS]
    browser = None
    for needles, name in _BROWSERS:
        if any(n in agent for n in needles):
            browser = name
            break
    if browser is None and "Version/" in agent and "Safari/" in agent:
        browser = "Safari"
    if browser is None:
        return "Unknown browser"
    for needle, name in _SYSTEMS:
        if needle in agent:
            return f"{browser} on {name}"
    return browser
