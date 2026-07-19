from __future__ import annotations


def get_impersonate_target(user_agent: str) -> str:
    """
    Map a User-Agent string to the closest curl_cffi impersonation target.
    """
    ua_lower = user_agent.lower()

    if "edg/" in ua_lower:
        return "edge101"
    if "chrome/" in ua_lower and "edg/" not in ua_lower:
        if "android" in ua_lower:
            return "chrome131"
        return "chrome131"
    if "firefox/" in ua_lower:
        return "firefox133"
    if "safari/" in ua_lower and "chrome" not in ua_lower:
        if "iphone" in ua_lower or "ipad" in ua_lower:
            return "safari172_ios"
        return "safari155"

    return "chrome131"
