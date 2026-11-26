import pytest

from tldw_Server_API.app.core.AuthNZ.device_label import device_label

CHROME_MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
SAFARI_IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"


@pytest.mark.parametrize(
    "agent,expected",
    [
        (CHROME_MAC, "Chrome on macOS"),
        (SAFARI_IPHONE, "Safari on iOS"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0", "Firefox on Windows"),
        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 Edg/130.0.0.0", "Edge on Windows"),
        ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 OPR/115.0.0.0", "Opera on Linux"),
        ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36", "Chrome on Android"),
        ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15", "Safari on macOS"),
        ("Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36", "Chrome on ChromeOS"),
        ("Mozilla/5.0 (Unknown OS) Chrome/130.0.0.0", "Chrome"),
        ("python-requests/2.32.3", "Unknown browser"),
        ("curl/8.7.1", "Unknown browser"),
        ("Unknown", "Unknown browser"),
        ("", "Unknown browser"),
        (None, "Unknown browser"),
    ],
)
def test_device_label(agent, expected):
    assert device_label(agent) == expected


def test_label_never_echoes_versions_or_the_raw_agent():
    label = device_label(CHROME_MAC)
    assert "130" not in label and "Mozilla" not in label and "10_15" not in label


def test_only_the_first_512_characters_are_read():
    assert device_label("x" * 600 + " Firefox/132.0 (Windows NT 10.0)") == "Unknown browser"
