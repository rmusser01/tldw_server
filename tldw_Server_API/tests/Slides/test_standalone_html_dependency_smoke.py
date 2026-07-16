import html5lib
import tinycss2

from tldw_Server_API.app.core.Slides.standalone_html_validator import validate_standalone_html


def test_direct_parser_dependencies_and_accepted_html_css_path() -> None:
    assert html5lib.__version__ == "1.1"
    assert tinycss2.__version__ == "1.4.0"

    result = validate_standalone_html(
        "<!doctype html><html><head><meta charset=utf-8><title>Smoke</title>"
        "<style>:root{--ink:#111}body{color:var(--ink)}</style></head>"
        '<body><section class="slide"><h1>Ready</h1></section>'
        "<script>document.addEventListener('keydown',()=>{});</script></body></html>"
    )

    assert result.title == "Smoke"
    assert result.indexable_text == "Ready"
