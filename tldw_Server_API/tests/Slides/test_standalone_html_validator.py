from __future__ import annotations

import hashlib
import traceback

import pytest

from tldw_Server_API.app.core.Slides import standalone_html_validator as validator_module
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationError,
)
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    validate_standalone_html,
)


def _document(
    *,
    title: str = "Deck",
    slides: str = '<section class="slide"><h1>Hello</h1><p>World</p></section>',
    styles: str = "body { color: var(--ink); } :root { --ink: #111; }",
    script: str = "document.addEventListener('keydown', () => {});",
    head_extra: str = "",
    body_before_script: str = "",
) -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>{styles}</style>{head_extra}</head>"
        f"<body>{slides}{body_before_script}<script>{script}</script></body></html>"
    )


def _document_with_exact_bytes(size: int) -> str:
    base = _document()
    marker = "<script>"
    insertion = base.index(marker)
    remaining = size - len(base.encode("utf-8"))
    if remaining < 0:
        raise AssertionError("requested size is below the fixture's base size")
    chunks: list[str] = []
    while remaining:
        payload = min(16_000, max(0, remaining - 7))
        if payload == 0:
            chunks.append(" " * remaining)
            remaining = 0
            break
        comment = "<!--" + ("x" * payload) + "-->"
        chunks.append(comment)
        remaining -= len(comment)
    result = base[:insertion] + "".join(chunks) + base[insertion:]
    assert len(result.encode("utf-8")) == size
    return result


def _error(document: str | bytes, **kwargs: object) -> StandaloneHtmlValidationError:
    with pytest.raises(StandaloneHtmlValidationError) as caught:
        validate_standalone_html(document, **kwargs)
    return caught.value


def test_valid_document_returns_frozen_derived_metadata_only() -> None:
    document = _document(title="  Cafe\u0301\t Deck  ")

    result = validate_standalone_html(document)

    assert result.title == "Caf\u00e9 Deck"
    assert result.slide_count == 1
    assert result.html_bytes == len(document.encode("utf-8"))
    assert result.html_sha256 == hashlib.sha256(document.encode("utf-8")).hexdigest()
    assert result.indexable_text == "Hello World"
    with pytest.raises((AttributeError, TypeError)):
        result.title = "changed"  # type: ignore[misc]
    assert not hasattr(result, "html_document")


@pytest.mark.parametrize("control", ["\x00", "\x01", "\x08", "\x0b", "\x0c", "\x1f", "\x7f", "\x80", "\x9f"])
def test_document_rejects_nul_and_non_html_whitespace_controls(control: str) -> None:
    error = _error(_document(title=f"Bad{control}Title"))
    assert error.code == "standalone_html_invalid_document"


@pytest.mark.parametrize("whitespace", ["\t", "\n", "\r"])
def test_document_accepts_html_whitespace_controls(whitespace: str) -> None:
    result = validate_standalone_html(_document(title=f"A{whitespace}B"))
    assert result.title == "A B"


def test_document_rejects_invalid_utf8_and_lone_surrogates() -> None:
    assert _error(b"\xff").code == "standalone_html_invalid_document"
    assert _error(_document(title="\ud800")).code == "standalone_html_invalid_document"


def test_document_accepts_exactly_one_mib_and_rejects_one_byte_more() -> None:
    maximum = _document_with_exact_bytes(1_048_576)
    assert validate_standalone_html(maximum).html_bytes == 1_048_576

    error = _error(_document_with_exact_bytes(1_048_577))
    assert error.code == "standalone_html_validation_budget_exceeded"
    assert error.status_code == 422


@pytest.mark.parametrize(
    "document",
    [
        "<html><head><title>x</title></head><body></body></html>",
        "<!doctype html><head><title>x</title></head><body></body>",
        "<!doctype html><html><body><section class=slide></section><script></script></body></html>",
        "<!doctype html><html><head><title>x</title></head><section class=slide></section><script></script></html>",
        "<!doctype html><html><head><title>x</title></head><body><section class=slide></section><script></script>",
        "<!doctype html><html><head><title>x</title></head><body><section class=slide></body></html>",
        _document() + _document(),
    ],
)
def test_document_requires_one_complete_explicit_document(document: str) -> None:
    assert _error(document).code == "standalone_html_invalid_document"


@pytest.mark.parametrize("count", [1, 30])
def test_slide_count_accepts_v1_bounds(count: int) -> None:
    slides = "".join(f'<section class="slide"><h1>{index}</h1></section>' for index in range(count))
    assert validate_standalone_html(_document(slides=slides)).slide_count == count


@pytest.mark.parametrize("count", [0, 31])
def test_slide_count_rejects_outside_v1_bounds(count: int) -> None:
    slides = "".join('<section class="slide"></section>' for _ in range(count))
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


def test_html_lexical_token_element_attribute_and_depth_budgets_fail_closed() -> None:
    too_many_tokens = _document(body_before_script="<!--x-->" * 50_001)
    assert _error(too_many_tokens).code == "standalone_html_validation_budget_exceeded"

    too_many_elements = _document(slides='<section class="slide">' + ("<i></i>" * 10_001) + "</section>")
    assert _error(too_many_elements).code == "standalone_html_validation_budget_exceeded"

    tags = []
    for group in range(101):
        attrs = " ".join(f"data-{group}-{index}=x" for index in range(200))
        tags.append(f"<i {attrs}></i>")
    too_many_attributes = _document(slides='<section class="slide">' + "".join(tags) + "</section>")
    assert _error(too_many_attributes).code == "standalone_html_validation_budget_exceeded"

    too_deep = _document(slides='<section class="slide">' + ("<div>" * 129) + ("</div>" * 129) + "</section>")
    assert _error(too_deep).code == "standalone_html_validation_budget_exceeded"


def test_html_single_token_budget_rejects_65537_bytes() -> None:
    oversized_comment = "<!--" + ("x" * 65_530) + "-->"
    error = _error(_document(body_before_script=oversized_comment))
    assert error.code == "standalone_html_validation_budget_exceeded"


@pytest.mark.parametrize(
    "script_fragment",
    [
        "fetch('/x')",
        "new XMLHttpRequest()",
        "new WebSocket('wss://x')",
        "new Worker('x.js')",
        "navigator.serviceWorker.register('sw.js')",
        "localStorage.setItem('x','y')",
        "sessionStorage.x = 1",
        "indexedDB.open('x')",
        "window.open('about:blank')",
        "location.href = '/x'",
        "//# sourceMappingURL=x.js.map",
    ],
)
def test_script_static_active_resource_sinks_are_rejected(script_fragment: str) -> None:
    assert _error(_document(script=script_fragment)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.replace("<script>", '<script type="text/javascript">', 1),
        lambda doc: doc.replace("<script>", '<script type="module">', 1),
        lambda doc: doc.replace("<script>", '<script src="x.js">', 1),
        lambda doc: doc.replace("</script></body>", "</script><div></div></body>"),
        lambda doc: doc.replace("<script>", "<script></script><script>", 1),
    ],
)
def test_only_one_attribute_free_classic_script_as_final_body_element_is_allowed(mutation) -> None:
    assert _error(mutation(_document())).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "fragment",
    [
        '<a href="#x">x</a>',
        '<img src="x">',
        '<div style="color:red">x</div>',
        '<button onclick="x()">x</button>',
        "<form></form>",
        "<iframe></iframe>",
        "<frame>",
        "<object></object>",
        "<embed>",
        '<base href="/">',
        '<meta http-equiv="refresh" content="1; url=/">',
        '<link rel="stylesheet" href="x.css">',
        "<video></video>",
        "<audio></audio>",
        '<svg><image href="data:image/png;base64,x"/></svg>',
        '<svg><a xlink:href="https://example.com">x</a></svg>',
        '<math><a href="https://example.com">x</a></math>',
    ],
)
def test_html_active_and_resource_sinks_are_rejected(fragment: str) -> None:
    slides = f'<section class="slide"><h1>Safe</h1>{fragment}</section>'
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "styles",
    [
        '@import "theme.css";',
        "body { background: url(data:image/png;base64,x); }",
        "body { background: u\\72l(https://example.com/x); }",
        "@font-face { font-family: x; src: url(x.woff2); }",
    ],
)
def test_css_resource_semantics_are_rejected_after_tinycss2_decoding(styles: str) -> None:
    assert _error(_document(styles=styles)).code == "standalone_html_invalid_document"


def test_css_stylesheet_byte_token_declaration_depth_and_error_budgets() -> None:
    # _document contributes one style element; 63 extras exercise the limit of 64.
    style_elements = "".join("<style>a{color:red}</style>" for _ in range(63))
    assert validate_standalone_html(_document(styles="", head_extra=style_elements)).slide_count == 1
    assert (
        _error(_document(styles="", head_extra=style_elements + "<style>b{color:blue}</style>")).code
        == "standalone_html_validation_budget_exceeded"
    )

    huge_css = "/*" + ("x" * 32_000) + "*/"
    assert _error(_document(styles=huge_css * 17)).code == "standalone_html_validation_budget_exceeded"

    too_many_tokens = "a," * 50_001
    assert _error(_document(styles=too_many_tokens)).code == "standalone_html_validation_budget_exceeded"

    too_many_declarations = "a{" + ("x:0;" * 10_001) + "}"
    assert _error(_document(styles=too_many_declarations)).code == "standalone_html_validation_budget_exceeded"

    long_identifier = "a" * 65_537
    assert _error(_document(styles=long_identifier + "{}")).code == "standalone_html_validation_budget_exceeded"

    too_deep = ("a{" * 65) + ("}" * 65)
    assert _error(_document(styles=too_deep)).code == "standalone_html_validation_budget_exceeded"

    assert _error(_document(styles="a{color: red")).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "title",
    [
        "   ",
        "x" * 201,
        "\U0001f600" * 129,
        "bad\u202etitle",
        "bad\u2066title",
        "bad\u200ftitle",
    ],
)
def test_title_rejects_blank_scalar_byte_and_bidi_boundaries(title: str) -> None:
    assert _error(_document(title=title)).code == "standalone_html_invalid_document"


def test_title_accepts_exact_scalar_and_byte_boundaries() -> None:
    assert len(validate_standalone_html(_document(title="x" * 200)).title) == 200
    assert len(validate_standalone_html(_document(title=("\u00e9" * 200))).title.encode("utf-8")) == 400


def test_notes_cardinality_depends_on_validation_context() -> None:
    one_note = '<section class="slide"><h1>A</h1><aside class="notes">Private</aside></section>'
    no_note = '<section class="slide"><h1>A</h1></section>'
    duplicate = '<section class="slide"><div class="notes"></div><div class="notes"></div></section>'
    misplaced = '<div class="notes"></div><section class="slide"><h1>A</h1></section>'

    assert validate_standalone_html(_document(slides=one_note)).slide_count == 1
    assert validate_standalone_html(_document(slides=no_note)).slide_count == 1
    assert validate_standalone_html(_document(slides=one_note), delivery_style="speaker-led").slide_count == 1
    assert _error(_document(slides=no_note), delivery_style="speaker-led").code == "standalone_html_invalid_document"
    assert validate_standalone_html(_document(slides=no_note), delivery_style="self-guided").slide_count == 1
    assert _error(_document(slides=one_note), delivery_style="self-guided").code == "standalone_html_invalid_document"
    assert _error(_document(slides=duplicate)).code == "standalone_html_invalid_document"
    assert _error(_document(slides=misplaced)).code == "standalone_html_invalid_document"


def test_semantic_text_is_iterative_bounded_and_excludes_noncontent_subtrees() -> None:
    nested = "<span>" * 100 + "Deep" + "</span>" * 100
    slides = (
        '<section class="slide"><div class="deck-header"><p>Header</p></div>'
        f"<h1>Visible {nested}</h1><p>Paragraph <code>code</code></p>"
        "<ul><li>Item</li></ul><table><caption>Caption</caption><tr><th>Head</th><td>Cell</td></tr></table>"
        "<blockquote>Quote</blockquote><pre>Pre</pre><figure><figcaption>Figure</figcaption></figure>"
        '<aside class="notes"><p>Private</p></aside><div class="slide-number"><p>9</p></div>'
        "<script>ignored()</script><style>.ignored{}</style><template><p>Template</p></template>"
        "</section>"
    )
    error = _error(_document(slides=slides))
    assert error.code == "standalone_html_invalid_document"  # extra script is rejected before extraction

    slides = slides.replace("<script>ignored()</script>", "").replace("<style>.ignored{}</style>", "")
    result = validate_standalone_html(_document(slides=slides))
    assert result.indexable_text == "Visible Deep Paragraph code Item Caption Head Cell Quote Pre Figure"
    assert all(word not in result.indexable_text for word in ("Header", "Private", "Template"))

    long_blocks = "".join(f"<p>{'x' * 62_501}</p>" for _ in range(4))
    truncated = validate_standalone_html(_document(slides=f'<section class="slide">{long_blocks}</section>'))
    assert len(truncated.indexable_text) == 250_000


def test_failures_are_fixed_bounded_and_source_redacted() -> None:
    secret = "TOP-SECRET-HTML-SOURCE"
    error = _error(_document(slides=f'<section class="slide"><img src="{secret}"></section>'))

    rendered = str(error)
    assert error.code in {"standalone_html_invalid_document", "standalone_html_validation_budget_exceeded"}
    assert len(rendered) <= 128
    assert secret not in rendered
    assert secret not in repr(error)


def test_failures_drop_source_bearing_context_and_traceback_details() -> None:
    secret = b"TOP-SECRET-DECODE-CONTEXT"
    decode_error = _error(secret + b"\xff")
    assert decode_error.__context__ is None
    assert secret.decode("ascii") not in "".join(traceback.format_exception(decode_error))

    parser_secret = "TOP-SECRET-PARSER-ATTRIBUTE"
    malformed = _document(
        slides=('<section class="slide"><div ' f'{parser_secret}="one" {parser_secret}="two"></div></section>')
    )
    parser_error = _error(malformed)
    assert parser_error.__context__ is None
    assert parser_secret not in "".join(traceback.format_exception(parser_error))
    assert parser_error.line is None or 1 <= parser_error.line <= 1_000_000
    assert parser_error.column is None or 1 <= parser_error.column <= 1_000_000


@pytest.mark.parametrize("separator", ["\u00a0", "&nbsp;"])
def test_class_semantics_use_html_ascii_whitespace_only(separator: str) -> None:
    disguised_slide = f'<section class="slide{separator}x"><h1>Visible</h1></section>'
    assert _error(_document(slides=disguised_slide)).code == "standalone_html_invalid_document"

    disguised_notes = (
        '<section class="slide"><h1>Visible</h1>' f'<aside class="notes{separator}x">Private</aside></section>'
    )
    assert (
        _error(
            _document(slides=disguised_notes),
            delivery_style="speaker-led",
        ).code
        == "standalone_html_invalid_document"
    )

    ascii_separated = '<section class="x\tslide y"><h1>Visible</h1></section>'
    assert validate_standalone_html(_document(slides=ascii_separated)).slide_count == 1


@pytest.mark.parametrize(
    "fragment",
    [
        '<svg><rect filter="url(https://example.com/filter.svg#x)"/></svg>',
        '<svg><rect fill="url(data:image/svg+xml,x)"/></svg>',
        '<svg xml:base="https://example.com/"><rect/></svg>',
        '<svg><set attributeName="href" to="https://example.com/x"/></svg>',
        '<math><annotation definitionURL="https://example.com/x">x</annotation></math>',
    ],
)
def test_namespace_aware_resource_attributes_are_rejected(fragment: str) -> None:
    slides = f'<section class="slide"><h1>Safe</h1>{fragment}</section>'
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "styles",
    [
        'body { background: image-set("https://example.com/a.png" 1x); }',
        'body { background: image("https://example.com/a.png"); }',
        'body { background: -webkit-image-set("https://example.com/a.png" 1x); }',
    ],
)
def test_css_string_resource_functions_are_rejected(styles: str) -> None:
    assert _error(_document(styles=styles)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "script",
    [
        "fetch ('/x')",
        "new Worker ('x.js')",
        "navigator.serviceWorker . register ('sw.js')",
        "import ('module.js')",
        "window.location = '/x'",
        "document.cookie = 'x=y'",
        "caches.open ('x')",
    ],
)
def test_obvious_spaced_script_sinks_are_diagnostically_rejected(script: str) -> None:
    assert _error(_document(script=script)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "script",
    [
        "const value = `${fetch ('/x')}`;",
        "const value = `${new Worker('x.js')}`;",
        "const value = `safe ${navigator.serviceWorker.register('sw.js')}`;",
    ],
)
def test_template_interpolation_script_sinks_are_rejected(script: str) -> None:
    assert _error(_document(script=script)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "script",
    [
        'globalThis["fetch"]("/x")',
        'new globalThis["Worker"]("x.js")',
        'navigator["serviceWorker"]["register"]("sw.js")',
        'const value = `${ /}/.test("x") && fetch("/x") }`;',
    ],
)
def test_static_bracket_and_template_regex_script_sinks_are_rejected(script: str) -> None:
    assert _error(_document(script=script)).code == "standalone_html_invalid_document"


def test_raw_template_text_is_not_treated_as_executable_script() -> None:
    script = "const help = `fetch('/example') is documented text`;"
    assert validate_standalone_html(_document(script=script)).slide_count == 1


def test_character_reference_budget_fails_before_html5lib(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = "".join(f"<p>{'&amp;' * 10_000}</p>" for _ in range(6))
    document = _document(slides=f'<section class="slide">{chunks}</section>')

    def parser_must_not_run(_source: str):
        raise AssertionError("html5lib received over-budget character references")

    monkeypatch.setattr(validator_module, "_parse_html", parser_must_not_run)
    error = _error(document)
    assert error.code == "standalone_html_validation_budget_exceeded"
    assert error.reason == "html_tokens"


def test_css_candidate_budget_fails_before_tinycss2(monkeypatch: pytest.MonkeyPatch) -> None:
    style_elements = "".join(f"<style>{'a{color:red}' * count}</style>" for count in (4_000, 4_000, 2_001))
    document = _document(styles="", head_extra=style_elements)

    def parser_must_not_run(*_args, **_kwargs):
        raise AssertionError("tinycss2 received over-budget declaration candidates")

    monkeypatch.setattr(validator_module.tinycss2, "parse_stylesheet", parser_must_not_run)
    error = _error(document)
    assert error.code == "standalone_html_validation_budget_exceeded"
    assert error.reason == "css_declarations"


def test_structural_roles_require_the_xhtml_namespace() -> None:
    foreign_slide = '<svg><section class="slide"><text>Not an HTML slide</text></section></svg>'
    assert _error(_document(slides=foreign_slide)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize("trailing", ["TRAIL", "&nbsp;", "<!--after-->"])
def test_final_script_rejects_meaningful_text_or_comments_after_it(trailing: str) -> None:
    document = _document().replace("</script></body>", f"</script>{trailing}</body>")
    assert _error(document).code == "standalone_html_invalid_document"


def test_semantic_extraction_preserves_inline_and_comment_tail_adjacency() -> None:
    slides = '<section class="slide"><p>co<em>op</em>erate</p>' "<p>Hello<!--ignored-->world</p></section>"
    result = validate_standalone_html(_document(slides=slides))
    assert result.indexable_text == "cooperate Helloworld"


def test_css_aggregate_budgets_are_exercised_below_each_html_raw_token_limit() -> None:
    byte_styles = "".join(f"<style>/*{'x' * 31_996}*/</style>" for _ in range(17))
    byte_error = _error(_document(styles="", head_extra=byte_styles))
    assert byte_error.reason == "css_bytes"

    token_styles = "".join(f"<style>{'a,' * 20_000}</style>" for _ in range(3))
    token_error = _error(_document(styles="", head_extra=token_styles))
    assert token_error.reason == "css_tokens"


def test_direct_css_and_counting_tree_boundaries_abort_inside_parser_layers() -> None:
    with pytest.raises(validator_module._BudgetExceeded) as css_token:
        validator_module._preflight_css(["a" * 65_537])
    assert css_token.value.reason == "css_token_bytes"

    too_many_elements = _document(slides='<section class="slide">' + ("<i></i>" * 10_001) + "</section>")
    with pytest.raises(validator_module._BudgetExceeded) as elements:
        validator_module._parse_html(too_many_elements)
    assert elements.value.reason == "html_elements"

    attributes = " ".join(f"data-a-{index}=x" for index in range(20_001))
    too_many_attributes = _document(slides=f'<section class="slide"><i {attributes}></i></section>')
    with pytest.raises(validator_module._BudgetExceeded) as attrs:
        validator_module._parse_html(too_many_attributes)
    assert attrs.value.reason == "html_attributes"

    too_deep = _document(slides='<section class="slide">' + ("<div>" * 129) + ("</div>" * 129) + "</section>")
    with pytest.raises(validator_module._BudgetExceeded) as depth:
        validator_module._parse_html(too_deep)
    assert depth.value.reason == "html_depth"


def test_malformed_attribute_quote_is_bounded_not_an_infinite_scan() -> None:
    malformed = _document(slides='<section class="slide"><div "bad"></div></section>')
    assert _error(malformed).code == "standalone_html_invalid_document"


def test_inline_svg_namespace_declaration_is_not_treated_as_a_resource() -> None:
    slides = (
        '<section class="slide"><h1>Safe</h1>'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect width="10" height="10" fill="#123456"/></svg></section>'
    )
    assert validate_standalone_html(_document(slides=slides)).slide_count == 1


@pytest.mark.parametrize(
    "fragment",
    [
        '<svg><script>fetch("https://example.com/x")</script></svg>',
        '<svg><style>@import url("https://example.com/x.css");</style></svg>',
        '<math><style>body{background:url("https://example.com/x")}</style></math>',
    ],
)
def test_foreign_script_and_style_elements_cannot_bypass_policy(fragment: str) -> None:
    slides = f'<section class="slide"><h1>Safe</h1>{fragment}</section>'
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "fragment",
    [
        '<math altimg="https://example.com/fallback.png"><mi>x</mi></math>',
        '<applet code="https://example.com/Evil.class"></applet>',
        '<div itemscope itemtype="https://example.com/Thing">Thing</div>',
    ],
)
def test_additional_url_bearing_attributes_and_applet_are_rejected(fragment: str) -> None:
    slides = f'<section class="slide"><h1>Safe</h1>{fragment}</section>'
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "fragment",
    [
        '<div about="relative-subject">Subject</div>',
        '<div resource="relative-resource">Resource</div>',
        '<div vocab="relative-vocabulary">Vocabulary</div>',
        '<div prefix="og: https://example.com/ns#">Prefix</div>',
        '<div profile="relative-profile">Profile</div>',
        '<div itemid="relative-item">Item</div>',
        '<svg><rect color-profile="url(https://example.com/profile.icc)"/></svg>',
    ],
)
def test_rdfa_metadata_and_svg_color_profile_resources_are_rejected(fragment: str) -> None:
    slides = f'<section class="slide"><h1>Safe</h1>{fragment}</section>'
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/resource",
        "//example.com/resource",
        "data:text/plain,resource",
        "blob:https://example.com/id",
        "url(https://example.com/resource)",
    ],
)
def test_unmistakable_url_syntax_is_rejected_in_arbitrary_attributes(value: str) -> None:
    slides = '<section class="slide"><h1>Safe</h1>' f'<div data-presentation-value="{value}">Value</div></section>'
    assert _error(_document(slides=slides)).code == "standalone_html_invalid_document"


@pytest.mark.parametrize("value", ["#label", "/label"])
def test_benign_relative_markers_in_non_resource_attributes_remain_allowed(value: str) -> None:
    slides = (
        '<section class="slide"><h1>Safe</h1>'
        f'<div aria-label="{value}">Value</div>'
        '<svg><rect fill="#123456"/></svg></section>'
    )
    assert validate_standalone_html(_document(slides=slides)).slide_count == 1


def test_html_depth_budget_counts_elements_not_comments_or_text() -> None:
    # The document wrappers and slide consume part of the 128-element budget.
    prefix = '<section class="slide">' + ("<div>" * 125)
    suffix = ("</div>" * 125) + "</section>"

    with_text = prefix + "text" + suffix
    with_comment = prefix + "<!--comment-->" + suffix

    assert validate_standalone_html(_document(slides=with_text)).slide_count == 1
    assert validate_standalone_html(_document(slides=with_comment)).slide_count == 1


def _raw_css_token(kind: str, size: int) -> str:
    if kind == "comment":
        return "/*" + ("x" * (size - 4)) + "*/"
    if kind == "string":
        return '"' + ("x" * (size - 2)) + '"'
    if kind == "function":
        return ("f" * (size - 1)) + "()"
    if kind == "url":
        return "url(" + ("/" * (size - 5)) + ")"
    raise AssertionError("unknown CSS token fixture")


@pytest.mark.parametrize("kind", ["comment", "string", "function", "url"])
def test_css_raw_lexical_token_byte_boundaries(kind: str) -> None:
    for size in (validator_module.MAX_CSS_TOKEN_BYTES - 1, validator_module.MAX_CSS_TOKEN_BYTES):
        validator_module._preflight_css([_raw_css_token(kind, size)])

    with pytest.raises(validator_module._BudgetExceeded) as caught:
        validator_module._preflight_css([_raw_css_token(kind, validator_module.MAX_CSS_TOKEN_BYTES + 1)])
    assert caught.value.reason == "css_token_bytes"


@pytest.mark.parametrize(
    "hidden_slide",
    [
        '<template><section class="slide"><h1>Hidden Template</h1></section></template>',
        '<div class="deck-header"><section class="slide"><h1>Hidden Header</h1></section></div>',
        '<div class="deck-footer"><section class="slide"><h1>Hidden Footer</h1></section></div>',
    ],
)
def test_only_slides_under_excluded_ancestors_do_not_form_a_deck(hidden_slide: str) -> None:
    error = _error(_document(slides=hidden_slide))
    assert error.code == "standalone_html_invalid_document"
    assert error.reason == "slide_count"


@pytest.mark.parametrize(
    "hidden_slide",
    [
        '<template><section class="slide"><h1>Hidden Template</h1></section></template>',
        '<div class="deck-header"><section class="slide"><h1>Hidden Header</h1></section></div>',
        '<div class="deck-footer"><section class="slide"><h1>Hidden Footer</h1></section></div>',
        '<aside class="notes"><section class="slide"><h1>Hidden Notes</h1></section></aside>',
        '<section class="slide"><h1>Hidden Nested</h1></section>',
    ],
)
def test_hidden_or_nested_slides_do_not_affect_count_or_indexable_text(hidden_slide: str) -> None:
    slides = f'<section class="slide"><h1>Visible</h1>{hidden_slide}</section>'
    result = validate_standalone_html(_document(slides=slides))
    assert result.slide_count == 1
    assert result.indexable_text == "Visible"


def test_slides_inside_ordinary_wrappers_remain_discoverable() -> None:
    slides = '<main><div class="deck"><section class="slide"><h1>Visible</h1></section></div></main>'
    result = validate_standalone_html(_document(slides=slides))
    assert result.slide_count == 1
    assert result.indexable_text == "Visible"


@pytest.mark.parametrize(
    "script",
    [
        'new window.Worker("worker.js")',
        'new globalThis.WebSocket("wss://example.invalid")',
        'window.fetch("/data")',
        'globalThis.fetch("/data")',
        'const request = fetch; request("/data")',
        'let Socket = globalThis.WebSocket; new Socket("wss://example.invalid")',
        'var request = window.fetch; request("/data")',
    ],
)
def test_qualified_and_simply_aliased_script_sinks_are_rejected(script: str) -> None:
    assert _error(_document(script=script)).reason == "script_policy"


@pytest.mark.parametrize(
    "script",
    [
        "const request = safeRequest; request();",
        "const request = () => 1; request();",
        "const workerName = 'Worker'; console.log(workerName);",
    ],
)
def test_non_sink_aliases_remain_allowed(script: str) -> None:
    assert validate_standalone_html(_document(script=script)).slide_count == 1


def test_aliased_script_sink_error_remains_source_redacted() -> None:
    secret = "TOP-SECRET-ALIASED-SCRIPT-SOURCE"
    error = _error(_document(script=f'const request = fetch; request("{secret}")'))
    rendered = "".join(traceback.format_exception(error))
    assert secret not in rendered
    assert secret not in str(error)
    assert secret not in repr(error)
