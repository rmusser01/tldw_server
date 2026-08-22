import builtins
import importlib
import importlib.util
import io
import json
import zipfile

import pytest

from tldw_Server_API.app.core.Slides.slides_export import (
    SlidesAssetsMissingError,
    SlidesExportInputError,
    _normalize_pdf_options,
    _sanitize_custom_css,
    _sanitize_markdown,
    export_presentation_bundle,
    export_presentation_json,
    export_presentation_markdown,
)
from tldw_Server_API.app.core.Slides.visual_style_resolver import resolve_builtin_visual_style

_SAMPLE_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAn8B9XgU1b0AAAAASUVORK5CYII="


def _build_assets(tmp_path):
    assets_dir = tmp_path / "revealjs"
    (assets_dir / "plugin" / "notes").mkdir(parents=True)
    (assets_dir / "theme").mkdir(parents=True)
    (assets_dir / "reveal.js").write_text("// reveal.js", encoding="utf-8")
    (assets_dir / "reveal.css").write_text("/* reveal.css */", encoding="utf-8")
    (assets_dir / "plugin" / "notes" / "notes.js").write_text("// notes", encoding="utf-8")
    (assets_dir / "theme" / "black.css").write_text("/* theme */", encoding="utf-8")
    (assets_dir / "theme" / "night.css").write_text("/* theme */", encoding="utf-8")
    (assets_dir / "LICENSE.revealjs.txt").write_text("license", encoding="utf-8")
    return assets_dir


def test_export_bundle_includes_assets(tmp_path):
    assets_dir = _build_assets(tmp_path)
    slides = [
        {
            "order": 0,
            "layout": "title",
            "title": "Deck",
            "content": "",
            "speaker_notes": None,
            "metadata": {},
        },
        {
            "order": 1,
            "layout": "content",
            "title": "Slide",
            "content": "Hello",
            "speaker_notes": "Notes",
            "metadata": {
                "images": [
                    {
                        "mime": "image/png",
                        "data_b64": _SAMPLE_PNG_B64,
                        "alt": "Logo",
                        "width": 64,
                        "height": 64,
                    }
                ]
            },
        },
    ]
    bundle = export_presentation_bundle(
        title="Deck",
        slides=slides,
        theme="black",
        settings={"controls": True},
        custom_css=".reveal { color: red; }",
        assets_dir=assets_dir,
    )
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = set(zf.namelist())
        assert "index.html" in names
        assert "assets/reveal/reveal.js" in names
        assert "assets/reveal/reveal.css" in names
        assert "assets/reveal/theme/black.css" in names
        assert "assets/reveal/plugin/notes/notes.js" in names
        assert "LICENSE.revealjs.txt" in names
        assert "assets/custom.css" in names
        index_html = zf.read("index.html").decode("utf-8")
        assert "assets/custom.css" in index_html
        assert "data:image/png;base64," in index_html
        assert 'alt="Logo"' in index_html


def test_export_bundle_escapes_settings_for_inline_script(tmp_path):
    assets_dir = _build_assets(tmp_path)
    bundle = export_presentation_bundle(
        title="Deck",
        slides=[
            {
                "order": 0,
                "layout": "content",
                "title": "Slide",
                "content": "Hello",
                "speaker_notes": None,
                "metadata": {},
            }
        ],
        theme="black",
        settings={"transition": "</script><script>alert(1)</script>", "controls": True},
        custom_css=None,
        assets_dir=assets_dir,
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        index_html = zf.read("index.html").decode("utf-8")

    assert "</script><script>alert(1)</script>" not in index_html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in index_html


def test_sanitize_markdown_uses_bleach_without_css_sanitizer():
    html = _sanitize_markdown("- one")

    assert "<ul>" in html
    assert "<li>one</li>" in html
    assert "&lt;ul" not in html


def test_sanitize_custom_css_preserves_safe_selector_stylesheet():
    css = ".reveal .slides section { color: red; background-color: #fff; }"

    sanitized = _sanitize_custom_css(css)

    assert sanitized is not None
    assert ".reveal .slides section" in sanitized
    assert "color: red;" in sanitized
    assert "background-color: #fff;" in sanitized


@pytest.mark.parametrize(
    "css",
    [
        r".slide { background: u\72l(https://example.invalid/x); }",
        r".slide { background: var(--fallback, u\72l(https://example.invalid/x)); }",
    ],
)
def test_sanitize_custom_css_rejects_escaped_and_nested_url_tokens(css):
    with pytest.raises(SlidesExportInputError, match="custom_css_url_blocked"):
        _sanitize_custom_css(css)


@pytest.mark.parametrize("missing_name", ["CSSSanitizer", "tinycss2"])
def test_sanitize_custom_css_fails_closed_when_sanitizer_dependency_is_missing(
    monkeypatch,
    missing_name,
):
    import tldw_Server_API.app.core.Slides.slides_export as slides_export_module

    monkeypatch.setattr(slides_export_module, missing_name, None)

    assert slides_export_module._sanitize_custom_css(".slide { color: red; }") is None


def test_sanitize_custom_css_revalidates_sanitizer_output(monkeypatch):
    import tldw_Server_API.app.core.Slides.slides_export as slides_export_module

    class _InjectingSanitizer:
        def __init__(self, **_kwargs):
            pass

        def sanitize_css(self, _css):
            return "background: url(https://example.invalid/injected);"

    monkeypatch.setattr(slides_export_module, "CSSSanitizer", _InjectingSanitizer)

    with pytest.raises(SlidesExportInputError, match="custom_css_url_blocked"):
        slides_export_module._sanitize_custom_css(".slide { color: red; }")


def test_slides_export_reraises_unexpected_css_sanitizer_import_errors(monkeypatch):
    import tldw_Server_API.app.core.Slides.slides_export as slides_export_module

    original_import = builtins.__import__
    spec = importlib.util.spec_from_file_location(
        "tldw_Server_API.app.core.Slides._slides_export_import_probe",
        slides_export_module.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    probe_module = importlib.util.module_from_spec(spec)

    def _raise_for_css_sanitizer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "bleach.css_sanitizer":
            raise RuntimeError("unexpected import failure")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(builtins, "__import__", _raise_for_css_sanitizer)
        with pytest.raises(RuntimeError, match="unexpected import failure"):
            spec.loader.exec_module(probe_module)


@pytest.mark.unit
def test_export_bundle_stamps_style_hooks_and_includes_builtin_pack_css(tmp_path):
    assets_dir = _build_assets(tmp_path)
    resolved_style = resolve_builtin_visual_style("notebooklm-blueprint")
    assert resolved_style is not None

    bundle = export_presentation_bundle(
        title="Deck",
        slides=[
            {
                "order": 0,
                "layout": "content",
                "title": "Styled",
                "content": "Blueprint summary",
                "speaker_notes": None,
                "metadata": {},
            }
        ],
        theme="night",
        settings=resolved_style.snapshot["resolution"]["resolved_settings"],
        custom_css=resolved_style.appearance["custom_css"],
        visual_style_snapshot=resolved_style.snapshot,
        assets_dir=assets_dir,
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        index_html = zf.read("index.html").decode("utf-8")
        custom_css = zf.read("assets/custom.css").decode("utf-8")

    assert 'data-visual-style="notebooklm-blueprint"' in index_html
    assert 'data-style-pack="technical_grid"' in index_html
    assert '[data-style-pack="technical_grid"]' in custom_css
    assert "--surface: #0f172a;" in custom_css
    assert "url(" not in custom_css.lower()


def test_export_bundle_missing_assets(tmp_path):
    assets_dir = tmp_path / "missing"
    assets_dir.mkdir()
    with pytest.raises(SlidesAssetsMissingError):
        export_presentation_bundle(
            title="Deck",
            slides=[],
            theme="black",
            settings=None,
            custom_css=None,
            assets_dir=assets_dir,
        )


def test_export_bundle_blocks_custom_css_url(tmp_path):
    assets_dir = _build_assets(tmp_path)
    with pytest.raises(SlidesExportInputError):
        export_presentation_bundle(
            title="Deck",
            slides=[],
            theme="black",
            settings=None,
            custom_css="@import url('https://example.com')",
            assets_dir=assets_dir,
        )


def test_export_markdown_theme_mapping():
    slides = [
        {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
    ]
    md = export_presentation_markdown(title="Deck", slides=slides, theme="black")
    assert "marp: true" in md
    assert "theme: default" in md


def test_export_markdown_marp_override():
    slides = [
        {"order": 0, "layout": "title", "title": "Deck", "content": "", "speaker_notes": None, "metadata": {}},
    ]
    md = export_presentation_markdown(title="Deck", slides=slides, theme="black", marp_theme="gaia")
    assert "theme: gaia" in md


def test_export_markdown_includes_images():
    slides = [
        {
            "order": 0,
            "layout": "content",
            "title": "Slide",
            "content": "Hello",
            "speaker_notes": None,
            "metadata": {
                "images": [
                    {
                        "mime": "image/png",
                        "data_b64": _SAMPLE_PNG_B64,
                        "alt": "Logo",
                    }
                ]
            },
        },
    ]
    md = export_presentation_markdown(title="Deck", slides=slides, theme="black")
    assert "![Logo](data:image/png;base64," in md


@pytest.mark.unit
def test_export_bundle_renders_structured_visual_blocks_without_duplicate_text_fallback(tmp_path):
    assets_dir = _build_assets(tmp_path)
    fallback_lines = [
        "- 1776: Declaration - American independence declared",
        "- Scope: federal, state",
        "1. Capture - Gather evidence",
        "- Revenue: $5M - FY2025",
    ]

    bundle = export_presentation_bundle(
        title="Deck",
        slides=[
            {
                "order": 0,
                "layout": "content",
                "title": "Structured",
                "content": "\n".join(fallback_lines),
                "speaker_notes": None,
                "metadata": {
                    "visual_blocks": [
                        {
                            "type": "timeline",
                            "items": [
                                {
                                    "label": "1776",
                                    "title": "Declaration",
                                    "description": "American independence declared",
                                }
                            ],
                        },
                        {
                            "type": "comparison_matrix",
                            "rows": [
                                {
                                    "label": "Scope",
                                    "values": ["federal", "state"],
                                }
                            ],
                        },
                        {
                            "type": "process_flow",
                            "steps": [
                                {
                                    "title": "Capture",
                                    "description": "Gather evidence",
                                }
                            ],
                        },
                        {
                            "type": "stat_group",
                            "items": [
                                {
                                    "label": "Revenue",
                                    "value": "$5M",
                                    "context": "FY2025",
                                }
                            ],
                        },
                    ]
                },
            }
        ],
        theme="black",
        settings=None,
        custom_css=None,
        assets_dir=assets_dir,
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        index_html = zf.read("index.html").decode("utf-8")

    assert 'data-visual-block-type="timeline"' in index_html
    assert 'data-visual-block-type="comparison_matrix"' in index_html
    assert 'data-visual-block-type="process_flow"' in index_html
    assert 'data-visual-block-type="stat_group"' in index_html
    assert "1776" in index_html
    assert "Declaration" in index_html
    assert "Revenue" in index_html
    for line in fallback_lines:
        assert line not in index_html


@pytest.mark.unit
def test_export_markdown_preserves_text_fallback_for_visual_blocks():
    slides = [
        {
            "order": 0,
            "layout": "content",
            "title": "Timeline",
            "content": "- 1776: Declaration - American independence declared",
            "speaker_notes": None,
            "metadata": {
                "visual_blocks": [
                    {
                        "type": "timeline",
                        "items": [
                            {
                                "label": "1776",
                                "title": "Declaration",
                                "description": "American independence declared",
                            }
                        ],
                    }
                ]
            },
        }
    ]

    md = export_presentation_markdown(title="Deck", slides=slides, theme="black")

    assert "1776" in md
    assert "American independence declared" in md


@pytest.mark.unit
def test_export_markdown_keeps_text_fallback_for_supported_visual_blocks():
    slide_content = "\n".join(
        [
            "- 1776: Declaration - American independence declared",
            "- Scope: federal, state",
            "1. Capture - Gather evidence",
            "- Revenue: $5M - FY2025",
        ]
    )
    slides = [
        {
            "order": 0,
            "layout": "content",
            "title": "Structured",
            "content": slide_content,
            "speaker_notes": None,
            "metadata": {
                "visual_blocks": [
                    {
                        "type": "timeline",
                        "items": [
                            {
                                "label": "1776",
                                "title": "Declaration",
                                "description": "American independence declared",
                            }
                        ],
                    },
                    {
                        "type": "comparison_matrix",
                        "rows": [{"label": "Scope", "values": ["federal", "state"]}],
                    },
                    {
                        "type": "process_flow",
                        "steps": [{"title": "Capture", "description": "Gather evidence"}],
                    },
                    {
                        "type": "stat_group",
                        "items": [{"label": "Revenue", "value": "$5M", "context": "FY2025"}],
                    },
                ]
            },
        }
    ]

    md = export_presentation_markdown(title="Deck", slides=slides, theme="black")

    assert slide_content in md
    assert 'data-visual-block-type="' not in md


def test_export_bundle_resolves_output_asset_ref(tmp_path, monkeypatch):
    assets_dir = _build_assets(tmp_path)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_export.resolve_slide_asset",
        lambda asset_ref, **kwargs: {
            "asset_ref": asset_ref,
            "mime": "image/png",
            "data_b64": _SAMPLE_PNG_B64,
            "alt": "Cover",
        },
    )
    slides = [
        {
            "order": 0,
            "layout": "content",
            "title": "Slide",
            "content": "Hello",
            "speaker_notes": None,
            "metadata": {"images": [{"asset_ref": "output:123", "alt": "Cover"}]},
        },
    ]

    bundle = export_presentation_bundle(
        title="Deck",
        slides=slides,
        theme="black",
        settings=None,
        custom_css=None,
        assets_dir=assets_dir,
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        index_html = zf.read("index.html").decode("utf-8")
        assert "data:image/png;base64," in index_html
        assert 'alt="Cover"' in index_html


def test_export_markdown_resolves_output_asset_ref(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_export.resolve_slide_asset",
        lambda asset_ref, **kwargs: {
            "asset_ref": asset_ref,
            "mime": "image/png",
            "data_b64": _SAMPLE_PNG_B64,
            "alt": "Cover",
        },
    )
    slides = [
        {
            "order": 0,
            "layout": "content",
            "title": "Slide",
            "content": "Hello",
            "speaker_notes": "Narration",
            "metadata": {"images": [{"asset_ref": "output:123", "alt": "Cover"}]},
        }
    ]

    md = export_presentation_markdown(title="Deck", slides=slides, theme="black")

    assert "![Cover](data:image/png;base64," in md


def test_export_markdown_rejects_invalid_image():
    slides = [
        {
            "order": 0,
            "layout": "content",
            "title": "Slide",
            "content": "Hello",
            "speaker_notes": None,
            "metadata": {"images": [{"mime": "image/png", "data_b64": "not-base64"}]},
        },
    ]
    with pytest.raises(SlidesExportInputError):
        export_presentation_markdown(title="Deck", slides=slides, theme="black")


def test_normalize_pdf_options_requires_width_height():
    with pytest.raises(SlidesExportInputError):
        _normalize_pdf_options({"width": "10in"})


def test_normalize_pdf_options_rejects_invalid_format():
    with pytest.raises(SlidesExportInputError):
        _normalize_pdf_options({"format": "!!bad!!"})


def test_export_json_preserves_html_discriminator_and_unicode_source():
    payload = {
        "id": "html-deck",
        "content_kind": "standalone_html",
        "title": "\u4e09",
        "html_document": "<!doctype html><title>\u4e09</title>",
        "html_sha256": "a" * 64,
        "html_bytes": 38,
        "html_slide_count": 1,
    }

    exported = export_presentation_json(payload)

    assert json.loads(exported) == payload
    assert '"content_kind": "standalone_html"' in exported
    assert "\u4e09" in exported
    assert "\\u4e09" not in exported
