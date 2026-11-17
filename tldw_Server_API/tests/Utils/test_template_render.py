from tldw_Server_API.app.core.Chat import prompt_template_manager
from tldw_Server_API.app.core.Utils import template_render


def test_templates_render_in_utils_and_the_chat_path_re_exports_them() -> None:
    assert template_render.apply_template_to_string("Hi {{ name || 'x' }}", {"name": "Ada"}) == "Hi Ada"
    assert template_render.apply_template_to_string(None, {}) == ""
    assert prompt_template_manager.apply_template_to_string is template_render.apply_template_to_string
    assert prompt_template_manager.safe_render is template_render.safe_render
