from textual.widgets import Static


class TitleBar(Static):
    """A one-line decorative title bar with emoji art."""
    def __init__(self) -> None:
        art = "✨🤖  [b]tldw-cli – LLM Command Station[/b]  📝🚀"
        super().__init__(art, markup=True, id="app-titlebar")   # markup=True !!