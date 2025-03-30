# cbs_handler.py
import re
import random
from typing import List

from PoC_Version.App_Function_Libraries.Personas.models import CharacterCardV3


class CBSHandler:
    """Handles Curly Braced Syntaxes (CBS) in strings."""

    CBS_PATTERN = re.compile(r'\{\{(.*?)\}\}')

    def __init__(self, character_card: CharacterCardV3, user_display_name: str):
        self.character_card = character_card
        self.user_display_name = user_display_name

    def replace_cbs(self, text: str) -> str:
        """Replaces CBS in the given text with appropriate values."""
        def replacer(match):
            cbs_content = match.group(1).strip()
            if cbs_content.lower() == 'char':
                return self.character_card.data.nickname or self.character_card.data.name
            elif cbs_content.lower() == 'user':
                return self.user_display_name
            elif cbs_content.lower().startswith('random:'):
                options = self._split_escaped(cbs_content[7:])
                return random.choice(options) if options else ''
            elif cbs_content.lower().startswith('pick:'):
                options = self._split_escaped(cbs_content[5:])
                return random.choice(options) if options else ''
            elif cbs_content.lower().startswith('roll:'):
                return self._handle_roll(cbs_content[5:])
            elif cbs_content.lower().startswith('//'):
                return ''
            elif cbs_content.lower().startswith('hidden_key:'):
                # Placeholder for hidden_key logic
                return ''
            elif cbs_content.lower().startswith('comment:'):
                # Placeholder for comment logic
                return ''
            elif cbs_content.lower().startswith('reverse:'):
                return cbs_content[8:][::-1]
            else:
                # Unknown CBS; return as is or empty
                return ''

        return self.CBS_PATTERN.sub(replacer, text)

    def _split_escaped(self, text: str) -> List[str]:
        """Splits a string by commas, considering escaped commas."""
        return [s.replace('\\,', ',') for s in re.split(r'(?<!\\),', text)]

    def _handle_roll(self, value: str) -> str:
        """Handles the roll:N CBS."""
        value = value.lower()
        if value.startswith('d'):
            value = value[1:]
        if value.isdigit():
            return str(random.randint(1, int(value)))
        return ''

    def handle_comments(self, text: str) -> str:
        """Handles comments in CBS."""
        # Implementation depends on how comments should be displayed
        # For simplicity, remove comments
        return re.sub(r'\{\{comment:.*?\}\}', '', text)