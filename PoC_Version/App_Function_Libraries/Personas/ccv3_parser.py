# ccv3_parser.py
#
#
# Imports
from typing import Any, Dict, List, Optional, Union
import re
#
# External Imports
#
# Local Imports
from App_Function_Libraries.Personas.models import Lorebook, Asset, CharacterCardV3, CharacterCardV3Data, Decorator, \
    LorebookEntry
from App_Function_Libraries.Personas.utils import validate_iso_639_1, extract_json_from_charx, parse_json_file, \
    extract_text_chunks_from_png, decode_base64
#
############################################################################################################
#
# Functions:

class CCv3ParserError(Exception):
    """Custom exception for CCv3 Parser errors."""
    pass


class CharacterCardV3Parser:
    REQUIRED_SPEC = 'chara_card_v3'
    REQUIRED_VERSION = '3.0'

    def __init__(self, input_data: Union[str, bytes], input_type: str):
        """
        Initialize the parser with input data.

        :param input_data: The input data as a string or bytes.
        :param input_type: The type of the input data: 'json', 'png', 'apng', 'charx'.
        """
        self.input_data = input_data
        self.input_type = input_type.lower()
        self.character_card: Optional[CharacterCardV3] = None

    def parse(self):
        """Main method to parse the input data based on its type."""
        if self.input_type == 'json':
            self.parse_json_input()
        elif self.input_type in ['png', 'apng']:
            self.parse_png_apng_input()
        elif self.input_type == 'charx':
            self.parse_charx_input()
        else:
            raise CCv3ParserError(f"Unsupported input type: {self.input_type}")

    def parse_json_input(self):
        """Parse JSON input directly."""
        try:
            data = parse_json_file(
                self.input_data.encode('utf-8') if isinstance(self.input_data, str) else self.input_data)
            self.character_card = self._build_character_card(data)
        except Exception as e:
            raise CCv3ParserError(f"Failed to parse JSON input: {e}")

    def parse_png_apng_input(self):
        """Parse PNG or APNG input by extracting 'ccv3' tEXt chunk."""
        try:
            text_chunks = extract_text_chunks_from_png(self.input_data)
            if 'ccv3' not in text_chunks:
                raise CCv3ParserError("PNG/APNG does not contain 'ccv3' tEXt chunk.")
            ccv3_base64 = text_chunks['ccv3']
            ccv3_json_bytes = decode_base64(ccv3_base64)
            data = parse_json_file(ccv3_json_bytes)
            self.character_card = self._build_character_card(data)
        except Exception as e:
            raise CCv3ParserError(f"Failed to parse PNG/APNG input: {e}")

    def parse_charx_input(self):
        """Parse CHARX input by extracting 'card.json' from the ZIP archive."""
        try:
            data = extract_json_from_charx(self.input_data)
            self.character_card = self._build_character_card(data)
        except Exception as e:
            raise CCv3ParserError(f"Failed to parse CHARX input: {e}")

    def _build_character_card(self, data: Dict[str, Any]) -> CharacterCardV3:
        """Build the CharacterCardV3 object from parsed data."""
        # Validate required fields
        spec = data.get('spec')
        spec_version = data.get('spec_version')
        if spec != self.REQUIRED_SPEC:
            raise CCv3ParserError(f"Invalid spec: Expected '{self.REQUIRED_SPEC}', got '{spec}'")
        if spec_version != self.REQUIRED_VERSION:
            # As per spec, should not reject but handle versions
            # For now, proceed if version is >=3.0
            try:
                version_float = float(spec_version)
                if version_float < 3.0:
                    raise CCv3ParserError(f"Unsupported spec_version: '{spec_version}' (must be >= '3.0')")
            except ValueError:
                raise CCv3ParserError(f"Invalid spec_version format: '{spec_version}'")

        data_field = data.get('data')
        if not data_field:
            raise CCv3ParserError("Missing 'data' field in CharacterCardV3 object.")

        # Extract required fields
        required_fields = ['name', 'description', 'tags', 'creator', 'character_version',
                           'mes_example', 'extensions', 'system_prompt',
                           'post_history_instructions', 'first_mes',
                           'alternate_greetings', 'personality', 'scenario',
                           'creator_notes', 'group_only_greetings']
        for field_name in required_fields:
            if field_name not in data_field:
                raise CCv3ParserError(f"Missing required field in data: '{field_name}'")

        # Parse assets
        assets_data = data_field.get('assets', [{
            'type': 'icon',
            'uri': 'ccdefault:',
            'name': 'main',
            'ext': 'png'
        }])
        assets = self._parse_assets(assets_data)

        # Parse creator_notes_multilingual
        creator_notes_multilingual = data_field.get('creator_notes_multilingual')
        if creator_notes_multilingual:
            if not isinstance(creator_notes_multilingual, dict):
                raise CCv3ParserError("'creator_notes_multilingual' must be a dictionary.")
            # Validate ISO 639-1 codes
            for lang_code in creator_notes_multilingual.keys():
                if not validate_iso_639_1(lang_code):
                    raise CCv3ParserError(f"Invalid language code in 'creator_notes_multilingual': '{lang_code}'")

        # Parse character_book
        character_book_data = data_field.get('character_book')
        character_book = self._parse_lorebook(character_book_data) if character_book_data else None

        # Build CharacterCardV3Data
        character_card_data = CharacterCardV3Data(
            name=data_field['name'],
            description=data_field['description'],
            tags=data_field['tags'],
            creator=data_field['creator'],
            character_version=data_field['character_version'],
            mes_example=data_field['mes_example'],
            extensions=data_field['extensions'],
            system_prompt=data_field['system_prompt'],
            post_history_instructions=data_field['post_history_instructions'],
            first_mes=data_field['first_mes'],
            alternate_greetings=data_field['alternate_greetings'],
            personality=data_field['personality'],
            scenario=data_field['scenario'],
            creator_notes=data_field['creator_notes'],
            character_book=character_book,
            assets=assets,
            nickname=data_field.get('nickname'),
            creator_notes_multilingual=creator_notes_multilingual,
            source=data_field.get('source'),
            group_only_greetings=data_field['group_only_greetings'],
            creation_date=data_field.get('creation_date'),
            modification_date=data_field.get('modification_date')
        )

        return CharacterCardV3(
            spec=spec,
            spec_version=spec_version,
            data=character_card_data
        )

    def _parse_assets(self, assets_data: List[Dict[str, Any]]) -> List[Asset]:
        """Parse and validate assets."""
        assets = []
        for asset_data in assets_data:
            # Validate required fields
            for field in ['type', 'uri', 'ext']:
                if field not in asset_data:
                    raise CCv3ParserError(f"Asset missing required field: '{field}'")
                if not isinstance(asset_data[field], str):
                    raise CCv3ParserError(f"Asset field '{field}' must be a string.")
            # Optional 'name'
            name = asset_data.get('name', '')
            # Validate 'ext'
            ext = asset_data['ext'].lower()
            if not re.match(r'^[a-z0-9]+$', ext):
                raise CCv3ParserError(f"Invalid file extension in asset: '{ext}'")
            # Append to assets list
            assets.append(Asset(
                type=asset_data['type'],
                uri=asset_data['uri'],
                name=name,
                ext=ext
            ))
        return assets

    def _parse_lorebook(self, lorebook_data: Dict[str, Any]) -> Lorebook:
        """Parse and validate Lorebook object."""
        # Validate Lorebook fields
        if not isinstance(lorebook_data, dict):
            raise CCv3ParserError("Lorebook must be a JSON object.")

        # Extract fields with defaults
        name = lorebook_data.get('name')
        description = lorebook_data.get('description')
        scan_depth = lorebook_data.get('scan_depth')
        token_budget = lorebook_data.get('token_budget')
        recursive_scanning = lorebook_data.get('recursive_scanning')
        extensions = lorebook_data.get('extensions', {})
        entries_data = lorebook_data.get('entries', [])

        # Parse entries
        entries = self._parse_lorebook_entries(entries_data)

        return Lorebook(
            name=name,
            description=description,
            scan_depth=scan_depth,
            token_budget=token_budget,
            recursive_scanning=recursive_scanning,
            extensions=extensions,
            entries=entries
        )

    def _parse_lorebook_entries(self, entries_data: List[Dict[str, Any]]) -> List[LorebookEntry]:
        """Parse and validate Lorebook entries."""
        entries = []
        for entry_data in entries_data:
            # Validate required fields
            for field in ['keys', 'content', 'enabled', 'insertion_order']:
                if field not in entry_data:
                    raise CCv3ParserError(f"Lorebook entry missing required field: '{field}'")
            if not isinstance(entry_data['keys'], list) or not all(isinstance(k, str) for k in entry_data['keys']):
                raise CCv3ParserError("'keys' field in Lorebook entry must be a list of strings.")
            if not isinstance(entry_data['content'], str):
                raise CCv3ParserError("'content' field in Lorebook entry must be a string.")
            if not isinstance(entry_data['enabled'], bool):
                raise CCv3ParserError("'enabled' field in Lorebook entry must be a boolean.")
            if not isinstance(entry_data['insertion_order'], (int, float)):
                raise CCv3ParserError("'insertion_order' field in Lorebook entry must be a number.")

            # Optional fields
            use_regex = entry_data.get('use_regex', False)
            constant = entry_data.get('constant')
            selective = entry_data.get('selective')
            secondary_keys = entry_data.get('secondary_keys')
            position = entry_data.get('position')
            name = entry_data.get('name')
            priority = entry_data.get('priority')
            entry_id = entry_data.get('id')
            comment = entry_data.get('comment')

            if selective and not isinstance(selective, bool):
                raise CCv3ParserError("'selective' field in Lorebook entry must be a boolean.")
            if secondary_keys:
                if not isinstance(secondary_keys, list) or not all(isinstance(k, str) for k in secondary_keys):
                    raise CCv3ParserError("'secondary_keys' field in Lorebook entry must be a list of strings.")
            if position and not isinstance(position, str):
                raise CCv3ParserError("'position' field in Lorebook entry must be a string.")

            # Parse decorators from content
            decorators = self._extract_decorators(entry_data['content'])

            # Create LorebookEntry
            entries.append(LorebookEntry(
                keys=entry_data['keys'],
                content=entry_data['content'],
                enabled=entry_data['enabled'],
                insertion_order=int(entry_data['insertion_order']),
                use_regex=use_regex,
                constant=constant,
                selective=selective,
                secondary_keys=secondary_keys,
                position=position,
                decorators=decorators,
                name=name,
                priority=priority,
                id=entry_id,
                comment=comment
            ))
        return entries

    def _extract_decorators(self, content: str) -> List[Decorator]:
        """Extract decorators from the content field."""
        decorators = []
        lines = content.splitlines()
        for line in lines:
            if line.startswith('@@'):
                decorator = self._parse_decorator_line(line)
                if decorator:
                    decorators.append(decorator)
        return decorators

    def _parse_decorator_line(self, line: str) -> Optional[Decorator]:
        """
        Parses a single decorator line.

        Example:
            @@decorator_name value
            @@@fallback_decorator value
        """
        fallback = None
        if line.startswith('@@@'):
            # Fallback decorator
            name_value = line.lstrip('@').strip()
            parts = name_value.split(' ', 1)
            name = parts[0]
            value = parts[1] if len(parts) > 1 else None
            fallback = Decorator(name=name, value=value)
            return fallback
        elif line.startswith('@@'):
            # Primary decorator
            name_value = line.lstrip('@').strip()
            parts = name_value.split(' ', 1)
            name = parts[0]
            value = parts[1] if len(parts) > 1 else None
            # Check for fallback decorators in subsequent lines
            # This assumes that fallback decorators follow immediately after the primary
            # decorator in the content
            # For simplicity, not implemented here. You can enhance this based on your needs.
            return Decorator(name=name, value=value)
        else:
            return None

    def get_character_card(self) -> Optional[CharacterCardV3]:
        """Returns the parsed CharacterCardV3 object."""
        return self.character_card

#
# End of ccv3_parser.py
############################################################################################################