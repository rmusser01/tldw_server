# example_usage.py
# Description: Example usage of the CharacterCardV3Parser and CBSHandler classes
#
# Imports
import json
import base64
import os
import sys
from io import BytesIO
from zipfile import ZipFile
from PIL import Image, PngImagePlugin
#
# External Imports
#
# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tldw')))
# Local Imports
from PoC_Version.App_Function_Libraries.Personas.ccv3_parser import CharacterCardV3Parser, CCv3ParserError
from PoC_Version.App_Function_Libraries.Personas.cbs_handlers import CBSHandler
#
############################################################################################################
#
# Functions:


def main():
    # Example JSON input
    json_input = '''
    {
      "spec": "chara_card_v3",
      "spec_version": "3.0",
      "data": {
        "name": "Example Character",
        "description": "A character for demo.",
        "tags": ["demo", "test"],
        "creator": "John Doe",
        "character_version": "1.0",
        "mes_example": "Hello, this is an example.",
        "extensions": {},
        "system_prompt": "You are {{char}}.",
        "post_history_instructions": "Remember to be nice.",
        "first_mes": "Hi there!",
        "alternate_greetings": ["Hello!", "Hi!", "Hey!"],
        "personality": "Friendly and helpful.",
        "scenario": "Assisting users with tasks.",
        "creator_notes": "These are creator notes.",
        "group_only_greetings": ["Hello Group!"],
        "assets": [
            {"type": "icon", "uri": "embeded://path/to/icon.png", "name": "main", "ext": "png"},
            {"type": "background", "uri": "https://example.com/background.jpg", "name": "main", "ext": "jpg"}
        ],
        "creator_notes_multilingual": {
            "en": "These are creator notes in English.",
            "es": "Estas son notas del creador en español."
        },
        "source": ["http://source.com", "https://source.org"],
        "nickname": "ExChar",
        "creation_date": 1700000000,
        "modification_date": 1700003600
      }
    }
    '''

    # Initialize parser for JSON input
    parser = CharacterCardV3Parser(input_data=json_input, input_type='json')
    try:
        parser.parse()
        character_card = parser.get_character_card()
        print("Parsed Character Card:")
        print(json.dumps(character_card.__dict__, indent=2, default=lambda o: o.__dict__))
    except CCv3ParserError as e:
        print(f"Error parsing Character Card V3: {e}")

    # Example PNG/APNG input
    # For demonstration, we'll create a simple PNG with a 'ccv3' tEXt chunk
    # In practice, you'd load an actual PNG/APNG file
    ccv3_json = json.loads(json_input)
    ccv3_str = json.dumps(ccv3_json).encode('utf-8')
    ccv3_base64 = base64.b64encode(ccv3_str).decode('utf-8')

    png = Image.new('RGBA', (100, 100), color='red')
    meta = PngImagePlugin.PngInfo()
    meta.add_text("ccv3", ccv3_base64)
    png_bytes_io = BytesIO()
    png.save(png_bytes_io, "PNG", pnginfo=meta)
    png_bytes = png_bytes_io.getvalue()

    # Initialize parser for PNG input
    parser_png = CharacterCardV3Parser(input_data=png_bytes, input_type='png')
    try:
        parser_png.parse()
        character_card_png = parser_png.get_character_card()
        print("\nParsed Character Card from PNG:")
        print(json.dumps(character_card_png.__dict__, indent=2, default=lambda o: o.__dict__))
    except CCv3ParserError as e:
        print(f"Error parsing Character Card V3 from PNG: {e}")

    # Example CHARX input
    # For demonstration, we'll create a simple CHARX zip in memory
    charx_bytes_io = BytesIO()
    with ZipFile(charx_bytes_io, 'w') as zipf:
        zipf.writestr('card.json', json.dumps(ccv3_json))
        zipf.writestr('assets/icon.png', png_bytes)  # Adding an asset as an example
    charx_bytes = charx_bytes_io.getvalue()

    # Initialize parser for CHARX input
    parser_charx = CharacterCardV3Parser(input_data=charx_bytes, input_type='charx')
    try:
        parser_charx.parse()
        character_card_charx = parser_charx.get_character_card()
        print("\nParsed Character Card from CHARX:")
        print(json.dumps(character_card_charx.__dict__, indent=2, default=lambda o: o.__dict__))
    except CCv3ParserError as e:
        print(f"Error parsing Character Card V3 from CHARX: {e}")

    # Handling CBS
    if character_card:
        cbs_handler = CBSHandler(character_card=character_card, user_display_name="Alice")
        processed_prompt = cbs_handler.replace_cbs(character_card.data.system_prompt)
        print("\nProcessed CBS in system_prompt:")
        print(processed_prompt)


if __name__ == "__main__":
    main()

"""
Expected output:
```
Parsed Character Card:
{
  "spec": "chara_card_v3",
  "spec_version": "3.0",
  "data": {
    "name": "Example Character",
    "description": "A character for demo.",
    "tags": [
      "demo",
      "test"
    ],
    "creator": "John Doe",
    "character_version": "1.0",
    "mes_example": "Hello, this is an example.",
    "extensions": {},
    "system_prompt": "You are {{char}}.",
    "post_history_instructions": "Remember to be nice.",
    "first_mes": "Hi there!",
    "alternate_greetings": [
      "Hello!",
      "Hi!",
      "Hey!"
    ],
    "personality": "Friendly and helpful.",
    "scenario": "Assisting users with tasks.",
    "creator_notes": "These are creator notes.",
    "character_book": null,
    "assets": [
      {
        "type": "icon",
        "uri": "embeded://path/to/icon.png",
        "name": "main",
        "ext": "png"
      },
      {
        "type": "background",
        "uri": "https://example.com/background.jpg",
        "name": "main",
        "ext": "jpg"
      }
    ],
    "nickname": "ExChar",
    "creator_notes_multilingual": {
      "en": "These are creator notes in English.",
      "es": "Estas son notas del creador en español."
    },
    "source": [
      "http://source.com",
      "https://source.org"
    ],
    "group_only_greetings": [
      "Hello Group!"
    ],
    "creation_date": 1700000000,
    "modification_date": 1700003600
  }
}

Parsed Character Card from PNG:
{
  "spec": "chara_card_v3",
  "spec_version": "3.0",
  "data": {
    "name": "Example Character",
    "description": "A character for demo.",
    "tags": [
      "demo",
      "test"
    ],
    "creator": "John Doe",
    "character_version": "1.0",
    "mes_example": "Hello, this is an example.",
    "extensions": {},
    "system_prompt": "You are {{char}}.",
    "post_history_instructions": "Remember to be nice.",
    "first_mes": "Hi there!",
    "alternate_greetings": [
      "Hello!",
      "Hi!",
      "Hey!"
    ],
    "personality": "Friendly and helpful.",
    "scenario": "Assisting users with tasks.",
    "creator_notes": "These are creator notes.",
    "character_book": null,
    "assets": [
      {
        "type": "icon",
        "uri": "embeded://path/to/icon.png",
        "name": "main",
        "ext": "png"
      },
      {
        "type": "background",
        "uri": "https://example.com/background.jpg",
        "name": "main",
        "ext": "jpg"
      }
    ],
    "nickname": "ExChar",
    "creator_notes_multilingual": {
      "en": "These are creator notes in English.",
      "es": "Estas son notas del creador en español."
    },
    "source": [
      "http://source.com",
      "https://source.org"
    ],
    "group_only_greetings": [
      "Hello Group!"
    ],
    "creation_date": 1700000000,
    "modification_date": 1700003600
  }
}

Parsed Character Card from CHARX:
{
  "spec": "chara_card_v3",
  "spec_version": "3.0",
  "data": {
    "name": "Example Character",
    "description": "A character for demo.",
    "tags": [
      "demo",
      "test"
    ],
    "creator": "John Doe",
    "character_version": "1.0",
    "mes_example": "Hello, this is an example.",
    "extensions": {},
    "system_prompt": "You are {{char}}.",
    "post_history_instructions": "Remember to be nice.",
    "first_mes": "Hi there!",
    "alternate_greetings": [
      "Hello!",
      "Hi!",
      "Hey!"
    ],
    "personality": "Friendly and helpful.",
    "scenario": "Assisting users with tasks.",
    "creator_notes": "These are creator notes.",
    "character_book": null,
    "assets": [
      {
        "type": "icon",
        "uri": "embeded://path/to/icon.png",
        "name": "main",
        "ext": "png"
      },
      {
        "type": "background",
        "uri": "https://example.com/background.jpg",
        "name": "main",
        "ext": "jpg"
      },
      {
        "type": "icon",
        "uri": "embeded://path/to/icon.png",
        "name": "main",
        "ext": "png"
      }
    ],
    "nickname": "ExChar",
    "creator_notes_multilingual": {
      "en": "These are creator notes in English.",
      "es": "Estas son notas del creador en español."
    },
    "source": [
      "http://source.com",
      "https://source.org"
    ],
    "group_only_greetings": [
      "Hello Group!"
    ],
    "creation_date": 1700000000,
    "modification_date": 1700003600
  }
}

Processed CBS in system_prompt:
You are ExChar.
"""

# Note: The output may vary slightly based on the actual data and CBS replacements.
############################################################################################################
