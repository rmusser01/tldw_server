# tests/test_parser.py
# Description: Tests for the parser
#
# Usage: python -m unittest discover -s tests
#
# Imports
import os
import sys
import unittest
#
# External Imports
#

# Add the tldw directory (one level up from Tests) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'tldw')))
#
# Local Imports
from App_Function_Libraries.Personas.ccv3_parser import CharacterCardV3Parser, CCv3ParserError
#
#############################################################################################################
#
# Tests

class TestCharacterCardV3Parser(unittest.TestCase):
    def setUp(self):
        self.valid_json = '''
        {
          "spec": "chara_card_v3",
          "spec_version": "3.0",
          "data": {
            "name": "Test Character",
            "description": "A test character.",
            "tags": ["test"],
            "creator": "Tester",
            "character_version": "1.0",
            "mes_example": "Test message.",
            "extensions": {},
            "system_prompt": "You are {{char}}.",
            "post_history_instructions": "Be brief.",
            "first_mes": "Hello!",
            "alternate_greetings": ["Hi!", "Hey!"],
            "personality": "Testy.",
            "scenario": "Testing scenario.",
            "creator_notes": "Test notes.",
            "group_only_greetings": [],
            "assets": [
                {"type": "icon", "uri": "ccdefault:", "name": "main", "ext": "png"}
            ]
          }
        }
        '''

        self.invalid_spec_json = '''
        {
          "spec": "chara_card_v2",
          "spec_version": "2.0",
          "data": {}
        }
        '''

        self.missing_field_json = '''
        {
          "spec": "chara_card_v3",
          "spec_version": "3.0",
          "data": {
            "name": "Incomplete Character"
            // Missing other required fields
          }
        }
        '''

    def test_valid_json(self):
        parser = CharacterCardV3Parser(input_data=self.valid_json, input_type='json')
        try:
            parser.parse()
            character_card = parser.get_character_card()
            self.assertIsNotNone(character_card)
            self.assertEqual(character_card.spec, 'chara_card_v3')
            self.assertEqual(character_card.data.name, 'Test Character')
        except CCv3ParserError:
            self.fail("CCv3ParserError raised unexpectedly!")

    def test_invalid_spec(self):
        parser = CharacterCardV3Parser(input_data=self.invalid_spec_json, input_type='json')
        with self.assertRaises(CCv3ParserError):
            parser.parse()

    def test_missing_field(self):
        parser = CharacterCardV3Parser(input_data=self.missing_field_json, input_type='json')
        with self.assertRaises(CCv3ParserError):
            parser.parse()

if __name__ == '__main__':
    unittest.main()