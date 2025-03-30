# Anki.py
# Description: Functions for Anki card generation
#
# Imports
import json
import zipfile
import sqlite3
import tempfile
import os
import shutil
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import re
from html.parser import HTMLParser
#
# External Imports
#from outlines import models, prompts
# Local Imports
#
############################################################################################################
#
# Functions:

class HTMLImageExtractor(HTMLParser):
    """Extract and validate image tags from HTML content."""

    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            attrs_dict = dict(attrs)
            if 'src' in attrs_dict:
                self.images.append(attrs_dict['src'])


def sanitize_html(content: str) -> str:
    """Sanitize HTML content while preserving valid image tags and basic formatting."""
    if not content:
        return ""

    # Allow basic formatting and image tags
    allowed_tags = {'img', 'b', 'i', 'u', 'div', 'br', 'p', 'span'}
    allowed_attrs = {'src', 'alt', 'class', 'style'}

    # Remove potentially harmful attributes
    content = re.sub(r'(on\w+)="[^"]*"', '', content)
    content = re.sub(r'javascript:', '', content)

    # Parse and rebuild HTML
    parser = HTMLParser()
    parser.feed(content)
    return content


def extract_media_from_apkg(zip_path: Any, temp_dir: str) -> Dict[str, str]:
    """Extract and process media files from APKG."""
    media_files = {}
    try:
        # Handle file path whether it's a string or file object
        if hasattr(zip_path, 'name'):
            # It's a file object from Gradio
            file_name = zip_path.name
        else:
            # It's a string path
            file_name = str(zip_path)

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            if 'media' in zip_ref.namelist():
                media_json = json.loads(zip_ref.read('media').decode('utf-8'))

                for file_id, filename in media_json.items():
                    if str(file_id) in zip_ref.namelist():
                        file_data = zip_ref.read(str(file_id))
                        file_path = os.path.join(temp_dir, filename)

                        # Save file temporarily
                        with open(file_path, 'wb') as f:
                            f.write(file_data)

                        # Process supported image types
                        if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                            try:
                                with open(file_path, 'rb') as f:
                                    file_content = f.read()
                                    file_ext = os.path.splitext(filename)[1].lower()
                                    media_type = f"image/{file_ext[1:]}"
                                    if file_ext == '.jpg':
                                        media_type = "image/jpeg"
                                    media_files[
                                        filename] = f"data:{media_type};base64,{base64.b64encode(file_content).decode('utf-8')}"
                            except Exception as e:
                                print(f"Error processing image {filename}: {str(e)}")

                        # Clean up temporary file
                        os.remove(file_path)

    except Exception as e:
        print(f"Error processing media: {str(e)}")
    return media_files


def validate_card_content(card: Dict[str, Any], seen_ids: set) -> list:
    """Validate individual card content and structure."""
    issues = []

    # Check required fields
    if 'id' not in card:
        issues.append("Missing ID")
    elif card['id'] in seen_ids:
        issues.append("Duplicate ID")
    else:
        seen_ids.add(card['id'])

    if 'type' not in card or card['type'] not in ['basic', 'cloze', 'reverse']:
        issues.append("Invalid card type")

    if 'front' not in card or not card['front'].strip():
        issues.append("Missing front content")

    if 'back' not in card or not card['back'].strip():
        issues.append("Missing back content")

    if 'tags' not in card or not card['tags']:
        issues.append("Missing tags")

    # Content-specific validation
    if card.get('type') == 'cloze':
        if '{{c1::' not in card['front']:
            issues.append("Invalid cloze format")

    # Image validation
    for field in ['front', 'back']:
        if '<img' in card[field]:
            extractor = HTMLImageExtractor()
            extractor.feed(card[field])
            for img_src in extractor.images:
                if not (img_src.startswith('data:image/') or img_src.startswith('http')):
                    issues.append(f"Invalid image source in {field}")

    return issues


def process_apkg_file(file_path: str) -> Tuple[Optional[Dict], Optional[Dict], str]:
    """Process APKG file with support for different Anki database versions."""
    if not file_path:
        return None, None, "No file provided"
    # Handle file path whether it's a string or file object
    if hasattr(file_path, 'name'):
        # It's a file object from Gradio
        file_name = file_path.name
    else:
        # It's a string path
        file_name = str(file_path)

    temp_dir = None
    db_conn = None
    cursor = None
    cards_data = {"cards": []}
    deck_info = None

    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Extract media files first
        media_files = extract_media_from_apkg(file_name, temp_dir)

        # Extract APKG contents
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            zip_ref.extractall(temp_dir)

        db_path = os.path.join(temp_dir, 'collection.anki2')

        # Process database with explicit connection management
        db_conn = sqlite3.connect(db_path)
        cursor = db_conn.cursor()

        try:
            # Get collection info
            cursor.execute("SELECT decks, models FROM col")
            decks_json, models_json = cursor.fetchone()
            deck_info = {
                "decks": json.loads(decks_json),
                "models": json.loads(models_json)
            }

            # Check if we're dealing with an older or newer Anki version
            try:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notetypes'")
                has_notetypes = cursor.fetchone() is not None

                if has_notetypes:
                    # New Anki version (2.1.28+)
                    cursor.execute("""
                        SELECT 
                            n.id, n.flds, n.tags, c.type, n.mid, 
                            m.name, n.sfld, m.flds, m.tmpls
                        FROM notes n
                        JOIN cards c ON c.nid = n.id
                        JOIN notetypes m ON m.id = n.mid
                    """)
                else:
                    # Older Anki version
                    cursor.execute("""
                        SELECT 
                            n.id, n.flds, n.tags, c.type, n.mid, 
                            m.name, n.sfld, m.flds, m.tmpls
                        FROM notes n
                        JOIN cards c ON c.nid = n.id
                        JOIN col AS m ON m.id = 1 AND json_extract(m.models, '$.' || n.mid) IS NOT NULL
                    """)

                rows = cursor.fetchall()

            except sqlite3.Error as e:
                # Fallback query for very old Anki versions
                cursor.execute("""
                    SELECT 
                        n.id, n.flds, n.tags, c.type, n.mid,
                        '', n.sfld, '[]', '[]'
                    FROM notes n
                    JOIN cards c ON c.nid = n.id
                """)
                rows = cursor.fetchall()

        finally:
            cursor.close()
            db_conn.close()

        # Process the fetched data
        for row in rows:
            note_id, fields, tags, card_type, model_id = row[0:5]
            model_name = row[5] if row[5] else "Unknown Model"
            fields_list = fields.split('\x1f')

            try:
                fields_config = json.loads(row[7]) if row[7] else []
                templates = json.loads(row[8]) if row[8] else []
            except json.JSONDecodeError:
                fields_config = []
                templates = []

            # Process fields with media
            processed_fields = []
            for field in fields_list:
                field_html = field
                for filename, base64_data in media_files.items():
                    field_html = field_html.replace(
                        f'<img src="{filename}"',
                        f'<img src="{base64_data}"'
                    )
                processed_fields.append(sanitize_html(field_html))

            # Determine card type (simplified logic)
            converted_type = 'basic'
            if any('cloze' in str(t).lower() for t in templates):
                converted_type = 'cloze'
            elif any('{{FrontSide}}' in str(t) for t in templates):
                converted_type = 'reverse'

            card_data = {
                "id": f"APKG_{note_id}",
                "type": converted_type,
                "front": processed_fields[0] if processed_fields else "",
                "back": processed_fields[1] if len(processed_fields) > 1 else "",
                "tags": tags.strip().split(" ") if tags.strip() else ["imported"],
                "note": f"Imported from deck: {model_name}",
                "has_media": any('<img' in field for field in processed_fields),
                "model_name": model_name,
                "field_names": [f.get('name', f'Field_{i}') for i, f in enumerate(fields_config)],
                "template_names": [t.get('name', f'Template_{i}') for i, t in enumerate(templates)]
            }

            cards_data["cards"].append(card_data)

        if not cards_data["cards"]:
            return None, None, "No cards found in the APKG file"

        return cards_data, deck_info, "APKG file processed successfully!"

    except sqlite3.Error as e:
        return None, None, f"Database error: {str(e)}"
    except json.JSONDecodeError as e:
        return None, None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, None, f"Error processing APKG file: {str(e)}"
    finally:
        # Clean up resources
        if cursor:
            try:
                cursor.close()
            except:
                pass
        if db_conn:
            try:
                db_conn.close()
            except:
                pass
        if temp_dir and os.path.exists(temp_dir):
            try:
                time.sleep(0.1)
                for root, dirs, files in os.walk(temp_dir):
                    for fname in files:
                        try:
                            os.chmod(os.path.join(root, fname), 0o777)
                        except:
                            pass
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {temp_dir}: {str(e)}")

def validate_flashcards(content: str) -> Tuple[bool, str]:
    """Validate flashcard content with enhanced image support."""
    try:
        data = json.loads(content)
        validation_results = []
        is_valid = True

        if not isinstance(data, dict) or 'cards' not in data:
            return False, "Invalid JSON format. Must contain 'cards' array."

        seen_ids = set()
        for idx, card in enumerate(data['cards']):
            card_issues = validate_card_content(card, seen_ids)

            if card_issues:
                is_valid = False
                validation_results.append(f"Card {card['id']}: {', '.join(card_issues)}")

        return is_valid, "\n".join(validation_results) if validation_results else "All cards are valid!"

    except json.JSONDecodeError:
        return False, "Invalid JSON format"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def enhanced_file_upload(file: Any, input_type: str) -> Tuple[Optional[str], Optional[Dict], str, List[str]]:
    """Enhanced file upload handler with better error handling."""
    if not file:
        return None, None, "❌ No file uploaded", []

    try:
        if input_type == "APKG":
            cards_data, deck_info, message = process_apkg_file(file)
            if cards_data:
                content = json.dumps(cards_data, indent=2)
                choices = update_card_choices(content)
                # Validate the converted content
                validation_msg = handle_validation(content, "APKG")
                return content, deck_info, validation_msg, choices
            return None, None, f"❌ {message}", []
        else:
            # Original JSON file handling
            content = file.read().decode('utf-8')
            json.loads(content)  # Validate JSON
            return content, None, "✅ JSON file loaded successfully!", update_card_choices(content)
    except Exception as e:
        return None, None, f"❌ Error processing file: {str(e)}", []

def handle_file_upload(file: Any, input_type: str) -> Tuple[Optional[str], Optional[Dict], str, List[str]]:
    """Handle file upload with proper validation message formatting and card choices update."""
    if not file:
        return None, None, "❌ No file uploaded", []

    if input_type == "APKG":
        cards_data, deck_info, message = process_apkg_file(file)
        if cards_data:
            content = json.dumps(cards_data, indent=2)
            return (
                content,
                deck_info,
                f"✅ {message}",
                update_card_choices(content)
            )
        return None, None, f"❌ {message}", []
    else:  # JSON
        try:
            content = file.read().decode('utf-8')
            json.loads(content)  # Validate JSON
            return (
                content,
                None,
                "✅ JSON file loaded successfully!",
                update_card_choices(content)
            )
        except Exception as e:
            return None, None, f"❌ Error loading JSON file: {str(e)}", []

def update_card_choices(content: str) -> List[str]:
    """Update card choices for the dropdown."""
    try:
        data = json.loads(content)
        return [f"{card['id']} - {card['front'][:50]}..." for card in data['cards']]
    except:
        return []


def update_card_content(
        current_content: str,
        card_id: str,
        card_type: str,
        front: str,
        back: str,
        tags: str,
        notes: str
) -> Tuple[str, str]:
    """Update card content and return updated JSON and status message."""
    try:
        data = json.loads(current_content)

        for card in data['cards']:
            if card['id'] == card_id:
                # Sanitize input content
                card['type'] = card_type
                card['front'] = sanitize_html(front)
                card['back'] = sanitize_html(back)
                card['tags'] = [tag.strip() for tag in tags.split(',')]
                card['note'] = notes

                # Update media status
                card['has_media'] = '<img' in front or '<img' in back

                return json.dumps(data, indent=2), "Card updated successfully!"

        return current_content, "Card not found!"

    except Exception as e:
        return current_content, f"Error updating card: {str(e)}"


def load_card_for_editing(card_selection: str, current_content: str) -> tuple:
    """
    Load a card for editing and generate previews.

    Args:
        card_selection (str): Selected card ID and preview text
        current_content (str): Current JSON content

    Returns:
        tuple: (card_type, front_content, back_content, tags, notes, front_preview, back_preview)
    """
    if not card_selection or not current_content:
        return "basic", "", "", "", "", "", ""

    try:
        data = json.loads(current_content)
        selected_id = card_selection.split(" - ")[0]

        for card in data['cards']:
            if card['id'] == selected_id:
                # Return all required fields with preview content
                return (
                    card['type'],
                    card['front'],
                    card['back'],
                    ", ".join(card['tags']),
                    card.get('note', ''),
                    sanitize_html(card['front']),
                    sanitize_html(card['back'])
                )

        return "basic", "", "", "", "", "", ""

    except Exception as e:
        print(f"Error loading card: {str(e)}")
        return "basic", "", "", "", "", "", ""


def export_cards(content: str, format_type: str) -> Tuple[str, Optional[Tuple[str, str, str]]]:
    """Export cards in the specified format."""
    try:
        is_valid, validation_message = validate_flashcards(content)
        if not is_valid:
            return "Please fix validation issues before exporting.", None

        data = json.loads(content)

        if format_type == "Anki CSV":
            output = "Front,Back,Tags,Type,Note\n"
            for card in data['cards']:
                output += f'"{card["front"]}","{card["back"]}","{" ".join(card["tags"])}","{card["type"]}","{card.get("note", "")}"\n'
            return "Cards exported successfully!", ("anki_cards.csv", output, "text/csv")

        elif format_type == "JSON":
            return "Cards exported successfully!", ("anki_cards.json", content, "application/json")

        else:  # Plain Text
            output = ""
            for card in data['cards']:
                # Replace image tags with placeholders
                front = re.sub(r'<img[^>]+>', '[IMG]', card['front'])
                back = re.sub(r'<img[^>]+>', '[IMG]', card['back'])
                output += f"Q: {front}\nA: {back}\nTags: {', '.join(card['tags'])}\n\n"
            return "Cards exported successfully!", ("anki_cards.txt", output, "text/plain")

    except Exception as e:
        return f"Export error: {str(e)}", None


def generate_card_choices(content: str) -> list:
    """Generate choices for card selector dropdown."""
    try:
        data = json.loads(content)
        return [f"{card['id']} - {card['front'][:50]}..." for card in data['cards']]
    except:
        return []

def format_validation_result(content: str) -> str:
    """Format validation results for display in Markdown component."""
    try:
        is_valid, message = validate_flashcards(content)
        return f"✅ {message}" if is_valid else f"❌ {message}"
    except Exception as e:
        return f"❌ Error during validation: {str(e)}"


def validate_for_ui(content: str) -> str:
    """Validate flashcards and return a formatted string for UI display."""
    if not content or not content.strip():
        return "❌ No content to validate. Please enter some flashcard data."

    try:
        # First try to parse the JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as je:
            # Provide more specific JSON error feedback
            line_col = f" (line {je.lineno}, column {je.colno})" if hasattr(je, 'lineno') else ""
            return f"❌ Invalid JSON format: {str(je)}{line_col}"

        # Check basic structure
        if not isinstance(data, dict):
            return "❌ Invalid format: Root element must be a JSON object"

        if "cards" not in data:
            return '❌ Invalid format: Missing "cards" array in root object'

        if not isinstance(data["cards"], list):
            return '❌ Invalid format: "cards" must be an array'

        if not data["cards"]:
            return "❌ No cards found in the data"

        # If we get here, perform the full validation
        is_valid, message = validate_flashcards(content)
        if is_valid:
            return f"✅ {message}"
        else:
            return f"❌ {message}"

    except Exception as e:
        return f"❌ Validation error: {str(e)}"


def update_card_with_validation(
        current_content: str,
        card_selection: str,
        card_type: str,
        front: str,
        back: str,
        tags: str,
        notes: str
) -> Tuple[str, str, List[str]]:
    """Update card and return properly formatted validation message and updated choices."""
    try:
        # Unpack the tuple returned by update_card_content
        updated_content, message = update_card_content(
            current_content,
            card_selection.split(" - ")[0],
            card_type,
            front,
            back,
            tags,
            notes
        )

        if "successfully" in message:
            return (
                updated_content,
                f"✅ {message}",
                update_card_choices(updated_content)
            )
        else:
            return (
                current_content,
                f"❌ {message}",
                update_card_choices(current_content)
            )
    except Exception as e:
        return (
            current_content,
            f"❌ Error updating card: {str(e)}",
            update_card_choices(current_content)
        )


def handle_validation(content: str, input_format: str) -> str:
    """Handle validation for both JSON and APKG formats."""
    if not content or not content.strip():
        return "❌ No content to validate"

    try:
        data = json.loads(content)

        if not isinstance(data, dict):
            return "❌ Invalid format: Root element must be a JSON object"

        if "cards" not in data:
            return '❌ Invalid format: Missing "cards" array in root object'

        if not isinstance(data["cards"], list):
            return '❌ Invalid format: "cards" must be an array'

        if not data["cards"]:
            return "❌ No cards found in the data"

        card_count = len(data["cards"])
        if input_format == "APKG":
            return f"✅ Successfully imported and validated {card_count} cards from APKG file"
        else:
            # For JSON input, perform additional validation
            is_valid, message = validate_flashcards(content)
            return f"✅ {message}" if is_valid else f"❌ {message}"

    except json.JSONDecodeError as je:
        line_col = f" (line {je.lineno}, column {je.colno})" if hasattr(je, 'lineno') else ""
        return f"❌ Invalid JSON format: {str(je)}{line_col}"
    except Exception as e:
        return f"❌ Validation error: {str(e)}"

#
# End of Anki.py
############################################################################################################
