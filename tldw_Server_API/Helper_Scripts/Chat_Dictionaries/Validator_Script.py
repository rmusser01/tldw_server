#!/usr/bin/env python3
"""
Chat Dictionary Validation Tool
Usage: validate_chat_dict.py [--config path/to/config.ini] [--file path/to/file.md]
"""

import re
import configparser
import argparse
from pathlib import Path
import logging
import sys
from typing import Dict, Set, Optional, List

from loguru import logger


# Reuse existing parser
#from App_Function_Libraries.Chat.Chat_Functions import parse_user_dict_markdown_file
def parse_user_dict_markdown_file(file_path: str) -> Dict[str, str]:
    logger.debug(f"Parsing user dictionary file: {file_path}")
    replacement_dict: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_value_lines: List[str] = []

    # Regex to match "key:", "key :", "key: value", "key : value" at the start of a line
    new_key_pattern = re.compile(r'^\s*([^:\n]+?)\s*:(.*)$')
    termination_pattern = re.compile(r'^\s*---@@@---\s*$')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line_content in enumerate(file, 1):
                line_stripped_for_logic = line_content.strip()  # Use for logic checks

                # 1. Check for multi-line termination
                if termination_pattern.match(line_stripped_for_logic):  # Match on stripped line for robustness
                    if current_key:
                        replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
                        logger.trace(f"L{line_number}: Terminated multi-line for '{current_key}'.")
                        current_key, current_value_lines = None, []
                    else:
                        logger.trace(f"L{line_number}: Found terminator but no active multi-line key.")
                    continue

                # 2. Check if this line defines a new key
                new_key_match = new_key_pattern.match(
                    line_content)  # Match on original line to capture leading spaces in values

                if new_key_match:
                    # If we were capturing a multi-line value, this new key definition finalizes it.
                    if current_key:
                        replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
                        logger.trace(
                            f"L{line_number}: New key found, finalizing previous multi-line key '{current_key}'.")

                    potential_new_key = new_key_match.group(1).strip()
                    # Get the value part and strip leading/trailing whitespace FROM THE VALUE PART ONLY
                    potential_value_part = new_key_match.group(2).strip()

                    if potential_value_part == '|':  # Start of new multi-line
                        current_key = potential_new_key
                        current_value_lines = []
                        logger.trace(f"L{line_number}: Starting multi-line for '{current_key}'.")
                    else:  # Single-line value for the new key
                        replacement_dict[potential_new_key] = potential_value_part
                        logger.trace(
                            f"L{line_number}: Parsed single-line key '{potential_new_key}':'{potential_value_part}'.")
                        current_key, current_value_lines = None, []  # Ensure reset
                    continue

                    # 3. If in multi-line mode, append the original line (minus only trailing newline)
                if current_key:
                    current_value_lines.append(line_content.rstrip('\n\r'))
                # 4. Else (not a terminator, not a new key, not in multi-line mode) -> line is ignored (comment/blank)
                # else:
                #    logger.trace(f"L{line_number}: Ignored line: '{line_stripped_for_logic}'")

        # After loop, finalize any pending multi-line value
        if current_key:
            replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
            logger.debug(f"Finalizing last multi-line key '{current_key}' at EOF.")

    except FileNotFoundError:
        logger.error(f"Chat dictionary file not found: {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing chat dictionary file {file_path}: {e}", exc_info=True)
        return {}

    logger.debug(f"Finished parsing chat dictionary. Keys: {list(replacement_dict.keys())}")
    return replacement_dict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ChatDictValidator:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.seen_keys: Set[str] = set()

    def validate_file(self, file_path: Path) -> None:
        """Validate a single Markdown file"""
        if not file_path.exists():
            self.errors.append(f"File not found: {file_path}")
            return

        try:
            entries = parse_user_dict_markdown_file(file_path)
            self._validate_entries(entries, str(file_path))
        except FileNotFoundError:
            self.errors.append(f"File not found: {file_path}")
        except PermissionError:
            self.errors.append(f"Permission denied: {file_path}")
        except Exception as e:
            self.errors.append(f"CRITICAL: Failed to parse {file_path}: {str(e)}")

    def _validate_entries(self, entries: Dict[str, str], source: str) -> None:
        """Validate parsed entries"""
        for key, value in entries.items():
            # Key validation
            if not key.strip():
                self.errors.append(f"Empty key in {source}")
                continue

            if key in self.seen_keys:
                self.errors.append(f"Duplicate key '{key}' in {source}")
            self.seen_keys.add(key)

            # Value validation
            if not value.strip():
                self.warnings.append(f"Empty value for key '{key}' in {source}")

            # Regex validation
            if key.startswith("/") and key.endswith("/"):
                try:
                    re.compile(key[1:-1])
                except re.error as e:
                    self.errors.append(f"Invalid regex '{key}' in {source}: {str(e)}")
                            # Markdown formatting validation (example: check for bold formatting)
            if "**" in value:
                # Check if bold formatting is correctly used
                if value.count("**") % 2 != 0:
                    self.warnings.append(f"Unbalanced bold formatting in key '{key}' in {source}")

            # Check for unterminated multi-line blocks
            if value.startswith('|'):
                self.warnings.append(f"Potential unterminated multi-line block for key '{key}' in {source}")

    def report(self) -> None:
        """Print validation results"""
        if self.warnings:
            logging.warning("\nWarnings:\n• %s", "\n• ".join(self.warnings))

        if self.errors:
            logging.error("\nErrors:\n• %s", "\n• ".join(self.errors))
            sys.exit(1)
        else:
            logging.info("Validation passed!")

def load_config_files(config_path: Path) -> list[Path]:
    """Load files from config"""
    if not config_path.exists():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)

    files = []
    if config.has_section('prompt_config') and config.has_option('prompt_config', 'markdown_files'):
        files = [
            f.strip() for f in
            config.get('prompt_config', 'markdown_files').split('\n')
            if f.strip()
        ]

    return [Path(f) for f in files if Path(f).exists()]

def main():
    parser = argparse.ArgumentParser(description="Validate Chat Dictionary Markdown files")
    parser.add_argument('--config', type=Path, default='config.ini', help="Path to config file")
    parser.add_argument('--file', type=Path, help="Validate single Markdown file")
    args = parser.parse_args()

    validator = ChatDictValidator()

    if args.file:
        validator.validate_file(args.file)
    else:
        for md_file in load_config_files(args.config):
            validator.validate_file(md_file)

    validator.report()

if __name__ == "__main__":
    main()