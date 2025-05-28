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
from typing import Dict, Set

# Reuse existing parser
#from App_Function_Libraries.Chat.Chat_Functions import parse_user_dict_markdown_file
def parse_user_dict_markdown_file(file_path):
    """
    Parse a Markdown file with custom termination symbol for multi-line values.
    """
    logging.debug(f"Parsing user dictionary file: {file_path}")
    replacement_dict = {}
    current_key = None
    current_value = []
    termination_pattern = re.compile(r'^\s*---@@@---\s*$')

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check for termination pattern first
            if termination_pattern.match(line):
                if current_key:
                    replacement_dict[current_key] = '\n'.join(current_value).strip()
                    current_key, current_value = None, []
                continue

            # Match key lines only when not in multi-line mode
            if not current_key:
                key_value_match = re.match(r'^\s*([^:\n]+?)\s*:\s*(.*?)\s*$', line)
                if key_value_match:
                    key, value = key_value_match.groups()
                    if value.strip() == '|':
                        current_key = key.strip()
                        current_value = []
                    else:
                        replacement_dict[key.strip()] = value.strip()
                continue

            # Processing multi-line content
            if current_key:
                # Strip trailing whitespace but preserve leading spaces
                cleaned_line = line.rstrip('\n\r')
                current_value.append(cleaned_line)

        # Handle any remaining multi-line value at EOF
        if current_key:
            replacement_dict[current_key] = '\n'.join(current_value).strip()

    logging.debug(f"Parsed entries: {replacement_dict}")
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