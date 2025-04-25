# text.py
# Description: This file contains utility functions for text processing, including color formatting and text manipulation.
#
# Imports
#
# 3rd-party Libraries
#
# Local Imports
#
######################################################################################################################
#
# Functions:
import logging
import re


def format_metadata_as_text(metadata):
    if not metadata:
        return "No metadata available"

    formatted_text = "Video Metadata:\n"
    for key, value in metadata.items():
        if value is not None:
            if isinstance(value, list):
                # Join list items with commas
                formatted_value = ", ".join(str(item) for item in value)
            elif key == 'upload_date' and len(str(value)) == 8:
                # Format date as YYYY-MM-DD
                formatted_value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
            elif key in ['view_count', 'like_count']:
                # Format large numbers with commas
                formatted_value = f"{value:,}"
            elif key == 'duration':
                # Convert seconds to HH:MM:SS format
                hours, remainder = divmod(value, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                formatted_value = str(value)

            # Replace underscores with spaces in the key name
            formatted_key = key.replace('_', ' ').capitalize()
            formatted_text += f"{formatted_key}: {formatted_value}\n"
    return formatted_text.strip()


def sanitize_filename(filename):
    """
    Sanitizes the filename by:
      1) Removing forbidden characters entirely (rather than replacing them with '-')
      2) Collapsing consecutive whitespace into a single space
      3) Collapsing consecutive dashes into a single dash
    """
    # 1) Remove forbidden characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # 2) Replace runs of whitespace with a single space
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    # 3) Replace consecutive dashes with a single dash
    sanitized = re.sub(r'-{2,}', '-', sanitized)
    return sanitized


def format_transcription(content):
    # Replace '\n' with actual line breaks
    content = content.replace('\\n', '\n')
    # Split the content by newlines first
    lines = content.split('\n')
    formatted_lines = []
    for line in lines:
        # Add extra space after periods for better readability
        line = line.replace('.', '. ').replace('.  ', '. ')

        # Split into sentences using a more comprehensive regex
        sentences = re.split('(?<=[.!?]) +', line)

        # Trim whitespace from each sentence and add a line break
        formatted_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        # Join the formatted sentences
        formatted_lines.append(' '.join(formatted_sentences))

    # Join the lines with HTML line breaks
    formatted_content = '<br>'.join(formatted_lines)

    return formatted_content


def extract_text_from_segments(segments, include_timestamps=True):
    logging.info(f"Segments received: {segments}")
    logging.info(f"Type of segments: {type(segments)}")

    def extract_text_recursive(data, include_timestamps):
        if isinstance(data, dict):
            text = data.get('Text', '')
            if include_timestamps and 'Time_Start' in data and 'Time_End' in data:
                return f"{data['Time_Start']}s - {data['Time_End']}s | {text}"
            for key, value in data.items():
                if key == 'Text':
                    return value
                elif isinstance(value, (dict, list)):
                    result = extract_text_recursive(value, include_timestamps)
                    if result:
                        return result
        elif isinstance(data, list):
            return '\n'.join(filter(None, [extract_text_recursive(item, include_timestamps) for item in data]))
        return None

    text = extract_text_recursive(segments, include_timestamps)

    if text:
        return text.strip()
    else:
        logging.error(f"Unable to extract text from segments: {segments}")
        return "Error: Unable to extract transcription"


def format_text_with_line_breaks(text):
    # Split the text into sentences and add line breaks
    sentences = text.replace('. ', '.<br>').replace('? ', '?<br>').replace('! ', '!<br>')
    return sentences


def format_transcript(raw_text: str) -> str:
    """Convert timestamped transcript to readable format"""
    lines = []
    for line in raw_text.split('\n'):
        if '|' in line:
            timestamp, text = line.split('|', 1)
            lines.append(f"{text.strip()}")
        else:
            lines.append(line.strip())
    return '\n'.join(lines)


#
# End of text.py
######################################################################################################################
