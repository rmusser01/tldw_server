import json
import os
import sys
from pathlib import Path


def extract_transcription(input_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Determine if the input path is a file or a folder
    input_path = Path(input_path)
    if input_path.is_file():
        json_files = [input_path]
    elif input_path.is_dir():
        json_files = list(input_path.glob('*.json'))
    else:
        print(f"Invalid input path: {input_path}")
        return

    # Process each JSON file
    for json_file in json_files:
        # Read the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Extract the segments
        if isinstance(data, dict):
            segments = data.get('segments', [])
        elif isinstance(data, list):
            segments = data
        else:
            print(f"Unexpected data format in {json_file.name}")
            continue

        # Define the output text file path
        output_text_path = Path(output_folder) / (json_file.stem + '_transcription.txt')

        # Open the output text file in write mode
        with open(output_text_path, 'w') as text_file:
            # Iterate over each segment and write the text to the output file
            for segment in segments:
                text_line = segment.get('Text', '')
                text_file.write(text_line + '\n')

        print(f"Transcription for {json_file.name} has been written to {output_text_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python Turn_JSON_into_Txt.py <input_file_or_folder> <output_folder>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_folder = sys.argv[2]

    extract_transcription(input_path, output_folder)