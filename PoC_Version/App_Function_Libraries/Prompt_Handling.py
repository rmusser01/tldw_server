import os
import shutil
import sqlite3
import tempfile
import zipfile
import re

from PoC_Version.App_Function_Libraries.Utils.Utils import get_database_path


def import_prompt_from_file(file):
    if file is None:
        return "No file uploaded. Please upload a file."

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get the original file name
            original_filename = file.name if hasattr(file, 'name') else 'unknown_file'

            # Create a path for the temporary file
            temp_file_path = os.path.join(temp_dir, original_filename)

            # Write the contents to the temporary file
            if isinstance(file, str):
                # If file is a string, it's likely a file path
                shutil.copy(file, temp_file_path)
            elif hasattr(file, 'read'):
                # If file has a 'read' method, it's likely a file-like object
                with open(temp_file_path, 'wb') as temp_file:
                    shutil.copyfileobj(file, temp_file)
            else:
                # If it's neither a string nor a file-like object, try converting it to a string
                with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                    temp_file.write(str(file))

            # Read and parse the content from the temporary file
            with open(temp_file_path, 'r', encoding='utf-8') as temp_file:
                file_content = temp_file.read()

            sections = parse_prompt_file(file_content)

        return sections['title'], sections['author'], sections['system'], sections['user'], sections['keywords']
    except Exception as e:
        return f"Error parsing file: {str(e)}"

def parse_prompt_file(file_content):
    sections = {
        'title': '',
        'author': '',
        'system': '',
        'user': '',
        'keywords': []
    }

    # Define regex patterns for the sections
    patterns = {
        'title': r'### TITLE ###\s*(.*?)\s*###',
        'author': r'### AUTHOR ###\s*(.*?)\s*###',
        'system': r'### SYSTEM ###\s*(.*?)\s*###',
        'user': r'### USER ###\s*(.*?)\s*###',
        'keywords': r'### KEYWORDS ###\s*(.*?)\s*###'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, file_content, re.DOTALL)
        if match:
            if key == 'keywords':
                # Split keywords by commas and strip whitespace
                sections[key] = [k.strip() for k in match.group(1).split(',') if k.strip()]
            else:
                sections[key] = match.group(1).strip()

    return sections


# FIXME - update to use DB Manager / ES Support
def import_prompt_data(name, details, system, user):
    if not name or not system:
        return "Name and System fields are required."

    try:
        conn = sqlite3.connect(get_database_path('prompts.db'))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Prompts (name, details, system, user)
            VALUES (?, ?, ?, ?)
        ''', (name, details, system, user))
        conn.commit()
        conn.close()
        return f"Prompt '{name}' successfully imported."
    except sqlite3.IntegrityError:
        return "Prompt with this name already exists."
    except sqlite3.Error as e:
        return f"Database error: {e}"


def import_prompts_from_zip(zip_file):
    if zip_file is None:
        return "No file uploaded. Please upload a file."

    prompts = []
    temp_dir = tempfile.mkdtemp()
    try:
        zip_path = os.path.join(temp_dir, zip_file.name)
        with open(zip_path, 'wb') as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('.txt') or filename.endswith('.md'):
                    with z.open(filename) as f:
                        file_content = f.read().decode('utf-8')
                        sections = parse_prompt_file(file_content)
                        if 'keywords' not in sections:
                            sections['keywords'] = []
                        prompts.append(sections)
        shutil.rmtree(temp_dir)
        return prompts
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error parsing zip file: {str(e)}"