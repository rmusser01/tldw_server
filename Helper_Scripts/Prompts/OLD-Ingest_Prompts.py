#!/usr/bin/env python
#
# Usage:
#           python ingest_prompts.py /path/to/your/parent/folder
#
import os
import sqlite3
import argparse
import re

DATABASE_PATH = '../../Databases/prompts.db'


def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def parse_prompt_file(file_content):
    sections = {
        'title': '',
        'author': '',
        'system': '',
        'user': '',
        'keywords': []
    }

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
                sections[key] = [k.strip() for k in match.group(1).split(',') if k.strip()]
            else:
                sections[key] = match.group(1).strip()

    return sections


def ingest_prompts_from_folder(parent_folder):
    print(f"Attempting to ingest prompts from folder: {parent_folder}")

    if not os.path.exists(parent_folder):
        print(f"Error: The specified folder does not exist: {parent_folder}")
        return

    if not os.path.isdir(parent_folder):
        print(f"Error: The specified path is not a directory: {parent_folder}")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return

    md_files = [f for f in os.listdir(parent_folder) if f.endswith('.md')]

    if not md_files:
        print(f"No .md files found in the specified folder: {parent_folder}")
        return

    print(f"Found {len(md_files)} .md files in the folder.")

    for filename in md_files:
        file_path = os.path.join(parent_folder, filename)
        print(f"Processing file: {file_path}")

        file_content = read_file_content(file_path)
        if file_content is None:
            continue

        prompt_data = parse_prompt_file(file_content)

        name = prompt_data['title'] or os.path.splitext(filename)[0]
        details = f"Author: {prompt_data['author'] or 'fabric project'}"
        system = prompt_data['system']
        user = prompt_data['user']
        keywords = prompt_data['keywords']

        try:
            # Insert into Prompts table
            cursor.execute('''
                INSERT INTO Prompts (name, details, system, user)
                VALUES (?, ?, ?, ?)
            ''', (name, details, system, user))
            prompt_id = cursor.lastrowid

            # Insert keywords
            for keyword in keywords:
                # Insert or get keyword_id
                cursor.execute('''
                    INSERT OR IGNORE INTO Keywords (keyword)
                    VALUES (?)
                ''', (keyword,))
                cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
                keyword_id = cursor.fetchone()[0]

                # Link prompt to keyword
                cursor.execute('''
                    INSERT OR IGNORE INTO PromptKeywords (prompt_id, keyword_id)
                    VALUES (?, ?)
                ''', (prompt_id, keyword_id))

            print(f"Successfully ingested prompt: {name}")
        except sqlite3.IntegrityError:
            print(f"Prompt with name '{name}' already exists.")
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    conn.commit()
    conn.close()
    print("Ingestion process completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ingest prompts from a specified folder into the Prompts database.')
    parser.add_argument('folder', type=str, help='The path to the parent folder containing prompt files.')
    args = parser.parse_args()

    print(f"Script executed with argument: {args.folder}")
    ingest_prompts_from_folder(args.folder)