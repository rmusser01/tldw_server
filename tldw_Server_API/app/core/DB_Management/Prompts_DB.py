# Prompts_DB.py
# Description: Functions to manage the prompts database.
#
# Imports
import sqlite3
#
# External Imports
import re
from typing import Tuple
#
# Local Imports
from PoC_Version.App_Function_Libraries.Utils import get_database_path, logging
#
#######################################################################################################################
#
# Functions to manage prompts DB

def create_prompts_db():
    logging.debug("create_prompts_db: Creating prompts database.")
    with sqlite3.connect(get_database_path('prompts.db')) as conn:
        cursor = conn.cursor()
        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS Prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                author TEXT,
                details TEXT,
                system TEXT,
                user TEXT
            );
            CREATE TABLE IF NOT EXISTS Keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE COLLATE NOCASE
            );
            CREATE TABLE IF NOT EXISTS PromptKeywords (
                prompt_id INTEGER,
                keyword_id INTEGER,
                FOREIGN KEY (prompt_id) REFERENCES Prompts (id),
                FOREIGN KEY (keyword_id) REFERENCES Keywords (id),
                PRIMARY KEY (prompt_id, keyword_id)
            );
            CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON Keywords(keyword);
            CREATE INDEX IF NOT EXISTS idx_promptkeywords_prompt_id ON PromptKeywords(prompt_id);
            CREATE INDEX IF NOT EXISTS idx_promptkeywords_keyword_id ON PromptKeywords(keyword_id);
        ''')

# FIXME - dirty hack that should be removed later...
# Migration function to add the 'author' column to the Prompts table
def add_author_column_to_prompts():
    with sqlite3.connect(get_database_path('prompts.db')) as conn:
        cursor = conn.cursor()
        # Check if 'author' column already exists
        cursor.execute("PRAGMA table_info(Prompts)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'author' not in columns:
            # Add the 'author' column
            cursor.execute('ALTER TABLE Prompts ADD COLUMN author TEXT')
            print("Author column added to Prompts table.")
        else:
            print("Author column already exists in Prompts table.")

add_author_column_to_prompts()

def normalize_keyword(keyword):
    return re.sub(r'\s+', ' ', keyword.strip().lower())


# FIXME - update calls to this function to use the new args
def add_prompt(name, author, details, system=None, user=None, keywords=None):
    logging.debug(f"add_prompt: Adding prompt with name: {name}, author: {author}, system: {system}, user: {user}, keywords: {keywords}")
    if not name:
        logging.error("add_prompt: A name is required.")
        return "A name is required."

    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO Prompts (name, author, details, system, user)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, author, details, system, user))
            prompt_id = cursor.lastrowid

            if keywords:
                normalized_keywords = [normalize_keyword(k) for k in keywords if k.strip()]
                for keyword in set(normalized_keywords):  # Use set to remove duplicates
                    cursor.execute('''
                        INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)
                    ''', (keyword,))
                    cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
                    keyword_id = cursor.fetchone()[0]
                    cursor.execute('''
                        INSERT OR IGNORE INTO PromptKeywords (prompt_id, keyword_id) VALUES (?, ?)
                    ''', (prompt_id, keyword_id))
        return "Prompt added successfully."
    except sqlite3.IntegrityError:
        return "Prompt with this name already exists."
    except sqlite3.Error as e:
        return f"Database error: {e}"


def fetch_prompt_details(name):
    logging.debug(f"fetch_prompt_details: Fetching details for prompt: {name}")
    with sqlite3.connect(get_database_path('prompts.db')) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.name, p.author, p.details, p.system, p.user, GROUP_CONCAT(k.keyword, ', ') as keywords
            FROM Prompts p
            LEFT JOIN PromptKeywords pk ON p.id = pk.prompt_id
            LEFT JOIN Keywords k ON pk.keyword_id = k.id
            WHERE p.name = ?
            GROUP BY p.id
        ''', (name,))
        return cursor.fetchone()


def list_prompts(page=1, per_page=10):
    logging.debug(f"list_prompts: Listing prompts for page {page} with {per_page} prompts per page.")
    offset = (page - 1) * per_page
    with sqlite3.connect(get_database_path('prompts.db')) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM Prompts LIMIT ? OFFSET ?', (per_page, offset))
        prompts = [row[0] for row in cursor.fetchall()]

        # Get total count of prompts
        cursor.execute('SELECT COUNT(*) FROM Prompts')
        total_count = cursor.fetchone()[0]

    total_pages = (total_count + per_page - 1) // per_page
    return prompts, total_pages, page


def insert_prompt_to_db(title, author, description, system_prompt, user_prompt, keywords=None):
    return add_prompt(title, author, description, system_prompt, user_prompt, keywords)


def get_prompt_db_connection():
    prompt_db_path = get_database_path('prompts.db')
    return sqlite3.connect(prompt_db_path)


# def search_prompts(query):
#     logging.debug(f"search_prompts: Searching prompts with query: {query}")
#     try:
#         with get_prompt_db_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("""
#                 SELECT p.name, p.details, p.system, p.user, GROUP_CONCAT(k.keyword, ', ') as keywords
#                 FROM Prompts p
#                 LEFT JOIN PromptKeywords pk ON p.id = pk.prompt_id
#                 LEFT JOIN Keywords k ON pk.keyword_id = k.id
#                 WHERE p.name LIKE ? OR p.details LIKE ? OR p.system LIKE ? OR p.user LIKE ? OR k.keyword LIKE ?
#                 GROUP BY p.id
#                 ORDER BY p.name
#             """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
#             return cursor.fetchall()
#     except sqlite3.Error as e:
#         logging.error(f"Error searching prompts: {e}")
#         return []


def search_prompts(query, search_fields):
    logging.debug(f"search_prompts: Searching prompts with query: {query}, fields: {search_fields}")
    try:
        with get_prompt_db_connection() as conn:
            cursor = conn.cursor()

            where_clauses = []
            params = []

            if 'title' in search_fields:
                where_clauses.append("p.name LIKE ?")
                params.append(f'%{query}%')
            if 'content' in search_fields:
                where_clauses.append("(p.details LIKE ? OR p.system LIKE ? OR p.user LIKE ?)")
                params.extend([f'%{query}%'] * 3)
            if 'keywords' in search_fields:
                where_clauses.append("k.keyword LIKE ?")
                params.append(f'%{query}%')

            where_clause = " OR ".join(where_clauses)

            cursor.execute(f"""
                SELECT DISTINCT p.name, p.details, p.system, p.user, GROUP_CONCAT(k.keyword, ', ') as keywords
                FROM Prompts p
                LEFT JOIN PromptKeywords pk ON p.id = pk.prompt_id
                LEFT JOIN Keywords k ON pk.keyword_id = k.id
                WHERE {where_clause}
                GROUP BY p.id
                ORDER BY p.name
            """, params)
            return cursor.fetchall()
    except sqlite3.Error as e:
        logging.error(f"Error searching prompts: {e}")
        return []


def fetch_item_details_with_keywords(media_id):
    logging.debug(f"fetch_item_details_with_keywords: Fetching details for media item with ID: {media_id}")
    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.content, mm.prompt, mm.summary, GROUP_CONCAT(k.keyword, ', ') as keywords
                FROM Media m
                LEFT JOIN MediaModifications mm ON m.id = mm.media_id
                LEFT JOIN MediaKeywords mk ON m.id = mk.media_id
                LEFT JOIN Keywords k ON mk.keyword_id = k.id
                WHERE m.id = ?
                GROUP BY m.id
            """, (media_id,))
            result = cursor.fetchone()
            if result:
                content, prompt, summary, keywords = result
                return content, prompt, summary, keywords
            return None, None, None, None
    except sqlite3.Error as e:
        return f"Database error: {e}"


def update_prompt_keywords(prompt_name, new_keywords):
    logging.debug(f"update_prompt_keywords: Updating keywords for prompt: {prompt_name}")
    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT id FROM Prompts WHERE name = ?', (prompt_name,))
            prompt_id = cursor.fetchone()
            if not prompt_id:
                return "Prompt not found."
            prompt_id = prompt_id[0]

            cursor.execute('DELETE FROM PromptKeywords WHERE prompt_id = ?', (prompt_id,))

            normalized_keywords = [normalize_keyword(k) for k in new_keywords if k.strip()]
            for keyword in set(normalized_keywords):  # Use set to remove duplicates
                cursor.execute('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', (keyword,))
                cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
                keyword_id = cursor.fetchone()[0]
                cursor.execute('INSERT INTO PromptKeywords (prompt_id, keyword_id) VALUES (?, ?)',
                               (prompt_id, keyword_id))

            # Remove unused keywords
            cursor.execute('''
                DELETE FROM Keywords
                WHERE id NOT IN (SELECT DISTINCT keyword_id FROM PromptKeywords)
            ''')
        return "Keywords updated successfully."
    except sqlite3.Error as e:
        return f"Database error: {e}"


def add_or_update_prompt(title, author, description, system_prompt, user_prompt, keywords=None):
    logging.debug(f"add_or_update_prompt: Adding or updating prompt: {title}")
    if not title:
        return "Error: Title is required."

    existing_prompt = fetch_prompt_details(title)
    if existing_prompt:
        # Update existing prompt
        result = update_prompt_in_db(title, author, description, system_prompt, user_prompt)
        if "successfully" in result:
            # Update keywords if the prompt update was successful
            keyword_result = update_prompt_keywords(title, keywords or [])
            result += f" {keyword_result}"
    else:
        # Insert new prompt
        result = insert_prompt_to_db(title, author, description, system_prompt, user_prompt, keywords)

    return result


def load_prompt_details(selected_prompt):
    logging.debug(f"load_prompt_details: Loading prompt details for {selected_prompt}")
    if selected_prompt:
        details = fetch_prompt_details(selected_prompt)
        if details:
            return details[0], details[1], details[2], details[3], details[4], details[5]
    return "", "", "", "", "", ""


def update_prompt_in_db(title, author, description, system_prompt, user_prompt):
    logging.debug(f"update_prompt_in_db: Updating prompt: {title}")
    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE Prompts SET author = ?, details = ?, system = ?, user = ? WHERE name = ?",
                (author, description, system_prompt, user_prompt, title)
            )
            if cursor.rowcount == 0:
                return "No prompt found with the given title."
        return "Prompt updated successfully!"
    except sqlite3.Error as e:
        return f"Error updating prompt: {e}"


def delete_prompt(prompt_id):
    logging.debug(f"delete_prompt: Deleting prompt with ID: {prompt_id}")
    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()

            # Delete associated keywords
            cursor.execute("DELETE FROM PromptKeywords WHERE prompt_id = ?", (prompt_id,))

            # Delete the prompt
            cursor.execute("DELETE FROM Prompts WHERE id = ?", (prompt_id,))

            if cursor.rowcount == 0:
                return f"No prompt found with ID {prompt_id}"
            else:
                conn.commit()
                return f"Prompt with ID {prompt_id} has been successfully deleted"
    except sqlite3.Error as e:
        return f"An error occurred: {e}"


def delete_prompt_keyword(keyword: str) -> str:
    """
    Delete a keyword and its associations from the prompts database.

    Args:
        keyword (str): The keyword to delete

    Returns:
        str: Success/failure message
    """
    logging.debug(f"delete_prompt_keyword: Deleting keyword: {keyword}")
    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()

            # First normalize the keyword
            normalized_keyword = normalize_keyword(keyword)

            # Get the keyword ID
            cursor.execute("SELECT id FROM Keywords WHERE keyword = ?", (normalized_keyword,))
            result = cursor.fetchone()

            if not result:
                return f"Keyword '{keyword}' not found."

            keyword_id = result[0]

            # Delete keyword associations from PromptKeywords
            cursor.execute("DELETE FROM PromptKeywords WHERE keyword_id = ?", (keyword_id,))

            # Delete the keyword itself
            cursor.execute("DELETE FROM Keywords WHERE id = ?", (keyword_id,))

            # Get the number of affected prompts
            affected_prompts = cursor.rowcount

            conn.commit()

            logging.info(f"Keyword '{keyword}' deleted successfully")
            return f"Successfully deleted keyword '{keyword}' and removed it from {affected_prompts} prompts."

    except sqlite3.Error as e:
        error_msg = f"Database error deleting keyword: {str(e)}"
        logging.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error deleting keyword: {str(e)}"
        logging.error(error_msg)
        return error_msg


def export_prompt_keywords_to_csv() -> Tuple[str, str]:
    """
    Export all prompt keywords to a CSV file with associated metadata.

    Returns:
        Tuple[str, str]: (status_message, file_path)
    """
    import csv
    import tempfile
    import os
    from datetime import datetime

    logging.debug("export_prompt_keywords_to_csv: Starting export")
    try:
        # Create a temporary file with a specific name in the system's temp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f'prompt_keywords_export_{timestamp}.csv')

        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()

            # Get keywords with related prompt information
            query = '''
            SELECT 
                k.keyword,
                GROUP_CONCAT(p.name, ' | ') as prompt_names,
                COUNT(DISTINCT p.id) as num_prompts,
                GROUP_CONCAT(DISTINCT p.author, ' | ') as authors
            FROM Keywords k
            LEFT JOIN PromptKeywords pk ON k.id = pk.keyword_id
            LEFT JOIN Prompts p ON pk.prompt_id = p.id
            GROUP BY k.id, k.keyword
            ORDER BY k.keyword
            '''

            cursor.execute(query)
            results = cursor.fetchall()

            # Write to CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'Keyword',
                    'Associated Prompts',
                    'Number of Prompts',
                    'Authors'
                ])

                for row in results:
                    writer.writerow([
                        row[0],  # keyword
                        row[1] if row[1] else '',  # prompt_names (may be None)
                        row[2],  # num_prompts
                        row[3] if row[3] else ''  # authors (may be None)
                    ])

        status_msg = f"Successfully exported {len(results)} prompt keywords to CSV."
        logging.info(status_msg)

        return status_msg, file_path

    except sqlite3.Error as e:
        error_msg = f"Database error exporting keywords: {str(e)}"
        logging.error(error_msg)
        return error_msg, "None"
    except Exception as e:
        error_msg = f"Error exporting keywords: {str(e)}"
        logging.error(error_msg)
        return error_msg, "None"


def view_prompt_keywords() -> str:
    """
    View all keywords currently in the prompts database.

    Returns:
        str: Markdown formatted string of all keywords
    """
    logging.debug("view_prompt_keywords: Retrieving all keywords")
    try:
        with sqlite3.connect(get_database_path('prompts.db')) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT k.keyword, COUNT(DISTINCT pk.prompt_id) as prompt_count 
                FROM Keywords k
                LEFT JOIN PromptKeywords pk ON k.id = pk.keyword_id
                GROUP BY k.id, k.keyword
                ORDER BY k.keyword
            """)

            keywords = cursor.fetchall()
            if keywords:
                keyword_list = [f"- {k[0]} ({k[1]} prompts)" for k in keywords]
                return "### Current Prompt Keywords:\n" + "\n".join(keyword_list)
            return "No keywords found."

    except Exception as e:
        error_msg = f"Error retrieving keywords: {str(e)}"
        logging.error(error_msg)
        return error_msg


def export_prompts(
        export_format='csv',
        filter_keywords=None,
        include_system=True,
        include_user=True,
        include_details=True,
        include_author=True,
        include_keywords=True,
        markdown_template=None
) -> Tuple[str, str]:
    """
    Export prompts to CSV or Markdown with configurable options.

    Args:
        export_format (str): 'csv' or 'markdown'
        filter_keywords (List[str], optional): Keywords to filter prompts by
        include_system (bool): Include system prompts in export
        include_user (bool): Include user prompts in export
        include_details (bool): Include prompt details/descriptions
        include_author (bool): Include author information
        include_keywords (bool): Include associated keywords
        markdown_template (str, optional): Template for markdown export

    Returns:
        Tuple[str, str]: (status_message, file_path)
    """
    import csv
    import tempfile
    import os
    import zipfile
    from datetime import datetime

    try:
        # Get prompts data
        with get_prompt_db_connection() as conn:
            cursor = conn.cursor()

            # Build query based on included fields
            select_fields = ['p.name']
            if include_author:
                select_fields.append('p.author')
            if include_details:
                select_fields.append('p.details')
            if include_system:
                select_fields.append('p.system')
            if include_user:
                select_fields.append('p.user')

            query = f"""
                SELECT DISTINCT {', '.join(select_fields)}
                FROM Prompts p
            """

            # Add keyword filtering if specified
            if filter_keywords:
                placeholders = ','.join(['?' for _ in filter_keywords])
                query += f"""
                    JOIN PromptKeywords pk ON p.id = pk.prompt_id
                    JOIN Keywords k ON pk.keyword_id = k.id
                    WHERE k.keyword IN ({placeholders})
                """

            cursor.execute(query, filter_keywords if filter_keywords else ())
            prompts = cursor.fetchall()

            # Get keywords for each prompt if needed
            if include_keywords:
                prompt_keywords = {}
                for prompt in prompts:
                    cursor.execute("""
                        SELECT k.keyword
                        FROM Keywords k
                        JOIN PromptKeywords pk ON k.id = pk.keyword_id
                        JOIN Prompts p ON pk.prompt_id = p.id
                        WHERE p.name = ?
                    """, (prompt[0],))
                    prompt_keywords[prompt[0]] = [row[0] for row in cursor.fetchall()]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format == 'csv':
            # Export as CSV
            temp_file = os.path.join(tempfile.gettempdir(), f'prompts_export_{timestamp}.csv')
            with open(temp_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ['Name']
                if include_author:
                    header.append('Author')
                if include_details:
                    header.append('Details')
                if include_system:
                    header.append('System Prompt')
                if include_user:
                    header.append('User Prompt')
                if include_keywords:
                    header.append('Keywords')
                writer.writerow(header)

                # Write data
                for prompt in prompts:
                    row = list(prompt)
                    if include_keywords:
                        row.append(', '.join(prompt_keywords.get(prompt[0], [])))
                    writer.writerow(row)

            return f"Successfully exported {len(prompts)} prompts to CSV.", temp_file

        else:
            # Export as Markdown files in ZIP
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(tempfile.gettempdir(), f'prompts_export_{timestamp}.zip')

            # Define markdown templates
            templates = {
                "Basic Template": """# {title}
{author_section}
{details_section}
{system_section}
{user_section}
{keywords_section}
""",
                "Detailed Template": """# {title}

## Author
{author_section}

## Description
{details_section}

## System Prompt
{system_section}

## User Prompt
{user_section}

## Keywords
{keywords_section}
"""
            }

            template = templates.get(markdown_template, markdown_template or templates["Basic Template"])

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for prompt in prompts:
                    # Create markdown content
                    md_content = template.format(
                        title=prompt[0],
                        author_section=f"Author: {prompt[1]}" if include_author else "",
                        details_section=prompt[2] if include_details else "",
                        system_section=prompt[3] if include_system else "",
                        user_section=prompt[4] if include_user else "",
                        keywords_section=', '.join(prompt_keywords.get(prompt[0], [])) if include_keywords else ""
                    )

                    # Create safe filename
                    safe_filename = re.sub(r'[^\w\-_\. ]', '_', prompt[0])
                    md_path = os.path.join(temp_dir, f"{safe_filename}.md")

                    # Write markdown file
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)

                    # Add to ZIP
                    zipf.write(md_path, os.path.basename(md_path))

            return f"Successfully exported {len(prompts)} prompts to Markdown files.", zip_path

    except Exception as e:
        error_msg = f"Error exporting prompts: {str(e)}"
        logging.error(error_msg)
        return error_msg, "None"


create_prompts_db()

#
# End of Propmts_DB.py
#######################################################################################################################

