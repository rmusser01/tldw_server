# Upload_Sink.py
# Description: Contains functions to handle file uploads and validate their safety and integrity.
#
# Imports
import os
import tempfile
#
# External Imports
#
# Local Imports
from App_Function_Libraries.Utils.Utils import load_and_log_configs, loaded_config_data, logging
#
#######################################################################################################################
#
# Functions:

# FIXME - Setup in Config
# Global variable for compiled Yara rules.
COMPILED_YARA_RULES = None

def compile_yara_rules(rules_path):
    """
    Compiles Yara rules from the given file path.
    Returns the compiled rules object or None if compilation fails.
    """
    try:
        import yara
        rules = yara.compile(filepath=rules_path)
        logging.info("Yara rules compiled successfully.")
        return rules
    except Exception as e:
        logging.error(f"Failed to compile Yara rules from {rules_path}: {e}")
        return None

def initialize_yara_scanner(rules_path):
    """
    Initializes the Yara scanner by compiling rules from a given file.
    This should be called at application startup.
    Returns True if successful, False otherwise.
    """
    global COMPILED_YARA_RULES
    try:
        import yara
    except ImportError:
        logging.error("Yara module not installed. Skipping Yara scanning.")
        return False

    COMPILED_YARA_RULES = compile_yara_rules(rules_path)
    if COMPILED_YARA_RULES is None:
        logging.error("Yara scanner initialization failed.")
        return False
    return True

def scan_file_with_yara(file_path, rules):
    """
    Scans the file using the provided Yara rules.
    Returns True if no matches are found (file is safe), False if any rule matches.
    """
    try:
        matches = rules.match(file_path)
        if matches:
            logging.error(f"Yara rule matched for file: {file_path}. Matches: {matches}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error scanning file with Yara: {e}")
        return False

def scan_file_for_malware(file_path):
    """
    Scans the file for malware using integrated Yara rule scanning if available.
    Returns True if the file passes scanning, False if a potential threat is detected.
    """
    try:
        import yara
    except ImportError:
        logging.warning("Yara module not installed. Skipping Yara scanning.")
        return True

    global COMPILED_YARA_RULES
    if COMPILED_YARA_RULES:
        if not scan_file_with_yara(file_path, COMPILED_YARA_RULES):
            return False

    # Additional malware scanning can be added here.
    return True

def validate_file(file_path, allowed_extensions, allowed_mimetypes=None):
    """
    Validate that a file exists, is non-empty, has an allowed extension, and optionally an allowed MIME type.
    Also performs a malware scan including Yara rule scanning.
    """
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        return False

    if os.path.getsize(file_path) == 0:
        logging.error(f"File is empty: {file_path}")
        return False

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        logging.error(f"Invalid file extension: {ext}. Allowed: {allowed_extensions}")
        return False

    try:
        import magic  # python-magic for MIME type detection
        mimetype = magic.from_file(file_path, mime=True)
        if allowed_mimetypes and mimetype not in allowed_mimetypes:
            logging.error(f"Unexpected MIME type: {mimetype}. Allowed: {allowed_mimetypes}")
            return False
    except ImportError:
        logging.warning("python-magic not installed. Skipping MIME type validation.")

    if not scan_file_for_malware(file_path):
        logging.error("Malware detected in file.")
        return False

    return True


########################################################
#
# Audio Upload & Processing Functions

def process_audio_file(file_path):
    allowed_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg'}
    allowed_mimetypes = {
        'audio/mpeg',
        'audio/wav',
        'audio/x-wav',
        'audio/flac',
        'audio/aac',
        'audio/ogg'
    }
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional audio-specific processing here.
    logging.info("Audio file validated successfully.")
    return True

#
# End of Audio Upload & Processing Functions
########################################################


########################################################
#
# Epub/Ebook Upload & Processing Functions

def process_ebook_file(file_path):
    allowed_extensions = {'.epub', '.mobi', '.azw', '.pdf'}  # ebooks may also be PDFs
    allowed_mimetypes = {
        'application/epub+zip',
        'application/pdf'
        # Extend with additional MIME types as needed.
    }
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional ebook-specific processing here.
    logging.info("Ebook file validated successfully.")
    return True

#
# End of Epub/Ebook Upload & Processing Functions
########################################################


########################################################
#
# HTML Upload & Processing Functions

def process_html_file(file_path):
    allowed_extensions = {'.html', '.htm'}
    allowed_mimetypes = {'text/html'}
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional HTML-specific processing here.
    logging.info("HTML file validated successfully.")
    return True

#
# End of HTML Upload & Processing Functions
########################################################


########################################################
#
# Office Documents Upload & Processing Functions

def process_office_document(file_path):
    allowed_extensions = {'.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
    allowed_mimetypes = {
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional office document processing here.
    logging.info("Office document validated successfully.")
    return True

#
# End of Office Document Upload & Processing Functions
########################################################


########################################################
#
# OPML Upload & Processing Functions

def process_opml_file(file_path):
    allowed_extensions = {'.opml', '.xml'}  # OPML commonly uses .opml; sometimes .xml is used
    allowed_mimetypes = {'text/xml', 'application/xml'}
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional OPML-specific processing here.
    logging.info("OPML file validated successfully.")
    return True

#
# End of OPML Upload & Processing Functions
########################################################


########################################################
#
# Plaintext Upload & Processing Functions

def process_text_file(file_path):
    allowed_extensions = {'.txt', '.csv', '.md'}
    allowed_mimetypes = {'text/plain', 'text/csv', 'text/markdown'}
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional plaintext-specific processing here.
    logging.info("Plaintext file validated successfully.")
    return True

#
# End of Plaintext Upload & Processing Functions
########################################################


########################################################
#
# PDF file Upload & Processing Functions

def process_pdf_file(file_path):
    allowed_extensions = {'.pdf'}
    allowed_mimetypes = {'application/pdf'}
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional PDF-specific processing here.
    logging.info("PDF file validated successfully.")
    return True

#
# End of PDF file Upload & Processing Functions
########################################################


########################################################
#
# XML Upload & Processing Functions

def process_xml_file(file_path):
    allowed_extensions = {'.xml'}
    allowed_mimetypes = {'text/xml', 'application/xml'}
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional XML-specific processing here.
    logging.info("XML file validated successfully.")
    return True

#
# End of XML Upload & Processing Functions
########################################################


########################################################
#
# Video Upload & Processing Functions

def process_video_file(file_path):
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    allowed_mimetypes = {
        'video/mp4',
        'video/x-msvideo',
        'video/quicktime',
        'video/x-matroska'
    }
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Insert additional video-specific processing here.
    logging.info("Video file validated successfully.")
    return True

#
# End of Video Upload & Processing Functions
########################################################


########################################################
#
# ZIP file Upload & Processing Functions

def process_zip_file(file_path):
    allowed_extensions = {'.zip'}
    allowed_mimetypes = {'application/zip', 'application/x-zip-compressed'}
    if not validate_file(file_path, allowed_extensions, allowed_mimetypes):
        return False

    # Optionally, scan ZIP contents for disallowed files or malicious content.
    logging.info("ZIP file validated successfully.")
    return True

#
# End of ZIP file Upload & Processing Functions
########################################################


def process_file_upload(file_path):
    """
    Main entry point. Determines file type by extension and dispatches to the corresponding processing function.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in {'.mp3', '.wav', '.flac', '.aac', '.ogg'}:
        return process_audio_file(file_path)
    elif ext in {'.txt', '.csv', '.md'}:
        return process_text_file(file_path)
    elif ext in {'.mp4', '.avi', '.mov', '.mkv'}:
        return process_video_file(file_path)
    elif ext == '.pdf':
        return process_pdf_file(file_path)
    elif ext in {'.epub', '.mobi', '.azw'}:
        return process_ebook_file(file_path)
    elif ext in {'.html', '.htm'}:
        return process_html_file(file_path)
    elif ext in {'.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}:
        return process_office_document(file_path)
    elif ext == '.opml':
        return process_opml_file(file_path)
    elif ext == '.xml':
        # Additional inspection can be added if needed to differentiate XML from OPML.
        return process_xml_file(file_path)
    elif ext == '.zip':
        return process_zip_file(file_path)
    else:
        logging.error(f"Unsupported file type: {ext}")
        return False




#
# End of Upload_Sink.py
#######################################################################################################################
