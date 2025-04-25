# logger_config.py
#
# Imports
import json
import sys
import os
from datetime import datetime
#
# 3rd-Party Imports
from loguru import logger
# Local Imports
#
#
############################################################################################################
#
# Functions:

loaded_config_data = "loaded_config_data" #FIXME - load_and_log_configs()
log_metrics_file = loaded_config_data['logging']['log_metrics_file'] or './Logs/tldw_metrics_logs.json'

def retention_function(files):
    """
    A retention function to mimic backupCount=5.

    Given a list of log file paths, this function sorts them by modification time
    and returns the list of files to be removed so that at most 5 are kept.
    """
    if len(files) > 5:
        # Sort files by modification time (oldest first)
        files.sort(key=lambda filename: os.path.getmtime(filename))
        # Remove all but the 5 most recent files.
        return files[:-5]
    return []


def json_formatter(record):
    """
    Custom JSON formatter for file logging.
    """
    try:
        # Format the log time as a string.
        dt = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")
        # Grab any extra data passed with the log.
        extra = record["extra"]

        # Handle potential non-serializable fields in 'extra'
        def serialize(value):
            if isinstance(value, datetime):
                return value.isoformat()
            return value

        log_record = {
            "time": dt,
            "levelname": record["level"].name,
            "name": record["name"],
            "message": record["message"],
            "event": extra.get("event"),
            "type": extra.get("type"),
            "value": extra.get("value"),
            "labels": extra.get("labels"),
            "timestamp": serialize(extra.get("timestamp")),
        }
        return json.dumps(log_record)
    except Exception as e:
        # Fallback to a safe JSON structure if serialization fails
        return json.dumps({
            "error": f"Log formatting failed: {str(e)}",
            "original_message": record.get("message", "")
        })


def setup_logger(args):
    """
    Sets up Loguru using command-line arguments (if provided)
    and configuration file settings.

    This function adds:
      - A console sink with a simple human‑readable format.
      - A file sink for standard logs.
      - Optionally, a file sink with JSON formatting for metrics.
    """
    # Remove any previously added sinks.
    logger.remove()

    # Determine the log level (from args; default to DEBUG)
    log_level = args.log_level.upper() if hasattr(args, "log_level") else "DEBUG"

    # Console sink with simple format
    logger.add(
        sys.stdout,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
    )

    # Determine the file sink for standard logs.
    # Prefer the command-line argument if provided; otherwise, use the config.
    if hasattr(args, "log_file") and args.log_file:
        file_log_path = args.log_file
        logger.info(f"Log file created at: {file_log_path}")
    else:
        config = "config" #FIXME - load_and_log_configs()
        file_log_path = config['logging']['log_file']
        logger.info(f"No logfile provided via command-line. Using default: {file_log_path}")

    # Ensure directory exists
    log_dir = os.path.dirname(file_log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Standard file sink
    logger.add(
        file_log_path,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
    )

    # Metrics file sink with JSON formatting
    config = "config" #FIXME - load_and_log_configs()
    metrics_log_file = config['logging'].get('log_metrics_file')
    if metrics_log_file:
        metrics_dir = os.path.dirname(metrics_log_file)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

        logger.add(
            metrics_log_file,
            level="DEBUG",
            format="{time} - {level} - {message}",  # Simple format for JSON sink
            serialize=True,  # This enables JSON serialization
            rotation="10 MB",
            # Loguru’s built-in retention can be a simple number (e.g., 5) meaning “keep 5 files”
            retention=5,
        )
    return logger

# def setup_logger(log_file_path="tldw_app_logs.json"):
#     """
#     Sets up the logger with both StreamHandler and FileHandler, formatted in JSON.
#
#     Parameters:
#         log_file_path (str): Path to the JSON log file.
#
#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     logger = logging.getLogger("tldw_app_logs")
#     logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
#
#     # Prevent adding multiple handlers if the logger is already configured
#     if not logger.handlers:
#         # StreamHandler for console output
#         stream_handler = logging.StreamHandler()
#         stream_formatter = jsonlogger.JsonFormatter(
#             '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
#         )
#         stream_handler.setFormatter(stream_formatter)
#         logger.addHandler(stream_handler)
#
#         # Ensure the directory for log_file_path exists
#         log_dir = os.path.dirname(log_file_path)
#         if log_dir and not os.path.exists(log_dir):
#             os.makedirs(log_dir, exist_ok=True)
#
#         # RotatingFileHandler for writing logs to a JSON file with rotation
#         file_handler = RotatingFileHandler(
#             log_file_path, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, keep 5 backups
#         )
#         file_formatter = jsonlogger.JsonFormatter(
#             '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
#         )
#         file_handler.setFormatter(file_formatter)
#         logger.addHandler(file_handler)
#
#     return logger
#
# # Initialize the logger
# logger = setup_logger()


#
# End of Functions
############################################################################################################
