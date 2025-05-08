# tldw_Server_API/app/api/v1/API_Deps/validation_deps.py
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator
from tldw_Server_API.app.core.Utils.Utils import logging # Your logger
#
########################################################################################################################
#
#
# Configure python-magic if a custom magic file path is provided
if settings.MAGIC_FILE_PATH:
    try:
        import magic
        magic.Magic(magic_file=settings.MAGIC_FILE_PATH) # This call might affect global state for magic
        logging.info(f"Configured python-magic with custom magic file: {settings.MAGIC_FILE_PATH}")
    except Exception as e:
        logging.error(f"Failed to configure python-magic with {settings.MAGIC_FILE_PATH}: {e}")


file_validator_instance = FileValidator(
    yara_rules_path=settings.YARA_RULES_PATH,
    # custom_media_configs can be loaded from settings too if needed
)

def get_file_validator() -> FileValidator:
    return file_validator_instance

#
# End of validations_deps.py
#######################################################################################################################
