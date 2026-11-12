# tldw_Server_API/app/api/v1/API_Deps/validation_deps.py
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    FileValidator,
    get_default_file_validator,
)

#
########################################################################################################################
#
#
# Rely on FileValidator to configure python-magic when available; avoid global side effects


file_validator_instance = get_default_file_validator()

def get_file_validator() -> FileValidator:
    return file_validator_instance

#
# End of validations_deps.py
#######################################################################################################################
