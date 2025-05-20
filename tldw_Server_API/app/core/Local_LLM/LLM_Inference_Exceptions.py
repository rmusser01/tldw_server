# LLM_Inference_Exceptions.py
#
# Imports
#
# Third-party imports
#
# Local imports
#
#######################################################################################################################
#
# Classes:

class LLMInfereceLibError(Exception):
    """Base exception for this library."""
    pass

class ModelNotFoundError(LLMInfereceLibError):
    """Raised when a model is not found."""
    pass

class ModelDownloadError(LLMInfereceLibError):
    """Raised when a model download fails."""
    pass

class ServerError(LLMInfereceLibError):
    """Raised for server-related errors (start, stop, connection)."""
    pass

class InferenceError(LLMInfereceLibError):
    """Raised during model inference."""
    pass

#
# End of LLM_Inference_Exceptions.py
# ########################################################################################################################
