# LLM_Base_Handler.py
#
# Imports
import abc
from pathlib import Path
import asyncio
from typing import Dict, Any
#
# Third-party imports
from loguru import logger as logging
#
# Local imports
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import BaseHandlerConfig
#
#######################################################################################################################
#
# Functions:


class BaseLLMHandler(abc.ABC):
    def __init__(self, config: BaseHandlerConfig, global_app_config: Dict[str, Any]):
        self.config = config
        self.global_app_config = global_app_config
        self.logger = logging # Use the logger from utils_loader

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        pass

    @abc.abstractmethod
    async def is_model_available(self, model_name: str) -> bool:
        pass

    # Further abstract methods can be defined for download, serve, infer, etc.
    # For now, specific handlers will implement their versions.

    async def _run_subprocess(self, cmd_list: list[str], **kwargs) -> tuple[str, str]:
        """Helper to run subprocesses asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            self.logger.error(f"Command '{' '.join(cmd_list)}' failed with code {process.returncode}")
            self.logger.error(f"Stderr: {stderr.decode().strip()}")
            raise ServerError(f"Command execution failed: {stderr.decode().strip()}")
        return stdout.decode().strip(), stderr.decode().strip()


#
# End of LLM_Base_Handler.py
########################################################################################################################
