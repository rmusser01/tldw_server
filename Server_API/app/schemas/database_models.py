# Server_API/app/api/schemas/database_models.py
# Description: This code provides schema models for DB usage
#
# Imports
from datetime import datetime
from typing import Dict, List, Any, Optional
#
# 3rd-party imports
from fastapi import HTTPException
from pydantic import BaseModel
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import fetch_item_details_single
from Server_API.app.api.v1.endpoints.media import router
#
#######################################################################################################################
#
# Functions:




#
# End of database_models.py
########################################################################################################################
