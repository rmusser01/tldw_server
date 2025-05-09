# v1-endpoint-deps.py
# Description: This file is to serve as a sink for dependencies across the v1 endpoints.
# Imports
#
# 3rd-party Libraries
from fastapi.security import OAuth2PasswordBearer
#
# Local Imports
#
#######################################################################################################################
#
# Static Variables
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
#
# Functions:



#
# End of v1-endpoint-deps.py
#######################################################################################################################
