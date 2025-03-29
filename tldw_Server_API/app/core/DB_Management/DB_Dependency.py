# DB_Dependency.py
# Description: Simple hack for FastAPI to use a custom DB object as a dependency.
#
# Imports
#
# 3rd-Party Imports
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.DB_Manager import db
#
########################################################################################################################
#
# Functions:

def get_db_manager():
    """
    Simple FastAPI dependency that returns the custom 'db' object.
    """
    return db

#
# End of DB_Dependency.py
########################################################################################################################
