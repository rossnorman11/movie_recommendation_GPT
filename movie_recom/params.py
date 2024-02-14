import os
import numpy as np

##################  VARIABLES  ##################
# Get the parent folder of the current file (goes up 2 levels)
PARENT_FOLDER_PATH = os.path.dirname(os.path.dirname(__file__))
EMBEDDING_TYPE = os.environ.get("EMBEDDING_TYPE")
SEARCH_TYPE = os.environ.get("SEARCH_TYPE")
