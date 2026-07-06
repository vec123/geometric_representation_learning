import inspect
from pathlib import Path

def get_project_root():
    """
    Returns the absolute path of the parent directory of the 
    folder containing this script (the file where this function resides).
    """
    # __file__ is the path to this current script
    # resolve() makes it an absolute path
    # .parent gets the folder containing this file
    # .parent.parent gets the parent of that folder
    return Path(__file__).resolve().parent.parent