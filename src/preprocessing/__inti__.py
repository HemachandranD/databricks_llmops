import re
import sys

try:
    path_lists = sys.path
    path_find = [
        (x, x.find("src"))
        for x in path_lists
        if re.search("^/Workspace/Users/hema16deena@gmail.com/.*./src*", x)
    ]
    code_path = path_find[0][0][: path_find[0][1] + 4]
    sys.path.append(code_path)
except Exception as e:
    print(f"An error occurred in the path search: {str(e)}")


def initialize(name: str):
    """
    Initializes the Configuration module, printing a message to the console.

    Returns
    -------
    None
    """
    print("Configuration Module Initialized.\n")
    print("Running the " + name + " module")


def get_code_path():
    """
    Returns the path of the 'src' directory.

    Returns
    -------
    str
        The path of the 'src' directory.
    """
    return code_path
