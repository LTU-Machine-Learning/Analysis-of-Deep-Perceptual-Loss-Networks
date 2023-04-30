'''
This file only defines the home_path of the workspace.
This is used by other files to navigate without depending on where in the
filesystem the workspace is located.
'''
import pathlib
home_path = pathlib.Path(__file__).parent.parent.resolve()