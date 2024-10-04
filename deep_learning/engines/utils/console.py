# This file contains the Console class which is used to print messages to the console with different colors
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_END = "\033[0m"

class Console(  ):
    def __init__(self, enable = False):
        self.enable = enable 

    def debug(self, message, color=ANSI_YELLOW):
        if self.enable:
            print(f'{color}{message}{ANSI_END}')
    
    def info(self, message, color=ANSI_END):
        print(f'{color}{message}{ANSI_END}')
