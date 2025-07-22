from utils.parse import Parser

class Designer:
    def __init__(self, config_file):
        self.config = Parser(config_file)