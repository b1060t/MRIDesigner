from utils.parse import Parser
import process.magnet

class Designer:
    def __init__(self, config_file):
        self.config = Parser(config_file)

    def interalEval(self, params):
        pass

    def externalEval(self, params):
        pass

    def getObj(params):
        pass

class HalbachDesigner(Designer):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.type = self.config['type']
        self.algos = self.config['algo'][0]
        self.meta = self.config['meta']

    def getObj(self,params):
        super().getObj(params)
        halbach = process.magnet.Halbach(params, self.meta)
        zpos = self.meta['size'] / 2
