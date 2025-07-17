import json

class Parser:
    def __init__(self, pth):
        self.pth = pth
        self.raw = json.load(open(pth, 'r'))

    def __getitem__(self, key_chain):
        # JSON parser, use . to access dict and @ to access list
        # @0, @1, @-1, @-2, etc. to access list elements
        # @ to access the whole list
        # @. to access the whole list as a dict
        keys = key_chain.split('.')
        dic = self.raw
        for k in keys:
            if '@' in k:
                parts = k.split('@')
                list_key = parts[0]
                list_res = int(parts[1]) if parts[1].isdigit() else parts[1]
                if list_key in dic:
                    dic = dic[list_key]
                elif isinstance(dic, list):
                    dic = [d[list_key] for d in dic if list_key in d]
                else:
                    raise KeyError(f"Key '{list_key}' not found in the configuration file.")
                if not isinstance(dic, list):
                    raise TypeError(f"Expected a list for key '{list_key}', but got {type(dic).__name__}.")
                # Access the list with index
                if isinstance(list_res, int) or list_res.lstrip('-').isdigit():
                    list_res = int(list_res)
                    l = list(dic)
                    dic = l[list_res]
                elif list_res == '':
                    continue
                else:
                    raise ValueError(f"Invalid index '{list_res}' for list '{list_key}'.")
            else:
                if isinstance(dic, list):
                    dic = [d[k] for d in dic if k in d]
                elif k in dic:
                    dic = dic[k]
                else:
                    raise KeyError(f"Key '{k}' not found in the configuration file.")
        return dic