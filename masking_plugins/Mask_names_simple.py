from masking_plugins.Mask_abstract import Mask_abstract
import random

class Mask_names_simple(Mask_abstract):
    def __init__(self):
        self.names = ["Joe","Mary","Elton"]
        self.replacements = {}
    def mask(self,text_to_reduct):
        if text_to_reduct in self.replacements.keys():
            return self.replacements[text_to_reduct]
        else:
            self.replacements[text_to_reduct] = random.choice(self.names)
            return self.replacements[text_to_reduct]
