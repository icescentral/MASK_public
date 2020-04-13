""" First name randomized mask """

from random import sample
from masking_plugins.Mask_abstract import Mask_abstract
from Dictionaries.populate import first_names


class MaskFirstNameRandomized(Mask_abstract):
    """This class replace the first name with randomized name from the dictionary"""
    def __init__(self):
        self.replacements = {}

    def mask(self, text_to_reduct) -> str:
        """ Replace the name with a random name from dictionary"""
        if text_to_reduct not in self.replacements.keys():
            replacement = sample(first_names(), 1)
            self.replacements[text_to_reduct]= replacement[0].title()
        return self.replacements[text_to_reduct]