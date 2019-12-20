""" OHIP randomized mask """

from random import randrange
from masking_plugins.Mask_abstract import Mask_abstract


class MaskOhipRandomized(Mask_abstract):
    """This class masks a given ohip and
    returns a new one with the last four characters randomized"""
    def __init__(self):
        self.replacements = {}

    def mask(self, text_to_reduct) -> str:
        """ change the last three characters of the OHIP with random characters"""
        if text_to_reduct not in self.replacements.keys():
            last_four_chars = str(randrange(1000, 9999, 1))
            self.replacements[text_to_reduct]= text_to_reduct[:-4] + last_four_chars
        return self.replacements[text_to_reduct]
