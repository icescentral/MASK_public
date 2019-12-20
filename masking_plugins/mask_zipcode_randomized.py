""" Zipcode randomized mask """


from random import randrange, choice
import string
from masking_plugins.Mask_abstract import Mask_abstract

class MaskZipcodeRandomized(Mask_abstract):
    """This class masks a given zipcode and
    returns a new one with the last three characters randomized"""
    def mask(self, text_to_reduct):
        """ change the last three characters of the zipcode with random characters"""
        last_three_chars = str(randrange(10)) + choice(string.ascii_letters) + str(randrange(10))
        return text_to_reduct[:-3] + last_three_chars
