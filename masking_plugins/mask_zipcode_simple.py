""" simple mask plugin for zipcode """

from masking_plugins.Mask_abstract import Mask_abstract
class MaskZipcodeSimple(Mask_abstract):
    """Abstract class that other masking plugins should implement"""

    def mask(self, text_to_reduct):
        """Implementation of the method that should perform masking.
            Takes a token as input and returns a set string "DATE"
        """
        return "ZIPCODE"
