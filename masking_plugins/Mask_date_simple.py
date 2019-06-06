from masking_plugins.Mask_abstract import Mask_abstract
class Mask_date_simple(Mask_abstract):
    """Abstract class that other masking plugins should implement"""
    def __init__(self):
        """Implementation of initialization"""

    def mask(self,text):
        """Implementation of the method that should perform masking. Takes a token as input and returns a set string "DATE"

        """
        return "DATE"