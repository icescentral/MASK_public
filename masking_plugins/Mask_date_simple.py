class Mask_date_simple(object):
    """Abstract class that other masking plugins should implement"""
    def __init__(self):
        """Implementation of initialization"""

    def mask(self,text):
        """Implementation of the method that should perform masking"""
        return "DATE"