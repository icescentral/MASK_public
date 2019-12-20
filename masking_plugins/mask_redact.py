""" FRedact plugin mask """

from masking_plugins.Mask_abstract import Mask_abstract


class MaskRedact(Mask_abstract):
    """Redact the token with a PHI type"""
    def __init__(self):
        self.replacements = {}

    def mask(self, text_to_reduct, data_type) -> str:
        """ Replace any give value with data type, e.g. replace 12-12-2019 with DATE
            Or replace John with FIRST_NAME        
        """
        if text_to_reduct not in self.replacements.keys():
            self.replacements[text_to_reduct] = data_type
        return self.replacements[text_to_reduct]