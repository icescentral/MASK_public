"""  Job title randomized mask """

from random import sample
from masking_plugins.Mask_abstract import Mask_abstract
from Dictionaries.populate import job_titles


class MaskJobRandomized(Mask_abstract):
    """This class replace the job title with randomized jobs from the dictionary"""
    def __init__(self):
        self.replacements = {}

    def mask(self, text_to_reduct) -> str:
        """ Replace the job with a random title from dictionary"""
        if text_to_reduct not in self.replacements.keys():
            replacement = sample(job_titles(), 1)
            self.replacements[text_to_reduct]= replacement[0].title()
        return self.replacements[text_to_reduct]
