"""
Copyright 2020 ICES, University of Manchester, Evenset Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Code by: Evenset inc.
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
