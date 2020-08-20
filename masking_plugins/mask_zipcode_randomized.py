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
