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
# Code by: Nikola Milosevic
class Mask_abstract(object):
    """Abstract class that other masking plugins should implement"""
    def __init__(self):
        """Implementation of initialization"""

    def mask(self,text_to_reduct,context=[],document=[],replacement_list={}):
        """Implementation of the method that should perform masking. Returns changed token

          :param text_to_reduct: a token that should be changed
          :type language: str
          :param context: a context around the token that should be reducted. It is a list of tokens. Can be sentence or more. It is optional variable and does not need to be used
          :type language: list
          :param document: a whole document as a list of tokens. Can be used as context. Optional variable and does not need to be used
          :type language: list
          :param replacement_list: list of strings with their replacements. Can be used to search for tokens or part of it, in order to replace with the value of dictionary.
          :type language: dictionary
        """
