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
from masking_plugins.Mask_abstract import Mask_abstract
class Mask_date_simple(Mask_abstract):
    """Abstract class that other masking plugins should implement"""
    def __init__(self):
        """Implementation of initialization"""

    def mask(self,text):
        """Implementation of the method that should perform masking. Takes a token as input and returns a set string "DATE"

        """
        return "DATE"
