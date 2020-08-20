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
