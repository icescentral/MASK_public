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

#Code by Nikola Milosevic
class NER_abstract(object):
    """Abstract class that other NER plugins should implement"""
    def __init__(self):
        """Implementation of initialization"""

    def perform_NER(self,text):
        """Implementation of the method that should perform named entity recognition"""

    def transform_sequences(self,tokens_labels):
        """method that transforms sequences of (token,label) into feature sequences. Returns two sequence lists for X and Y"""

    def learn(self, X_train,Y_train):
        """Function that actually train the algorithm"""
    def evaluate(self, X_test,Y_test):
        """Function to evaluate algorithm"""
