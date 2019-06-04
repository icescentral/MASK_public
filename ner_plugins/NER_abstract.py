
class NER_abstract(object):
    """Abstract class that other NER plugins should implement"""
    def __init__(self):
        """Implementation of initialization"""

    def perform_NER(self,text):
        """Implementation of the method that should perform named entity recognition"""