
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