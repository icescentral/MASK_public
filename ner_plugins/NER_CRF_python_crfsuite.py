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
import os
from itertools import chain

import pycrfsuite
import pickle
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from ner_plugins.NER_abstract import NER_abstract
from utils.spec_tokenizers import tokenize_fa

class NER_CRF_python_crfsuite(NER_abstract):
    """
    The class for executing CRF labelling based on i2b2 dataset (2014).

    """
    def __init__(self):
        filename = 'Models/CRF_crfsuite.crfsuite'
        self.crf_model =pycrfsuite.Tagger()
        self._treebank_word_tokenizer = TreebankWordTokenizer()
        if os.path.exists(filename):
            self.crf_model.open('Models/CRF_crfsuite.crfsuite')
        else:
            self.crf_model = None

        pass

    def shape(self,word):
        shape = ""
        for letter in word:
            if letter.isdigit():
                shape = shape + "d"
            elif letter.isalpha():
                if letter.isupper():
                    shape = shape + "W"
                else:
                    shape = shape + "w"
            else:
                shape = shape + letter
        return shape

    def word2features(self,sent, i):
        """
                  Transforms words into features that are fed into CRF model

                  :param sent: a list of tokens in a single sentence
                  :param i: position of a transformed word in a given sentence (token sequence)
                  :type i: int
                  """
        word = sent[i][0]
        #postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.shape()':self.shape(word),
            'word.isalnum()':word.isalnum(),
            'word.isalpha()':word.isalpha(),
            # 'postag': postag,
            # 'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i - 1][0]
            #postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.isalnum()':word1.isalnum(),
                '-1:word.isalpha()':word1.isalpha(),
                # '-1:postag': postag1,
                # '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i > 1:
            word2 = sent[i - 2][0]
            #postag2 = sent[i - 2][1]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
                '-2:word.isdigit()': word2.isdigit(),
                '-2:word.isalnum()': word2.isalnum(),
                '-2:word.isalpha()': word2.isalpha(),
                # '-2:postag': postag2,
                # '-2:postag[:2]': postag2[:2],
            })
        else:
            features['BOS1'] = True
        if i > 2:
            word3 = sent[i - 3][0]
            #postag3 = sent[i - 3][1]
            features.update({
                '-3:word.lower()': word3.lower(),
                '-3:word.istitle()': word3.istitle(),
                '-3:word.isupper()': word3.isupper(),
                '-3:word.isdigit()': word3.isdigit(),
                '-3:word.isalnum()': word3.isalnum(),
                '-3:word.isalpha()': word3.isalpha(),
                # '-3:postag': postag3,
                # '-3:postag[:2]': postag3[:2],
            })
        else:
            features['BOS2'] = True

        if i > 3:
            word4 = sent[i - 4][0]
            #postag4 = sent[i - 4][1]
            features.update({
                '-4:word.lower()': word4.lower(),
                '-4:word.istitle()': word4.istitle(),
                '-4:word.isupper()': word4.isupper(),
                '-4:word.isdigit()': word4.isdigit(),
                '-4:word.isalnum()': word4.isalnum(),
                '-4:word.isalpha()': word4.isalpha(),
                # '-4:postag': postag4,
                # '-4:postag[:2]': postag4[:2],
            })
        else:
            features['BOS2'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.isalnum()': word1.isalnum(),
                '+1:word.isalpha()': word1.isalpha(),
                # '+1:postag': postag1,
                # '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
        if i < len(sent) - 2:
            word12 = sent[i + 2][0]
            #postag12 = sent[i + 2][1]
            features.update({
                '+2:word.lower()': word12.lower(),
                '+2:word.istitle()': word12.istitle(),
                '+2:word.isupper()': word12.isupper(),
                '+2:word.isdigit()': word12.isdigit(),
                '+2:word.isalnum()': word12.isalnum(),
                '+2:word.isalpha()': word12.isalpha(),
                # '+2:postag': postag12,
                # '+2:postag[:2]': postag12[:2],
            })
        else:
            features['EOS2'] = True
        if i < len(sent) - 3:
            word13 = sent[i + 3][0]
            #postag13 = sent[i + 3][1]
            features.update({
                '+3:word.lower()': word13.lower(),
                '+3:word.istitle()': word13.istitle(),
                '+3:word.isupper()': word13.isupper(),
                '+3:word.isdigit()': word13.isdigit(),
                '+3:word.isalnum()': word13.isalnum(),
                '+3:word.isalpha()': word13.isalpha(),
                # '+3:postag': postag13,
                # '+3:postag[:2]': postag13[:2],
            })
        else:
            features['EOS2'] = True
        if i < len(sent) - 4:
            word14 = sent[i + 4][0]
            #postag14 = sent[i + 4][1]
            features.update({
                '+4:word.lower()': word14.lower(),
                '+4:word.istitle()': word14.istitle(),
                '+4:word.isupper()': word14.isupper(),
                '+4:word.isdigit()': word14.isdigit(),
                '+4:word.isalnum()': word14.isalnum(),
                '+4:word.isalpha()': word14.isalpha(),
                # '+4:postag': postag14,
                # '+4:postag[:2]': postag14[:2],
            })
        else:
            features['EOS2'] = True
        return features

    def doc2features(self,sent):
        """
                  Transforms a sentence to a sequence of features

                  :param sent: a set of tokens that will be transformed to features
                  :type language: list

                  """
        return [self.word2features(sent['tokens'], i) for i in range(len(sent['tokens']))]

    def word2labels(self, sent):
        return sent[1]

    def sent2tokens(self,sent):
        return [token for token, postag,capitalized, label in sent]
    def prepare_features(self):
        pass

    def save_model(self,path):
        pickle.dump(self.crf_model, open(path, 'wb'))

    def transform_sequences(self,tokens_labels):
        """
        Transforms sequences into the X and Y sets. For X it creates features, while Y is list of labels
        :param tokens_labels: Input sequences of tuples (token,lable)
        :return:
        """
        X_train = []
        y_train = []
        for seq in tokens_labels:
            features_seq = []
            labels_seq = []
            for i in range(0, len(seq)):
                features_seq.append(self.word2features(seq, i))
                labels_seq.append(self.word2labels(seq[i]))
            X_train.append(features_seq)
            y_train.append(labels_seq)
        return X_train,y_train




    def learn(self,X,Y,epochs =1):
        """
        Function for training CRF algorithm
        :param X: Training set input tokens and features
        :param Y: Training set expected outputs
        :param epochs: Epochs are basically used to calculate max itteration as epochs*200
        :return:
        """
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X, Y):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 0.1,  # coefficient for L1 penalty
            'c2': 0.05,  # coefficient for L2 penalty
            'max_iterations': 250,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train('Models/CRF_crfsuite.crfsuite')
        self.crf_model = pycrfsuite.Tagger()
        self.crf_model.open('Models/CRF_crfsuite.crfsuite')


    def save(self,model_path):
        """
        Function that saves the CRF model using pickle
        :param model_path: File name in Models/ folder
        :return:
        """
        pass
        # filename = "Models/"+model_path+"1.sav"
        # pickle.dump(self.crf_model, open(filename, 'wb'))

    def evaluate(self,X,Y):
        """
        Function that takes testing data and evaluates them by making classification report matching predictions with Y argument of the function
        :param X: Input sequences of words with features
        :param Y: True labels
        :return: Prints the classification report
        """
        from sklearn import metrics
        Y_pred = [self.crf_model.tag(xseq) for xseq in X]
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(Y)))

        labels = list(lb.classes_)
        print(labels)
        # labels = list(self.crf_model.classes_)
        labels.remove('O')
        Y_pred_flat  = [item for sublist in Y_pred for item in sublist]
        Y_flat = [item for sublist in Y for item in sublist]
        print(metrics.classification_report(Y_pred_flat, Y_flat,labels))

    def perform_NER(self,text):
        """
          Implemented function that performs named entity recognition using CRF. Returns a sequence of tuples (token,label).

          :param text: text over which should be performed named entity recognition
          :type language: str

          """
        X_test = []
        documents = [text]
        sequences = tokenize_fa(documents)
        word_sequences = []
        for seq in sequences:
            features_seq = []
            labels_seq = []
            sentence = []
            for i in range(0, len(seq)):
                features_seq.append(self.word2features(seq, i))
                labels_seq.append(self.word2labels(seq[i]))
                sentence.append(seq[i][0])
            X_test.append(features_seq)
            word_sequences.append(sentence)
        y_pred = [self.crf_model.tag(xseq) for xseq in X_test]
        #y_pred = self.crf_model.tag(X_test)
        final_sequences = []
        for i in range(0,len(y_pred)):
            sentence = []
            for j in range(0,len(y_pred[i])):
                sentence.append((word_sequences[i][j],y_pred[i][j]))
            final_sequences.append(sentence)
        return final_sequences
