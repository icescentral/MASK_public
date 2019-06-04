import sklearn_crfsuite
import pickle
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
from nltk.tokenize.util import align_tokens
from ner_plugins.NER_abstract import NER_abstract

class NER_CRF(NER_abstract):
    """
    The class for executing CRF labelling based on i2b2 dataset (2014).

    """
    def __init__(self):
        filename = 'Models/crf_baseline_model.sav'
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.05,
            max_iterations=200,
            all_possible_transitions=True
        )
        self._treebank_word_tokenizer = TreebankWordTokenizer()
        self.crf_model = pickle.load(open(filename, 'rb'))

        pass

    def custom_span_tokenize(self,text, language='english', preserve_line=True):
        """
                Returns a spans of tokens in text.

                :param text: text to split into words
                :param language: the model name in the Punkt corpus
                :type language: str
                :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
                :type preserver_line: bool
                """
        tokens = self.custom_word_tokenize(text)
        tokens = ['"' if tok in ['``', "''"] else tok for tok in tokens]
        return align_tokens(tokens, text)

    def custom_word_tokenize(self,text, language='english', preserve_line=True):
        """
        Return a tokenized copy of *text*,
        using NLTK's recommended word tokenizer
        (currently an improved :class:`.TreebankWordTokenizer`
        along with :class:`.PunktSentenceTokenizer`
        for the specified language).

        :param text: text to split into words
        :param text: str
        :param language: the model name in the Punkt corpus
        :type language: str
        :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
        :type preserver_line: bool
        """
        tokens = []
        sentences = [text] if preserve_line else nltk.sent_tokenize(text, language)
        for sent in sentences:
            for token in self._treebank_word_tokenizer.tokenize(sent):
                if "-" in token:
                    m = re.compile("(\d+)(-)([a-zA-z-]+)")
                    g = m.match(token)
                    if g:
                        for group in g.groups():
                            tokens.append(group)
                    else:
                        tokens.append(token)
                else:
                    tokens.append(token)
        return tokens
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

    def tokenize_fa(self,documents):
        """
                  Tokenization function. Returns list of sequences

                  :param documents: list of texts
                  :type language: list

                  """
        sequences = []
        sequence = []
        for doc in documents:
            if len(sequence) > 0:
                sequences.append(sequence)
            sequence = []
            text = doc
            text = text.replace("\"", "'")
            text = text.replace("`", "'")
            text = text.replace("``", "")
            text = text.replace("''", "")
            tokens = self.custom_span_tokenize(text)
            for token in tokens:
                token_txt = text[token[0]:token[1]]
                found = False
                if found == False:
                    token_tag = "O"
                    # token_tag_type = "O"
                sequence.append((token_txt, token_tag))
                if token_txt == ".":
                    sequences.append(sequence)
                    sequence = []
            sequences.append(sequence)
        return sequences

    def train(self):
        self.crf_model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.05,
            max_iterations=200,
            all_possible_transitions=True
        )
        self.crf_model.fit(self.X_train, self.y_train)
    def save_model(self,path):
        pickle.dump(self.crf_model, open(path, 'wb'))


    def perform_NER(self,text):
        """
                          Implemented function that performs named entity recognition using CRF. Returns a sequence of tuples (token,label).

                          :param text: text over which should be performed named entity recognition
                          :type language: str

                          """
        X_test = []
        documents = [text]
        sequences = self.tokenize_fa(documents)
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
        y_pred = self.crf_model.predict(X_test)
        final_sequences = []
        for i in range(0,len(y_pred)):
            sentence = []
            for j in range(0,len(y_pred[i])):
                sentence.append((word_sequences[i][j],y_pred[i][j]))
            final_sequences.append(sentence)

        return final_sequences