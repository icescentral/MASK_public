import nltk
nltk.download('punkt')
from nltk.tokenize.util import align_tokens
from nltk.tokenize.treebank import TreebankWordTokenizer
import re

_treebank_word_tokenizer = TreebankWordTokenizer()
def tokenize_fa(documents):
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
        tokens = custom_span_tokenize(text)
        for token in tokens:
            token_txt = text[token[0]:token[1]]
            found = False
            if found == False:
                token_tag = "O"
                # token_tag_type = "O"
            sequence.append((token_txt, token_tag))
            if token_txt == "." or token_txt == "?" or token_txt == "!":
                sequences.append(sequence)
                sequence = []
        sequences.append(sequence)
    return sequences


def custom_span_tokenize(text, language='english', preserve_line=True):
    """
            Returns a spans of tokens in text.

            :param text: text to split into words
            :param language: the model name in the Punkt corpus
            :type language: str
            :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
            :type preserver_line: bool
            """
    tokens = custom_word_tokenize(text)
    tokens = ['"' if tok in ['``', "''"] else tok for tok in tokens]
    return align_tokens(tokens, text)

def custom_word_tokenize(text, language='english', preserve_line=False):
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
        for token in _treebank_word_tokenizer.tokenize(sent):
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