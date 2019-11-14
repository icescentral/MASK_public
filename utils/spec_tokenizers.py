import nltk
nltk.download('punkt')
from nltk.tokenize.util import align_tokens
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import tensorflow_hub as hub
#from bert.tokenization import FullTokenizer
import tensorflow as tf
sess = tf.Session()
_treebank_word_tokenizer = TreebankWordTokenizer()

def tokenize_to_seq(documents):
    sequences = []
    sequence = []
    for doc in documents:
        if len(sequence)>0:
            sequences.append(sequence)
        sequence = []
        text = doc["text"]
        file = doc["id"]
        text = text.replace("\"", "'")
        text = text.replace("`", "'")
        text = text.replace("``", "")
        text = text.replace("''", "")
        tokens = custom_span_tokenize(text)
        for token in tokens:
            token_txt = text[token[0]:token[1]]
            found = False
            for tag in doc["tags"]:
                if int(tag["start"])<=token[0] and int(tag["end"])>=token[1]:
                    token_tag = tag["tag"]
                    #token_tag_type = tag["type"]
                    found = True
            if found==False:
                token_tag = "O"
                #token_tag_type = "O"
            sequence.append((token_txt,token_tag))
            if token_txt == "." or token_txt == "?" or token_txt == "!":
                sequences.append(sequence)
                sequence = []
        sequences.append(sequence)
    return sequences


# def create_tokenizer_from_hub_module(bert_path):
#     """Get the vocab file and casing info from the Hub module."""
#     bert_module = hub.Module(bert_path)
#     tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
#     vocab_file, do_lower_case = sess.run(
#         [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
#     )
#
#     return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


# def tokenize_bert(documents):
#     bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#     tokenizer = create_tokenizer_from_hub_module(bert_path)
#     sequences = []
#     sequence = []
#     max_seq_length = 256
#     for doc in documents:
#         if len(sequence) > 0:
#             sequences.append(sequence)
#         sequence = []
#         text = doc
#         tokens_a = tokenizer.tokenize(text['text'])
#         for tok in tokens_a:
#             sequence.append(tok)
#             if tok == "." or tok == "?" or tok == "!":
#                 sequences.append(sequence)
#                 sequence = []
#         sequences.append(sequence)
#
#     output_bert_seqs= []
#     for sequence in sequences:
#         if len(sequence) > max_seq_length - 2:
#             sequence = sequence[0: (max_seq_length - 2)]
#
#         tokens = []
#         segment_ids = []
#         sequence.append("[CLS]")
#         segment_ids.append(0)
#         for token in sequence:
#             tokens.append(token)
#             segment_ids.append(0)
#         tokens.append("[SEP]")
#         segment_ids.append(0)
#         tokens2 = custom_span_tokenize(text['text'])
#         sequence_tags = []
#         for token in tokens2:
#             token_txt = text['text'][token[0]:token[1]]
#             found = False
#             for tag in doc["tags"]:
#                 if int(tag["start"]) <= token[0] and int(tag["end"]) >= token[1]:
#                     token_tag = tag["tag"]
#                     # token_tag_type = tag["type"]
#                     found = True
#             if found == False:
#                 token_tag = "O"
#             sequence_tags.append(token_tag)
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)
#
#         # Zero-pad up to the sequence length.
#         while len(input_ids) < max_seq_length:
#             input_ids.append(0)
#             input_mask.append(0)
#             segment_ids.append(0)
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         output_bert_seqs.append(input_ids, input_mask, segment_ids,sequence_tags)
#     return output_bert_seqs


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
