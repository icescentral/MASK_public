from keras import Sequential
from keras.engine.saving import model_from_json
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from keras_preprocessing import sequence
from utils.spec_tokenizers import tokenize_fa
import numpy as np
from keras_preprocessing.text import Tokenizer
import pickle
import os

class NER_BiLSTM_Glove_i2b2(object):
    """Class that implements and performs named entity recognition using BiLSTM neural network architecture. The architecture uses GloVe
    embeddings trained on common crawl dataset. Then the algorithm is trained on i2b2 2014 dataset. """
    def __init__(self):
        """Implementation of initialization"""
        # load json and create model
        json_file = open('Models/BiLSTM_Glove_de_identification_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.GLOVE_DIR = "Resources/"
        # load weights into new model
        self.model.load_weights("Models/BiLSTM_Glove_de_identification_model.h5")
        print("Loaded model from disk")
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.word_index = pickle.load(open("Models/word_index.pkl","rb"))
        self.MAX_SEQUENCE_LENGTH = 200
        self.EMBEDDING_DIM = 300
        self.MAX_NB_WORDS = 2200000

    def build_tensor2(self,sequences,numrecs,word2index,maxlen,makecategorical=False,num_classes=0,is_label=False):
        """
        Function to create tensors out of sequences

        :param sequences: Sequences of words
        :param numrecs: size of the tensor
        :param word2index: mapping between words and its numerical representation (index). Loaded from file
        :param maxlen: Maximal lenght of the sequence
        :param makecategorical: Not used
        :param num_classes: Not used
        :param is_label: Not used, leave default for action performing
        :return:
        """
        data = np.empty((numrecs,), dtype=list)
        label_index = {'O': 0}
        label_set = ["DATE", "LOCATION", "NAME", "ID", "AGE", "CONTACT", "PROFESSION", "PHI"]
        for lbl in label_set:
            label_index[lbl] = len(label_index)

        lb = LabelBinarizer()
        lb.fit(list(label_index.values()))
        i = 0
        plabels = []
        for sent in tqdm(sequences, desc='Building tensor'):
            wids = []
            pl = []
            for word, label in sent:
                if is_label == False:
                    if word in word2index:
                        wids.append(word2index[word])
                    else:
                        wids.append(word2index['the'])
                else:
                    pl.append(label_index[label])
            plabels.append(pl)
            if not is_label:
                data[i] = wids
            i += 1
        if is_label:
            plabels = sequence.pad_sequences(plabels, maxlen=maxlen)
            print(plabels.shape)
            pdata = np.array([lb.transform(l) for l in plabels])
        else:
            pdata = sequence.pad_sequences(data, maxlen=maxlen)
        return pdata

    def build_tensor(self,sequences,numrecs,word2index,maxlen,makecategorical=False,num_classes=0,is_label=False):
        """
        Function to create tensors out of sequences

        :param sequences: Sequences of words
        :param numrecs: size of the tensor
        :param word2index: mapping between words and its numerical representation (index). Loaded from file
        :param maxlen: Maximal lenght of the sequence
        :param makecategorical: Not used
        :param num_classes: Not used
        :param is_label: Not used, leave default for action performing
        :return:
        """
        data = np.empty((numrecs,),dtype=list)
        label_index = {'O': 0}
        label_set = ["DATE", "LOCATION", "NAME", "ID", "AGE", "CONTACT", "PROFESSION", "PHI"]
        for lbl in label_set:
            label_index[lbl] = len(label_index)

        lb = LabelBinarizer()
        lb.fit(list(label_index.values()))
        i = 0
        plabels = []
        for sent in tqdm(sequences, desc='Building tensor'):
            wids = []
            pl = []
            for word in sent:
                if is_label == False:
                    if word[0] in word2index:
                        wids.append(word2index[word[0]])
                    else:
                        wids.append(word2index['the'])
            plabels.append(pl)
            if not is_label:
                data[i] = wids
            i +=1
        if is_label:
            plabels = sequence.pad_sequences(plabels, maxlen=maxlen)
            print(plabels.shape)
            pdata = np.array([lb.transform(l) for l in plabels])
        else:
            pdata = sequence.pad_sequences(data, maxlen=maxlen)
        return pdata

    def transform(self,sequence):
        X = self.build_tensor(sequence, len(sequence), self.word_index, 70)
        Y = self.build_tensor(sequence, len(sequence), self.word_index, 70, True, 9,
                                 True)

    def perform_NER(self,text):
        """
        Function that perform BiLSTM-based NER

        :param text: Text that should be analyzed and tagged
        :return: returns sequence of sequences with labels
        """
        sequences = tokenize_fa([text])
        word_sequences = []
        X_test = []
        tokens = []
        for seq in sequences:
            features_seq = []
            sentence = []
            for i in range(0, len(seq)):
                features_seq.append(seq[i][0])
                tokens.append(seq[i][0])
                sentence.append(seq[i][0])
            X_test.append(sentence)
            word_sequences.append(sentence)
        tensor = self.build_tensor(sequences, len(sequences), self.word_index, 70)
        predictions = self.model.predict(tensor)
        str_pred = []
        Y_pred_F = []
        for i in range(0,len(predictions)):
            seq= []
            for j in range(0,len(predictions[i])):
                max_k = 0
                max_k_val =0
                max_str = ""
                for k in range(0,len(predictions[i][j])):
                    if predictions[i][j][k]>max_k_val:
                        max_k_val = predictions[i][j][k]
                        max_k = k
                if max_k == 0:
                    max_str = "O"
                elif max_k == 1:
                    max_str = "DATE"
                elif max_k == 2:
                    max_str = "LOCATION"
                elif max_k == 3:
                    max_str = "NAME"
                elif max_k == 4:
                    max_str = "ID"
                elif max_k == 5:
                    max_str = "AGE"
                elif max_k == 6:
                    max_str = "CONTACT"
                elif max_k == 7:
                    max_str = "PROFESSION"
                elif max_k == 8:
                    max_str = "PHI"
                seq.append(max_str)
            Y_pred_F.append(seq)
        final_sequences = []
        for j in range(0,len(Y_pred_F)):
            sentence = []
            for i in range(len(Y_pred_F[j])-len(sequences[j]),len(Y_pred_F[j])):
                sentence.append((sequences[j][i-(len(Y_pred_F[j])-len(sequences[j]))][0],Y_pred_F[j][i]))
            final_sequences.append(sentence)
        return final_sequences


    def createModel(self, text,GLOVE_DIR):
        self.embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'),encoding='utf')
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            #word = values[0]
            coefs = np.asarray(values[-300:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(self.embeddings_index))
        tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, lower=False)
        tokenizer.fit_on_texts(text)

        self.word_index = tokenizer.word_index
        pickle.dump(self.word_index,open("word_index.pkl",'wb'))

        self.embedding_matrix = np.zeros((len(self.word_index) + 1, self.EMBEDDING_DIM))
        print(self.embedding_matrix.shape)
        for word, i in self.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector

        self.embedding_layer = Embedding(len(self.word_index) + 1,
                                         self.EMBEDDING_DIM,
                                         weights=[self.embedding_matrix],
                                         input_length=70,
                                         trainable=True)
        self.model = Sequential()
        self.model.add(self.embedding_layer)
        self.model.add(Bidirectional(LSTM(150, dropout=0.3, recurrent_dropout=0.6, return_sequences=True)))#{'sum', 'mul', 'concat', 'ave', None}
        self.model.add(Bidirectional(LSTM(60, dropout=0.2, recurrent_dropout=0.5, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(9, activation='softmax')))  # a dense layer as suggested by neuralNer
        self.model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
        self.model.summary()
        pass

    def transform_sequences(self,token_sequences):
        text = []
        for ts in token_sequences:
            for t in ts:
                text.append(t[0])
        self.createModel(text, self.GLOVE_DIR)
        X = self.build_tensor2(token_sequences, len(token_sequences), self.word_index, 70)
        Y = self.build_tensor2(token_sequences, len(token_sequences), self.word_index, 70, True, 9,
                                 True)
        return X,Y
