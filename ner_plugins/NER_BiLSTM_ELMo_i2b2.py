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
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda
from utils.spec_tokenizers import tokenize_fa


class NER_BiLSTM_ELMo_i2b2(object):
    """Class that implements and performs named entity recognition using BiLSTM
    neural network architecture.
    The architecture uses GloVe embeddings trained on common crawl dataset.
    Then the algorithm is trained on i2b2 2014 dataset. """
    def __init__(self):
        """Implementation of initialization"""
        # load json and create model
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

        self.max_len = 50
        self.batch_size = 32
        self.n_tags = 9
        self.model = self.createModel("", "")
        if os.path.exists("Models/NER_BiLSTM_ELMo.h5"):
            print("Loading model")
            self.model.load_weights("Models/NER_BiLSTM_ELMo.h5")
            print("Loaded model")
        self.GLOVE_DIR = ""
        self.MAX_SEQUENCE_LENGTH = 200
        self.EMBEDDING_DIM = 300
        self.MAX_NB_WORDS = 2200000
        self.tags = None


    def perform_NER(self, text):
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

        X = []

        remaining = len(word_sequences)%32
        additional_seq = 32 - remaining
        for i in range(0,additional_seq):
            X_seq = []
            for i in range(0,self.max_len):
                X_seq.append("PADword")
            word_sequences.append(X_seq)

        for tok_seq in word_sequences:
            X_seq = []
            for i in range(0, self.max_len):
                try:
                    X_seq.append(tok_seq[i])
                except:
                    X_seq.append("PADword")
            X.append(X_seq)
        for i in range(len(word_sequences), 32):
            X_seq = []
            for i in range(0, self.max_len):
                X_seq.append("PADword")
            X.append(X_seq)
        index2tags = {0:'O', 1:'ID', 2:'PHI', 3:'NAME', 4:'CONTACT',
                      5:'DATE', 6:'AGE', 7:'PROFESSION', 8:'LOCATION'}
        predictions = self.model.predict([X])
        Y_pred_F = []
        for i in range(0, len(word_sequences)):
            seq = []
            for j in range(0, len(word_sequences[i])):
                max_k = 0
                max_k_val = 0
                if j>=50:
                    continue
                for k in range(0, len(predictions[i][j])):
                    if predictions[i][j][k] > max_k_val:
                        max_k_val = predictions[i][j][k]
                        max_k = k
                max_str = index2tags[max_k]
                seq.append(max_str)
            Y_pred_F.append(seq)
        final_sequences = []
        for j in range(0, len(Y_pred_F)):
            sentence = []
            if j>=len(sequences):
                continue
            for i in range(len(Y_pred_F[j])-len(sequences[j]), len(Y_pred_F[j])):
                sentence.append((sequences[j][i-(len(Y_pred_F[j])-len(sequences[j]))][0], Y_pred_F[j][i]))
            final_sequences.append(sentence)
        return final_sequences

    def ElmoEmbedding(self, x):
        return self.elmo_model(
            inputs={"tokens": tf.squeeze(tf.cast(x, tf.string)), "sequence_len": tf.constant(self.batch_size * [self.max_len])
                    },
            signature="tokens",
            as_dict=True)["elmo"]

    def createModel(self, text, GLOVE_DIR):

        input_text = Input(shape=(self.max_len,), dtype="string")
        embedding = Lambda(self.ElmoEmbedding, output_shape=(self.max_len, 1024))(input_text)
        x = Bidirectional(LSTM(units=512, return_sequences=True,
                               recurrent_dropout=0.2, dropout=0.2))(embedding)
        x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                                   recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(self.n_tags, activation="softmax"))(x)
        self.model = Model(input_text, out)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        self.model.summary()
        return self.model

    def transform_sequences(self, token_sequences):
        text = []
        for ts in token_sequences:
            for t in ts:
                text.append(t[0])

        X = []
        Y = []
        all_tags = []
        for tok_seq in token_sequences:
            X_seq = []
            Y_seq = []
            for i in range(0, self.max_len):
                try:
                    X_seq.append(tok_seq[i][0])
                    Y_seq.append(tok_seq[i][1])
                    all_tags.append(tok_seq[i][1])
                except:
                    X_seq.append("PADword")
                    Y_seq.append("O")
            X.append(X_seq)
            Y.append(Y_seq)
        self.n_tags = len(set(all_tags))
        self.tags = set(all_tags)
        tags2index = {'O':0, 'ID':1, 'PHI':2, 'NAME':3, 'CONTACT':4,
                      'DATE':5, 'AGE':6, 'PROFESSION':7, 'LOCATION':8}


        Y = [[tags2index[w] for w in s] for s in Y]

        return X, Y

    def learn(self, X, Y, epochs=1):
        """
        Method for the training ELMo BiLSTM NER model
        :param X: Training sequences
        :param Y: Results of training sequences
        :param epochs: number of epochs
        :return:
        """
        first = int(np.floor(0.9*len(X)/self.batch_size))
        second = int(np.floor(0.1*len(X)/self.batch_size))
        X_tr, X_val = X[:first * self.batch_size], X[-second * self.batch_size:]
        y_tr, y_val = Y[:first * self.batch_size], Y[-second * self.batch_size:]
        y_tr = np.array(y_tr)
        y_val = np.array(y_val)
        y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)
        self.model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                       batch_size=self.batch_size, epochs=epochs)

    def evaluate(self, X, Y):
        """
        Function that evaluates the model and calculates precision, recall and F1-score
        :param X: sequences that should be evaluated
        :param Y: true positive predictions for evaluation
        :return: prints the table with precision,recall and f1-score
        """
        first = int(np.floor(int(len(X) / self.batch_size)) * self.batch_size)
        X = X[:first]
        Y = Y[:first]
        Y_pred = self.model.predict(np.array(X))
        from sklearn import metrics
        index2tags = {0:'O', 1:'ID', 2:'PHI', 3:'NAME', 4:'CONTACT',
                      5:'DATE', 6:'AGE', 7:'PROFESSION', 8:'LOCATION'}
        labels = ["ID", "PHI", "NAME", "CONTACT", "DATE", "AGE",
                  "PROFESSION", "LOCATION"]
        Y_pred_F = []
        for i in range(0, len(Y_pred)):
            for j in range(0, len(Y_pred[i])):
                max_k = 0
                max_k_val = 0
                for k in range(0, len(Y_pred[i][j])):
                    if Y_pred[i][j][k] > max_k_val:
                        max_k_val = Y_pred[i][j][k]
                        max_k = k
                Y_pred_F.append(index2tags[max_k])
        Y_test_F = []
        for i in range(0, len(Y)):
            for j in range(0, len(Y[i])):
                Y_test_F.append(index2tags[Y[i][j]])

        print(metrics.classification_report(Y_pred_F, Y_test_F, labels=labels))

    def save(self, model_path):
        """
        Function to save model. Models are saved as h5 files in Models directory. Name is passed as argument
        :param model_path: Name of the model file
        :return: Doesn't return anything
        """
        self.model.save("Models/"+model_path+".h5")
        print("Saved model to disk")
