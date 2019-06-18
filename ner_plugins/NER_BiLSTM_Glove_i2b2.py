from keras.engine.saving import model_from_json
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from keras_preprocessing import sequence
from utils.spec_tokenizers import tokenize_fa
import numpy as np
import pickle

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
        # load weights into new model
        self.model.load_weights("Models/BiLSTM_Glove_de_identification_model.h5")
        print("Loaded model from disk")
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.summary()
        self.word_index = pickle.load(open("Models/word_index.pkl","rb"))

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