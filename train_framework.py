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
"""
    *train_framework.py* - Trains algorithm of selection
    Example of starting: python train_framework.py --source_type i2b2 --source_location "../../NERo/Datasets/i2b2_data/training-PHI-Gold-Set1/" --algorithm NER_CRF_dictionaries --do_test yes --save_model yes
    Code by: Nikola Milosevic
"""

import argparse
import importlib

from sklearn.model_selection import train_test_split

from utils.readers import read_i2b2_data
import utils.spec_tokenizers

if __name__ == "__main__":

    """
    Trains algorithm of selection
    """

    print("Training framework")
    parser = argparse.ArgumentParser(description='Training framework for Named Entity recognition')
    parser.add_argument('--source_type', help = 'source type of the dataset (available values: i2b2)')
    parser.add_argument('--source_location', help='source location of the dataset on your hard disk')
    parser.add_argument('--do_test', help='source location of the dataset on your hard disk')
    parser.add_argument('--save_model',help='Whether to save the model on HDD in Models folder, with algorithm name')
    parser.add_argument('--algorithm', help='algorithm to use')
    parser.add_argument('--epochs',help='number of epochs or iteration to train')
    args = parser.parse_args()
    path = args.source_location
    documents= None
    if args.source_type == "i2b2":
        documents = read_i2b2_data(path)
    if documents== None:
        print("Error: No input source is defined")
        exit(2)
    # if "bert" in args.algorithm.lower():
    #     tokens_labels = utils.spec_tokenizers.tokenize_bert(documents)
    #     pass
    #else:

    tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)
    package = "ner_plugins."+ args.algorithm
    algorithm = args.algorithm
    inpor = importlib.import_module(package)
    # find a class and instantiate
    class_ = getattr(inpor, algorithm)
    instance = class_()
    X,Y = instance.transform_sequences(tokens_labels)
    if args.do_test == "yes":
        X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

        instance.learn(X_train,Y_train,int(args.epochs))
        #instance.save(args.algorithm)

        instance.evaluate(X_test,Y_test)
    else:
        instance.learn(X, Y)

    if args.save_model=='yes':
        instance.save(args.algorithm)

    print("Done!")
