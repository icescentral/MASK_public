import argparse
import importlib

import utils.readers
import utils.spec_tokenizers

if __name__ == "__main__":
    print("Training framework")
    parser = argparse.ArgumentParser(description='Training framework for Named Entity recognition')
    parser.add_argument('--source_type', help = 'source type of the dataset (available values: i2b2)')
    parser.add_argument('--source_location', help='source location of the dataset on your hard disk')
    parser.add_argument('--do_test', help='source location of the dataset on your hard disk')
    parser.add_argument('--algorithm', help='algorithm to use')
    args = parser.parse_args()
    path = args.source_location
    if args.source_type == "i2b2":
        documents = utils.readers.read_i2b2_data(path)
    tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)
    package = "ner_plugins."+ args.algorithm
    algorithm = args.algorithm
    inpor = importlib.import_module(package)
    # find a class and instantiate
    class_ = getattr(inpor, algorithm)
    instance = class_()
    X,Y = instance.transform_sequences(tokens_labels)

    print("Hi")
