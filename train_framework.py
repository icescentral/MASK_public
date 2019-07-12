import argparse
import importlib

from sklearn.model_selection import train_test_split

from utils.readers import read_i2b2_data
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
        documents = read_i2b2_data(path)
    tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)
    package = "ner_plugins."+ args.algorithm
    algorithm = args.algorithm
    inpor = importlib.import_module(package)
    # find a class and instantiate
    class_ = getattr(inpor, algorithm)
    instance = class_()
    X,Y = instance.transform_sequences(tokens_labels)
    X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    instance.learn(X_train,Y_train)
    instance.evaluate(X_test,Y_test)

    print("Hi")
