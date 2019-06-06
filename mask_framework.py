"""*mask_framework.py* -- Main MASK Framework module
               """

import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import importlib
import nltk
import re
from nltk.tokenize.treebank import TreebankWordTokenizer
import os
class Configuration():
    """Class for reading configuration file

        Init function that can take configuration file, or it uses default location: configuration.cnf file in folder
        where mask_framework is
    """
    def __init__(self,configuration = "configuration.cnf"):
        """Init function that can take configuration file, or it uses default location: configuration.cnf file in folder
        where mask_framework is
           """
        self.conf = configuration
        conf_doc = ET.parse(self.conf)
        root = conf_doc.getroot()
        print(root.text)
        self.entities_list = []
        for elem in root:
            if elem.tag=="project_name":
                self.project_name = elem.text
            if elem.tag=="project_start_date":
                self.project_start_date = elem.text
            if elem.tag=="project_owner":
                self.project_owner = elem.text
            if elem.tag=="project_owner_contact":
                self.project_owner_contact = elem.text
            if elem.tag=="algorithms":
                for entities in elem:
                    entity =  {}
                    for ent in entities:
                        entity[ent.tag] = ent.text
                    self.entities_list.append(entity)
            if elem.tag == "dataset":
                for ent in elem:
                    if ent.tag=="dataset_location":
                        self.dataset_location = ent.text
                    if ent.tag=="data_output":
                        self.data_output = ent.text
_treebank_word_tokenizer = TreebankWordTokenizer()

def main():
    """Main MASK Framework function
               """
    print("Welcome to MASK")
    cf = Configuration()
    print(cf.entities_list)
    data = [f for f in listdir(cf.dataset_location) if isfile(join(cf.dataset_location, f))]
    for file in data:
        text = open(cf.dataset_location+"/"+file,'r').read()
        output_text = ""
        tokens = []
        for entity in cf.entities_list:
            algorithm = "ner_plugins."+entity['algorithm']
            masking_type = entity['masking_type']
            entity_name = entity['entity_name']

            # Import the right module
            inpor = importlib.import_module(algorithm)

            # find a class and instantiate
            class_ = getattr(inpor, entity['algorithm'])

            instance = class_()

            # perform named entity recoginition
            result = instance.perform_NER(text)
            #Perform masking/redacting
            if masking_type == "Redact":
                for i in range(0,len(result)):
                    for j in range(0, len(result[i])):
                        if result[i][j][1]==entity_name:
                            if len(tokens)<i+j+1:
                                tokens.append("XXX")
                            else:
                                tokens[i+j]="XXX"
                        else:
                            if len(tokens)<i+j+1:
                                tokens.append(result[i][j][0])
            elif masking_type == "Mask":
                masking_class = entity['masking_class']
                inpor2 = importlib.import_module("masking_plugins." + masking_class)
                class_masking = getattr(inpor2, masking_class)
                masking_instance = class_masking()
                for i in range(0, len(result)):
                    for j in range(0, len(result[i])):
                        if result[i][j][1] == entity_name:
                            if len(tokens) < i + j + 1:
                                tokens.append(masking_instance.mask(result[i][j][0]))
                            else:
                                tokens[i + j] = masking_instance.mask(result[i][j][0])
                        else:
                            if len(tokens) < i + j + 1:
                                tokens.append(result[i][j][0])
            else:
                for i in range(0, len(result)):
                    for j in range(0, len(result[i])):
                        if len(tokens) < i + j + 1:
                            tokens.append(result[i][j][0])
        # create a text back from tokens
        for token in tokens:
            output_text = output_text + " "+token
            # Create target Directory if don't exist
        if not os.path.exists(cf.data_output):
            os.mkdir(cf.data_output)
        # write into output files
        f = open(cf.data_output+"/"+file,"w")
        f.write(output_text)
        f.close()









if __name__=="__main__":
    main()