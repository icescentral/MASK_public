"""*mask_framework.py* --
Main MASK Framework module
"""
from os import listdir, path, mkdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import importlib

import datetime
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.util import align_tokens
class Configuration():
    """Class for reading configuration file
        Init function that can take configuration file, or it uses default location:
        configuration.cnf file in folder where mask_framework is
    """
    def __init__(self, configuration="configuration.cnf"):
        """Init function that can take configuration file, or it uses default location:
            configuration.cnf file in folder where mask_framework is
        """
        self.conf = configuration
        conf_doc = ET.parse(self.conf)
        root = conf_doc.getroot()
        print(root.text)
        self.entities_list = []
        for elem in root:
            if elem.tag == "project_name":
                self.project_name = elem.text
            if elem.tag == "project_start_date":
                self.project_start_date = elem.text
            if elem.tag == "project_owner":
                self.project_owner = elem.text
            if elem.tag == "project_owner_contact":
                self.project_owner_contact = elem.text
            if elem.tag == "algorithms":
                for entities in elem:
                    entity = {}
                    for ent in entities:
                        entity[ent.tag] = ent.text
                    self.entities_list.append(entity)
            if elem.tag == "dataset":
                for ent in elem:
                    if ent.tag == "dataset_location":
                        self.dataset_location = ent.text
                    if ent.tag == "data_output":
                        self.data_output = ent.text
_treebank_word_tokenizer = TreebankWordTokenizer()

def consolidate_NER_results(final_sequences, text):
    """
    Function that from a list of sequences returned from the NER function is updated with spans
    :param final_sequences: Sequences returned from NER function. Sequence is a array of arrays of tokens in format (token,label).
    :param text: full text article
    :return: a list of tuples that includes spans in the following format: (token,label,span_begin,span_end)
    """
    tokens = []
    for a in final_sequences:
        for b in a:
            tokens.append(b[0])
    spans = align_tokens(tokens, text)
    fin = []
    multiplier = 0
    for i in range(0, len(final_sequences)):
        #multiplier = 0
        if i > 0:
            multiplier = multiplier + len(final_sequences[i-1])
            #subtractor = 1
        for j in range(0, len(final_sequences[i])):
            token = final_sequences[i][j][0]
            label = final_sequences[i][j][1]
            span_min = spans[multiplier+j][0]
            span_max = spans[multiplier+j][1]
            fin.append((token, label, span_min, span_max))
    return fin

def recalculate_tokens(token_array, index, token_size, replacement_size, new_text, new_token):
    """
    Function that recalculates token spans when the token is replaced

    :param token_array: Array of tokens with all information, including label and spans
    :param index: Index of the token in the array that is being replaced
    :param token_size: size of the token that is being replaced
    :param replacement_size: size of the new token that is replacing token
    :param new_text: whole text (have been used for debugging purposes, not obsolete and can be empty string)
    :param new_token: New string that is replacing the token.
    :return: new, modified list of tokens with information about labels and spans. Basically list of tuples (token,label,start_span,end_span)
    """
    shift = replacement_size - token_size
    new_token_array = []
    for i in range(0, len(token_array)):
        if i == index:
            new_start = token_array[i][2] #+ shift
            new_end = token_array[i][3] + shift
            tok = new_token
            new_token_array.append((tok, token_array[i][1], new_start, new_end))
        elif i > index:
            new_start = token_array[i][2] + shift
            new_end = token_array[i][3] + shift
            new_token_array.append((token_array[i][0], token_array[i][1], new_start, new_end))
        else:
            new_token_array.append(token_array[i])
    return new_token_array

def main():
    """Main MASK Framework function
               """
    print("Welcome to MASK")
    cf = Configuration()
    data = [f for f in listdir(cf.dataset_location) if isfile(join(cf.dataset_location, f))]
    algorithms = []
    # Load algorithms in data structure
    # TODO: Still optimize!
    for entity in cf.entities_list:
        algorithm = "ner_plugins." + entity['algorithm']
        masking_type = entity['masking_type']
        entity_name = entity['entity_name']
        if masking_type == "Redact":
            masking_class = ""
        else:
            masking_class = entity['masking_class']

        # Import the right module
        right_module = importlib.import_module(algorithm)

        # find a class and instantiate
        class_ = getattr(right_module, entity['algorithm'])

        instance = class_()
        algorithms.append({"algorithm":algorithm, "masking_type":masking_type, "entity_name":entity_name, "instance":instance, "masking_class":masking_class})

    mask_running_log = open('log_mask_running.log','w',encoding='utf-8')
    mask_running_log.write("Project name: "+cf.project_name+"\n")
    mask_running_log.write("Time of run: " + str(datetime.datetime.now()) + "\n\n")
    mask_running_log.write("RUN LOG \n")
    elements = []
    for file in data:
        mask_running_log.write("Running stats for file: "+file+'\n')
        text = open(cf.dataset_location+"/"+file, 'r').read()
        new_text = text
        for alg in algorithms:
            # perform named entity recoginition
            result = alg["instance"].perform_NER(new_text)
            result = consolidate_NER_results(result, new_text)
            #Perform masking/redacting

            if alg["masking_type"] == "Redact":
                for i in range(0, len(result)):
                    if result[i][1] == alg["entity_name"]:
                        token_size = result[i][3]-result[i][2]
                        old_token = result[i][0]
                        new_token = "XXX"
                        replacement_size = len(new_token)
                        new_text = new_text[:result[i][2]] + new_token+new_text[result[i][3]:]
                        result = recalculate_tokens(result, i, token_size, replacement_size, new_text, new_token)
                        elements.append(result[i][1])
                        mask_running_log.write("REDACTED ENTITY: "+result[i][1]+" -- "+old_token+' ->'+new_token+'\n')

            elif alg["masking_type"] == "Mask":
                masking_class = alg['masking_class']
                plugin_module = importlib.import_module("masking_plugins." + masking_class)
                class_masking = getattr(plugin_module, masking_class)
                masking_instance = class_masking()
                for i in range(0, len(result)):
                    if result[i][1] == alg["entity_name"]:
                        old_token = result[i][0]
                        token_size = result[i][3] - result[i][2]
                        new_token = masking_instance.mask(result[i][0])
                        replacement_size = len(new_token)
                        new_text = new_text[:result[i][2]] + new_token + new_text[result[i][3]:]
                        result = recalculate_tokens(result, i, token_size, replacement_size, new_text, new_token)
                        elements.append(result[i][1])
                        mask_running_log.write(
                            "MASKED ENTITY: " + result[i][1] + " -- " + old_token + ' ->' + new_token+'\n')
            # Create target Directory if don't exist
        if not path.exists(cf.data_output):
            mkdir(cf.data_output)
        # write into output files
        file_handler = open(cf.data_output + "/" + file, "w")
        file_handler.write(new_text)
        file_handler.close()
        for alg in algorithms:
            cnt = elements.count(alg['entity_name'])
            if alg["masking_type"] == "Mask":
                mask_running_log.write('Total masked for '+alg['entity_name']+": "+str(cnt)+'\n')
            if alg["masking_type"] == "Redact":
                mask_running_log.write('Total redacted for '+alg['entity_name']+": "+str(cnt)+'\n')
        mask_running_log.write('END for file:'+ file+'\n')
        mask_running_log.write('========================================================================')
    mask_running_log.close()


if __name__=="__main__":
    main()
