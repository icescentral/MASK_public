from os import listdir
import xml.etree.ElementTree as ET
from os.path import isfile, join


def read_i2b2_data(path):
    """
    Function that reads i2b2 files from path and returns a list of documents with text and tags
    :param path: Path where the files are located
    :return: list of documents containing dictionary {"id":file,"text":text,"tags":document_tags}
    """
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    documents = []
    for file in onlyfiles:
        tree = ET.parse(path+"/"+file)
        root = tree.getroot()
        document_tags = []
        for child in root:
            if child.tag == "TEXT":
                text = child.text
            if child.tag == "TAGS":
                for chch in child:
                    tag = chch.tag
                    attributes = chch.attrib
                    start = attributes["start"]
                    end = attributes["end"]
                    content = attributes["text"]
                    type= attributes["TYPE"]
                    document_tags.append({"tag":tag,"start":start,"end":end,"text":content,"type":type})
        documents.append({"id":file,"text":text,"tags":document_tags})
    return documents