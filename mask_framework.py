"""*mask_framework.py* -- Main MASK Framework module
               """

import xml.etree.ElementTree as ET


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

def main():
    """Main MASK Framework function
               """
    print("Welcome to MASK")
    cf = Configuration()
    print(cf.entities_list)

if __name__=="__main__":
    main()