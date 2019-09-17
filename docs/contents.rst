.. MASK Framework documentation master file, created by
   sphinx-quickstart on Mon Jun  3 17:35:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MASK Framework's documentation!
==========================================

.. toctree::
   :maxdepth: 5
   :caption: Contents:

Intorduction
============

**MASK Framework is an open-source framework for de-identification of medical free-text data**

In this project, we will develop an open-source framework for automated de-identification of medical textual data. Such data contains information that can be utilized to support clinical research, but its native form contains sensitive personal identifiable information (PII) that should not be accessed by anyone who does not provide direct clinical care.

The project aims to enhance the current processes and build an open-source platform that can be used for flexible masking of personal information, ensuring that de-identified medical text still contains enough information to facilitate research.

In order to facilitate flexibility, the de-identification system has to be configurable by the user in terms of:

* Types of PII that have to be identified in free-text data;
* Approaches to masking of the identified data (keep, redact, map, etc.);
* Disclosure risk analysis that is performed on the data;
* The methodology that is applied for each of the steps.


Architectural considerations
============================
Configuration
-------------

The requirements for the configuration file are:

* Store the information about algorithms that should be used for NER
* This can be done for per entity
* Store information about masking
    * Which named entities to mask
    * How these named entities should be masked
    * There can be a choice: do not mask, map and redact
* Talk to ICES what should we implement as examples (name, postcode, age intervals)
* User can pick algorithm for mapping
* Algorithms for mapping can be added as plugins
* Mapping algorithms should be defined for each NER

Architectural consideration for extendable framework and configuration
----------------------------------------------------------------------

For named entity recognition algorithms there are following considerations:

* All implementations should be implemented in a single file as a class
* All implementations should be stored in a single folder
* All implementations should inherit same abstract class, implement method initialize (should load the models), perform_NER (takes string and returns an array of tuples with class, begin span, end span).
* They should all return a subset of defined classes (PATIENT_NAME, DOCTOR_NAME, PROFESSION, ADDRESS, CITY, COUNTRY, POST_CODE, PHONE_NUMBER, EMAIL, WEB_ADDRESS, PATIENT_ID, DOCTOR_ID, ORGANIZATION, DATE)
* Defined functions in the config file should correspond to the class and file names in this directory

For extensions related to masking functions there are following considerations:

* All implementations should be implemented in a single file as a class
* All implementations should be all stored in a single folder
* All implementations should inherit the same abstract class and implement “mask” method that takes as input string to be masked and return masked string (either mapped or redacted in a particular manner).
* Defined functions in the config file should correspond to the class and file names in this directory

Configuration file example and explanation
------------------------------------------
Example of configuration file:

.. literalinclude:: ../configuration.cnf

Explanation
-----------
The whole configuration is wrapped in <project> tag. The user can name the project (using <project_name>), and give some basic information about creator and contact details. For each entity, user would like to mask, he/she needs to create <entity> tag.

Inside <entity> tag, user has to define entity name (using entity_name tag), he can specify original name that his named entity recognizer outputs (using original_name tag), specify NER algorithm for recognition (using <algorithm> tag) and define masking. Masking can be defined by specifying masking type (using masking_type tag). Possible values for masking type are:

* Nothing - does nothing, does not redact or mask entity, but leaves it in text.
* Mask - masks entity with another string. The way of masking has to be defined with the masking_class tag.
* Redact - redacts the entity (setting either XXX or entity name - to be discussed in the future).

Dependancies
============
The system is implemented in Python, using Python 3.5.2. Dependances are defined in requirements.txt file and can be installed by running:

.. code-block:: shell

    pip3 install -r requirements.txt

List of all requirements:

.. literalinclude:: ../requirements.txt

Example
=======
Input file (with configuration presented in the example configuration):

.. literalinclude:: ../dataset/input/example1

Example output (dates are substituted with DATE and names with XXX):

.. literalinclude:: ../dataset/output/example1

Classes and functions
=====================
.. automodule:: mask_framework
    :members:
.. automodule:: ner_plugins
    :members:
.. automodule:: train_framework
    :members:
.. automodule:: masking_plugins
    :members:
.. autoclass:: mask_framework.Configuration
    :members:
.. autoclass:: ner_plugins.NER_abstract.NER_abstract
    :members:
.. autoclass:: ner_plugins.NER_BiLSTM_ELMo_i2b2.NER_BiLSTM_ELMo_i2b2
    :members:
.. autoclass:: ner_plugins.NER_CRF.NER_CRF
    :members:
.. autoclass:: ner_plugins.NER_BiLSTM_Glove_i2b2.NER_BiLSTM_Glove_i2b2
    :members:
.. autoclass:: masking_plugins.Mask_abstract.Mask_abstract
    :members:
.. autoclass:: masking_plugins.Mask_date_simple.Mask_date_simple
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
