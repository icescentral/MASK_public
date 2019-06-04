.. MASK Framework documentation master file, created by
   sphinx-quickstart on Mon Jun  3 17:35:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MASK Framework's documentation!
==========================================

.. toctree::
   :maxdepth: 2
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




Classes and functions
=====================
.. automodule:: mask_framework
    :members:
.. automodule:: ner_plugins
    :members:
.. autoclass:: mask_framework.Configuration
    :members:
.. autoclass:: ner_plugins.NER_CRF.NER_CRF
    :members:
.. autoclass:: ner_plugins.NER_abstract.NER_abstract
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
