[![Build Status](https://travis-ci.com/icescentral/mask.svg?token=JiqKgisBJvSwPnKWKxhV&branch=develop)](https://travis-ci.com/icescentral/mask)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mask-a-flexible-framework-to-facilitate-de/named-entity-recognition-on-i2b2-de)](https://paperswithcode.com/sota/named-entity-recognition-on-i2b2-de?p=mask-a-flexible-framework-to-facilitate-de)

# Mask Project

Mask project is a collaborative project between [Institute for Clinical Evaluative Sciences](https://www.ices.on.ca/), [University of
Manchester](https://www.manchester.ac.uk/) and [Evenset Custom Medical Software Development](https://evenset.com). The purpose of this project is masking identifiable information from health related documents.

## Installation

Using a virtual environment to install project dependency is highly recommend. We're using Pipenv for python package management to ensure more stable dependency management.
[Read more](https://realpython.com/pipenv-guide/). Python 3.7 is recommended to run this package.

1. Install [Pipenv](https://docs.pipenv.org/en/latest/install/)
or simply run the following `python3 -m pip install --user pipenv`
2. The requirements file is in Pipefile, you can install dependency by running `pipenv install` or if it doesn't work `python -m pipenv install`
3. Sometimes lock file subprocess hangs, you can avoid it by running your installation with something like this `PIP_NO_CACHE_DIR=off pipenv install keras==2.2.4`
4. You can run python files locally on your dev machine through pipenv by `pipenv run python index.py` or activating your virtualenv by `pipenv shell`
5. Install/Downgrade Keras to version 2.6.0 by running `pip install keras==2.6.0`
6. You can download base models trained on i2b2 data [here](https://drive.google.com/file/d/1h-DADgBOMC5-B3D15xRJ_nBJa19Vilrp/view?usp=sharing). Once downloaded unzip and place content into the Models folder of MASK
## Running

There are two main files that are used to run Mask:
- train_framework.py - This file is used to train named entity recognizers that will be used for masking. It contains a set if command line parameters and it uses the classes in `ner_plugins` folder. For each model, it is advisable to change a name of the model file in the appropriate file/class in `ner_plugins` folder. This is the example of how to run this script:
`python train_framework.py --source-type i2b2 --source_location "<relative location to training files in i2b2 format>" --algorithm NER_Name_of_the_algo --do_test yes --save_model yes --epochs 5`

More concretly:

`python train_framework.py --source-type i2b2 --source_location "dataset/i2b2/" --algorithm NER_CRF --do_test yes --save_model yes --epochs 5`

- mask_framework.py - This file is used to run NER and masking. As input it uses a set of text files and outputs a set of text fules. Input and output paths, as well as masking and NER algorithms are defined in configuration.cnf file.

Training of NER algorithms is at the moment supported only if in i2b2 format (other format need to be converted to this format). i2b2 2014, which have been used in development of this tool can be requested at the following location: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Publications

Please cite the following paper:
`Milosevic, N., Kalappa, G., Dadafarin, H., Azimaee, M. and Nenadic, G., 2020. MASK: A flexible framework to facilitate de-identification of clinical texts. arXiv preprint arXiv:2005.11687.` https://arxiv.org/abs/2005.11687

## Contribution

### Git Branching

1. We generally follow [gitflow](https://datasift.github.io/gitflow/IntroducingGitFlow.html) for our branching and releases
2. We protect both Master and Develop branch for inadvertent code pushes

### Coding guideline

1. Please make sure to enable [editorconfig](https://editorconfig.org/) for your IDE
2. Please try to make your pull requests and tasks as small as possible so the reviewer has easier time to understand the code.
3. More documentation in your code and readme fils are always appreciated.

### Contributors
1. [Nikola Milosevic](http://inspiratron.org/)
2. [Arina Belova](mailto:arishkabelova1999@gmail.com)
3. [Hesam Dadafarin](https://evenset.com/blog/author/admin/)
