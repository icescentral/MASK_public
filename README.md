# Mask Project

Mask project is a collaborative project between [Institute for Clinical Evaluative Sciences](https://www.ices.on.ca/), [University of
Manchester](https://www.manchester.ac.uk/) and [Evenset Custom Medical Software Development](https://evenset.com). The purpose of this project is masking identifiable information from health related documents.

## Installation

Using a virtual environment to install project dependency is highly recommend. We're using Pipenv for python package management to ensure more stable dependency management.
[Read more](https://realpython.com/pipenv-guide/). Python 3.7 is recommended to run this package.

1. Install [Pipenv](https://docs.pipenv.org/en/latest/install/)
or simply run the following `python3 -m pip install --user pipenv`
2. The requirements file is in Pipefile, you can install dependency by running `pipenv install`
3. Sometimes lock file subprocess hangs, you can avoid it by running your installation with something like this `PIP_NO_CACHE_DIR=off pipenv install keras==2.2.4`
4. You can run python files locally on your dev machine through pipenv by `pipenv run python index.py` or activating your virtualenv by `pipenv shell`

## Contribution

### Git Branching

1. We generally follow [gitflow](https://datasift.github.io/gitflow/IntroducingGitFlow.html) for our branching and releases
2. We protect both Master and Develop branch for inadvertent code pushes

### Coding guideline

1. Please make sure to enable [editorconfig](https://editorconfig.org/) for your IDE
2. Please try to make your pull requests and tasks as small as possible so the reviewer has easier time to understand the code.
3. More documentation in your code and readme fils are always appreciated.
