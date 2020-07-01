# HAABSAStar
Code for "Adversarial Training for a Hybrid Approach to Aspect-Based Sentiment Analysis". This project builds on the code from https://github.com/ofwallaart/HAABSA
and https://github.com/mtrusca/HAABSA_PLUS_PLUS.

All software is written in PYTHON3 (https://www.python.org/) and makes use of the TensorFlow framework (https://www.tensorflow.org/).

## Installation Instructions (Windows):
### Dowload required files and add them to data/externalData folder:
1. Download ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData
2. Download SemEval2015 Datasets: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
3. Download SemEval2016 Dataset: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
4. Download Glove Embeddings: http://nlp.stanford.edu/data/glove.42B.300d.zip
5. Download Stanford CoreNLP parser: https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
6. Download Stanford CoreNLP Language models: https://nlp.stanford.edu/software/stanford-english-corenlp-2018-02-27-models.jar

### Setup Environment
1. Install chocolatey (a package manager for Windows): https://chocolatey.org/install
2. Open a command prompt.
3. Install python3 by running the following command: `code(choco install python)` (http://docs.python-guide.org/en/latest/starting/install3/win/).
4. Make sure that pip is installed and use pip to install the following packages: setuptools and virtualenv (http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref).
5. Create a virtual environemnt in a desired location by running the following command: `code(virtualenv ENV_NAME)`
6. Direct to the virtual environment source directory. 
7. Unzip the HAABSA_software.zip file in the virtual environment directrory. 
8. Activate the virtual environment by the following command: 'code(Scripts\activate.bat)`.
9. Install the required packages from the requirements.txt file by running the following command: `code(pip install -r requirements.txt)`.
10. Install the required space language pack by running the following command: `code(python -m spacy download en)`

Note: the files BERT768embedding2015.txt and BERT768embedding2016.txt are too large for GitHub. These can be generated using getBERTusingColab.py.

### Configure paths

The following scripts contain file paths to adapt to your computer (this is done by adding the path to you virtual environment before the filename. For example "/path/to/venv"+"data/programGeneratedData/GloVetraindata"): main_cross.py, main_hyper.py, config.py, HyperDataMaker.py, adversarial.py.


### Run Software
1. Configure one of the three main files to the required configuration (main.py, main_cross.py, main_hyper.py)
2. Run the program from the command line by the following command: `code(python PROGRAM_TO_RUN.py)` (where PROGRAM_TO_RUN is main/main_cross/main_hyper)


## Software explanation:
The environment contains the following main files that can be run: main.py, main_cross.py, main_hyper.py
- main.py: program to run single in-sample and out-of-sample valdition runs. Each method can be activated by setting its corresponding boolean to True e.g. to run the Adversarial method set runAdversarial= True.
- main_cross.py: similar to main.py but runs a 10-fold cross validation procedure for each method.
- main_hyper.py: program that is able to do hyperparameter optimzation for a given space of hyperparamters for each method. To change a method change the objective and space parameters in the run_a_trial() function.

- config.py: contains parameter configurations that can be changed such as: dataset_year, batch_size, iterations.

- dataReader2016.py, loadData.py: files used to read in the raw data and transform them to the required formats to be used by one of the algorithms

- lcrModel.py: Tensorflow implementation for the LCR-Rot algorithm
- lcrModelAlt.py: Tensorflow implementation for the LCR-Rot-hop algorithm
- lcrModelInverse.py: Tensorflow implementation for the LCR-Rot-inv algorithm
- cabascModel.py: Tensorflow implementation for the CABASC algorithm
- OntologyReasoner.py: PYTHON implementation for the ontology reasoner
- svmModel.py: PYTHON implementation for a BoW model using a SVM.
- adversarial.py: Tensorflow implementation of adversarial training for LCR-Rot-hop

- att_layer.py, nn_layer.py, utils.py: programs that declare additional functions used by the machine learning algorithms.

## Directory explanation:
The following directories are necessary for the virtual environment setup: \__pycache, \Include, \Lib, \Scripts, \tcl, \venv
- cross_results_2015: Results for a k-fold cross validation process for the SemEval-2015 dataset
- cross_results_2016: Results for a k-fold cross validation process for the SemEval-2015 dataset
- Results_Run_Adversarial: If WriteFile = True, a csv with accuracies per iteration is saved here
- data:
	- externalData: Location for the external data required by the methods
	- programGeneratedData: Location for preprocessed data that is generated by the programs
- hyper_results: Contains the stored results for hyperparameter optimzation for each method
- results: temporary store location for the hyperopt package


## Changed files with respect to https://github.com/mtrusca/HAABSA_PLUS_PLUS:
- main.py
- main_hyper.py
- main_cross.py
- config.py
- adversarial.py (added)
