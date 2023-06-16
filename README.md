## Welcome to the project code of "Using Multiverse Analysis for Estimating Response Models: Towards an Archive of Informative Features"
### Author
Lieve Göbbels

### Purpose 

This project code is part of the Applied Data Science Master's thesis 
written at Utrecht University under the supervision of Kyle M. Lang.


### Important components of the repo include:

#### Code to run the processing pipeline
To run the entire processing pipeline, run the following files:
- `classification_RF.py` (to run the pipeline for the Random Forest models)
- `classification_SVM.py` (to run the pipeline for the Support Vector Machine models)
- `classification_MLP.py` (to run the pipeline for the Multi-Layer Perceptron models)


#### Implementations of the multiverse approach pipelines.
This code can be found in:
- `pipeline_RF.py`
- `pipeline_SVM.py`
- `pipeline_MLP.py`

Moreover, the adapted version of the Artificial Bee Colony algorithm, inspired by Hive,
for each of the model classes can be found here: `ABC_algorithm_RF.py`, `ABC_algorithm_SVM.py`, `ABC_algorithm_MLP.py`.

### Results
add something here

### Dashboard
add something here


## General setup:

The section for it is for anyone interested in running the code.

After Fork in the repo to your local machine, follow these steps to install the Pipenv used in developing this library.

To setup the env:  
`pip install pipenv`  
`pipenv install --dev`

To install new packages:  
`pipenv install ... # some package`

To run tests:  
`pipenv run pytest`  
or  
`pipenv shell`  
`pytest`  

To enter the shell:  
`pipenv shell`  

To leave the shell:
`exit`  



