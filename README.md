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
With an average Matthews correlation coefficient of 0.71, 
the Random Forest class performs best, but all three classes 
provide approximately similar results with respect to the 
feature importance. Even though these classification models 
do not always perform well, a consensus on the types
of variables that are informative for a set of targets regarding 
the topics of politics and trust, attitudes toward sexual and
ethnic minorities, social behavior, religion, background, 
energy supplies and climate (change), social benefits (e.g. 
pension or child care) and employment, attitudes toward the 
European Union, and education can be distinguished from the 
results. More specifically, the results suggest the importance
of including variables employment, education level, domicile, 
and household and partner information, especially when dealing
with more fact-related variables. 

Though limitations remain in accounting for the 
researcher degrees of freedom and the missing data 
in the observed variables, this study shows that multiverse
analysis can adequately direct the process of identifying
predictors of non-response by constructing a set of models,
that classify item non-response in a set of targets, in a 
relatively flexible way due to the possibility of using
different types of models. This can benefit the successful
construction of an archive of informative predictors, due
to multiverse pipelines, like the one proposed in this study,
being easily adaptable to different contexts and purposes,
allowing researchers from different fields to contribute to
the construction of this archive.

### Dashboard
In the final dashboard effort is put into 
providing essential information in an exhaustive 
and transparent way, without compromising clarity, 
for a wide audience, ranging from beginning data science 
students to established researchers.

The code to build the dashboard can be found in the `dashboardapp` folder.
The dashboard itself can be accessed here: http://multiverseanalysisvisualizer.pythonanywhere.com/.

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



