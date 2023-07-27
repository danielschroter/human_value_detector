# Discovering Human Values in Arguments

This Repo contains the best performing system (Adam-Smith) of SemEval2023 Task 4 - ValueEval: Identification of Human Values behind Arguments

Link to competition: https://touche.webis.de/semeval23/touche23-web/index.html

You can also use our Huggingface Model for simple usage at: https://huggingface.co/tum-nlp/Deberta_Human_Value_Detector 

# Set Up Project 

## Install dependencies
Create a conda environment with python 3.10 and install the required packages.
```
pip install -r requirements.txt
```
##  1. Get Data 
Get Test Data from Competition and place it in data directory: https://zenodo.org/record/7550385/files/arguments-test.tsv?download=1 

## 2. Get Models 

The trained models can be downloaded under the following link: https://drive.google.com/drive/folders/1bN7N9OwT8r35elQZlTEnBpnNHpoBl0Pz?usp=share_link

Place them in checkpoints directory. 

# Reproduce Competiton Results: The Predict Notebook
The predict.ibynb notebook reproduces the competiton results.  
Make sure you have the models downloaded and placed in the checkpoints folder together with their PARAM Files.
This Notebook is the foundation for the docker container.

# Training Procedure
If you want to understand the training-process and retrain the models yourself.
The process is split into three steps: 
1. Generate the DataSet and the Leave-Out-DataSet (data_gen.ipynb)
2. Train the Model with the configurations from the paper (train.ipynb)
3. Calculate the optimal Threshold of the Ensemble (Ensemble_eval_and_predict.ipynb)

## Data Gen
Cleans the data and creates the training and test files

## Train
Train the model train.ipynb

## Ensemble_Eval
Ensemble the individual models and create predictions in ensembling_and_predict.ipynb

# Explanations
Run explanations.ipynb to create LOCO and LIME explanations. They are visualized with the SHAP library to use them in a survey. Further we calculate the comprehensiveness scores and AOPC








