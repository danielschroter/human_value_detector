# Discovering Human Values in Arguments

This Repo contains the best performing system (Adam-Smith) of SemEval2023 Task 4 - ValueEval: Identification of Human Values behind Arguments

Link to competition: https://touche.webis.de/semeval23/touche23-web/index.html

Link to WebDemo: https://values.args.me/

Linkt to System Description Paper: https://arxiv.org/abs/2305.08625

Link to Docker Container: https://github.com/touche-webis-de/team-adam-smith23

- [1. Set-Up](#1-set-up-project)
  - [1.1 Install dependencies](#11-install-dependencies)
  - [1.2 Get Data](#12-get-data)
  - [1.3 Get Models](#13-get-models)
- [2. Reproduce Competition Results](#2-reproduce-competiton-results)
- [3. Retrain Models from Scratch]()

![Alt text](/public/image.png)

# 1. Set Up Project

## 1.1 Install dependencies

Create a conda environment with python 3.10 and install the required packages.

```
pip install -r requirements.txt
```

## 1.2 Get Data

Get Data from Competition and place it in data directory: https://zenodo.org/record/7550385

## 1.3 Get Models

The trained models can be downloaded under the following link: https://drive.google.com/drive/folders/1bN7N9OwT8r35elQZlTEnBpnNHpoBl0Pz?usp=share_link
Place them in checkpoints directory. Download models and corresponding PARAMS Files.
If you want to train the models yourself, you find the instructions in the section (#Training)

# 2. Reproduce Competiton Results

The [ensmebling_and_predict.ipynb](/ensembling_and_predict.ipynb) notebook reproduces the competiton results. In order to run it, you need to have the trained Models in place
So Make sure you have the models downloaded and placed in the checkpoints folder together with their PARAM Files (See Get Models).
This notebook is the foundation for the docker container published in the context of the data science competition.

# 3. Retrain from Scratch

If you want to understand the training-process and retrain the models yourself.
The process is split into three steps:

1. Generate the DataSet and the Leave-Out-DataSet [data_generation.ipynb](/dataset_generation.ipynb)
2. Train the Model with the configurations from the paper [train.ipynb](/train.ipynb)
3. Calculate the optimal Threshold of the Ensemble and Predict Final Submission File. (Also includes Stacking for Ensemble Variations) [Ensemble_eval_and_predict.ipynb](/ensembling_and_predict.ipynb)
