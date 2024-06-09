# Predicting this year NBA Most Valuable Player (MVP)

Author: Emilien Daulhiac ([@EmilienDaulhiac](https://github.com/EmilienDaulhiac))

## Description

This repository contains the code for our project focused on predicting the NBA MVP (Most Valuable Player) for the 2023-24 season. the goal is to enhance our understanding of the factors influencing player success and MVP votes through a data-driven approach.

## Overview
In this project, we explore two distinct methods for predicting the NBA MVP:

* Regression-based Method: Predicting the MVP votes for each player through regression and ranking them accordingly.
* Convolutional Neural Network (CNN) Method: Selecting a group of eligible players and determining the MVP out of them using a convolutional network.

## Project Objectives

* Improve comprehension of the factors influencing player success and MVP votes.
* Compare the performance of the regression-based and CNN-based methods.
* Assess the model's performance on unseen data and current 2023-24 player statistics.

## Requirements

To install the required libraries, you can use pip. Run the following command:

```bash
pip install -r requirements.txt
```

## Contents
* Data/: Contains datasets used for training and testing. (NBA Players stats beween 1982 and 2022) Dataset is publicly available on ([kaggle](https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share)). The dataset was scrapped from ([Basketball Reference] (https://www.basketball-reference.com)).
* src/: All pythons script
    - src/PreProcessing : All preprocessing steps (MVP Eligibilty Criteria, Scaling, Encoding)
    - src/modeling : Function linked to train models and evaluate them
    - get_data_online : WebScrapping script to get up-to-date 2023-24 NBA players stats ([Basketball Reference] (https://www.basketball-reference.com)).
* cleaning_dataset.ipynb : jupyter notebook to dive into the process to clean the dataset
* modeling.ipynb : jupyter notebook describing the modeling process and evaluation
README.md: Overview of the project and instructions.

```
.
├── Data
│   ├── NBA_Dataset.csv
│   └── external
│       └── teams_abbreviation.json
├── README.md
├── cleaning_dataset.ipynb
├── main.py
├── modeling.ipynb
└── src
    ├── PreProcessing
    │   ├── Data_cleaning.py
    │   ├── __init__.py
    ├── get_data_online
    │   ├── __init__.py
    │   ├── baskeref_player_data_scraper.py
    │   └── html_scraper.py
    └── modeling
        ├── __init__.py
        ├── model_cnn.py
        ├── model_decison_tree.py
        ├── model_random_forest.py
        └── utils.py
```

## Run the project

At the root at the project, run the following command:

```bash
python main.py
```

This will run the following steps:
* Load training dataset and clean it
* Train all three models (Decision Tree, Random Forest, CNN)
* Test all three models on the unused part of the data
* Scrape BasketReference.com tp get up-to-date stats
* Prepare the data to match imput data of all models
* Predict this Year NBA MVP using all three models

At the end of process, three files will appear in a output folder

```
.
└── output
 ├── report_decision_tree.txt
 ├── report_test_cnn.txt
 └── report_test_random_forest.txt
```


Each file describe how the model did on the selected test season. 
For example here is what we can looks like:

```
Guess n°1  - season 1985
Success: True
Guessed MVP: Larry Bird
Odds computed: 0.966  Real odds: 0.978
Real MVP: Larry Bird
Odds computed: 0.966  Real odds: 0.978
```

The season selected, with who was the actual MVP that year aand who the model selected. This gives you an idea on how well the model is performing. 


## Notes 

Some important things to know:
- The webscrapping tool is heavily dependent on the BasketBall Reference website acrchitecture and a slight chqnge in the format might lead to errors when trying to run the code. This part might need some more works for a more robust solution.
- Chrome is neccessary to run the webscrapping tool.



