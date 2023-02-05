# Exploratory Projects 

In this github repo, I aim to create a AI/ML project every week.

This week I'm creating a simple neural network model to classify the Hotel reservation based on hotel bookings. 

## Context
The online hotel reservation channels have dramatically changed booking possibilities and customers’ behavior. A significant number of hotel reservations are called-off due to cancellations or no-shows. The typical reasons for cancellations include change of plans, scheduling conflicts, etc. This is often made easier by the option to do so free of charge or preferably at a low cost which is beneficial to hotel guests but it is a less desirable and possibly revenue-diminishing factor for hotels to deal with.

## Aim
To predict if the customer is going to honor the reservation or cancel it ?

Credits to https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset for providing the dataset(click for more info). 

## Setting up project

### Downloading Kaggle dataset
1. conda create envronment with `requirements.txt` and pip install `requirements_pip.txt`
2. You need to create your kaggle account and download your credential `kaggle.json` file
3. run bash file (kaggle datasets download -d ahsan81/hotel-reservations-classification-dataset, unzip into dataset folder)

## Run project

1. To run project, `python3 main.py`
