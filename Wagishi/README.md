## Task 1 – Vector Distance & Angle Computations
File: Task1.py

# Overview
This module implements Three core functions for measuring distances and angles between two numerical vectors:

euclidean_dist(a, b)
manhattan_dist(a, b)
cosine_dist_angle(a, b)

## Functions

# euclidean_dist(a, b)
   Compute the Euclidean (L2) distance between two numeric vectors.
   Formula
‖a − b‖₂ = √∑ᵢ (aᵢ − bᵢ)²

# manhattan_dist(a, b)
   Compute the Manhattan (L1) distance between two numeric vectors.
   Formula
‖a − b‖₁ = ∑ᵢ |aᵢ − bᵢ|

# cosine_dist_angle(a, b)
   Compute the Cosine distance (1 − cosine similarity) and the angle (in radians) between two numeric vectors.
   Formulas
Cosine similarity:
(a · b) / (‖a‖₂ · ‖b‖₂)
Cosine distance:
1 − cosine_similarity
Angle θ:
θ = arccos(cosine_similarity)
If either vector is zero (norm = 0), the function prints a warning and returns None.
# 

## Task 2.1 – Diabetes Prediction Model
File: Task2.1_diabetes.py


# Overview
This project uses machine learning to predict diabetes risk from health metrics, utilizing the Pima Indians Diabetes Database.

# Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

# Dataset
File: diabetes.csv

Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age, Outcome (target variable)

**Code Summary**
Data Loading and Exploration: Load the dataset and check for missing values and statistics.
Visualization: Create histograms, scatter plots, KDE plots, and a heatmap for insights.
Data Splitting: Split into training (80%) and testing (20%) sets.
Scaling: Standardize features using StandardScaler.
Model Training: Fit and evaluate Logistic Regression and Linear Regression models.

 # Results
Final accuracy scores of the models will indicate performance on the test set.

## Task 2.2 - Spotify

# overview
This project uses machine learning to predict Streams of a particular song on spotify .

# Technologies Used
as above 

# Code summary 
By observing the matrix , we can conclude that there is no need of encoding the data And
1.Spotify_playlist high correlate to deezer,apple playlist and streams
2.Spotify_charts highly correlate to deezer,apple,shazam charts
3.Stream correlate with apple and deezer playlist
4.Apple playlist correlate with deezer playlist

Create Predictions DataFrame:
Construct a DataFrame, predictions_df, that includes:
artist(s)_name: The name of the artist.
track_name: The title of the track.
predicted_streams: The model's predicted stream counts.

## Output
The resulting predictions_df DataFrame will provide a clear overview of the predicted streams for each track in the test set, including artist and track names for reference.

## Task 3 - Movie Review Sentiment Analysis 

# Overview
This project performs sentiment analysis on movie reviews, classifying them as positive or negative using NLP techniques.

# Technologies Used
Python
Pandas
Matplotlib
NLTK
WordCloud
Scikit-learn

**Dataset**
File: Movie_Review.csv

**Features:**
text: Review content
sentiment: Classification (positive/negative)

**Installation**
Install required libraries:

bash

Copy Code
pip install pandas matplotlib nltk wordcloud scikit-learn

# Implementation Steps
Load Data: Import and clean the dataset.
Text Preprocessing: Remove stop words.
Visualize: Generate word clouds for positive and negative reviews.
Feature Extraction: Use TfidfVectorizer to convert text to numerical format.
Model Training: Train a Logistic Regression model on the data.
Evaluation: Use a confusion matrix to assess model performance.
Save Model: Save the trained model and vectorizer using pickle.


# Running the Code
Run the script while ensuring Movie_Review.csv is in the same directory.


### Task 4 

Neural Network for Addition README
# Overview
This project implements a simple neural network in Python, The focus is on building the forward pass and backpropagation logic manually.

# Objectives
Forward Pass: Implement data flow through the network.
Backpropagation: Develop weight adjustment logic for learning.
Network Structure: Create a neural network with a few nodes.

# Technology Used
Python
NumPy

# Implementation Plan
Week 1: Forward Pass
Build the forward propagation logic for processing inputs.
Test with addition pairs to verify output.
Week 2: Backpropagation
Implement backpropagation with a single hidden node.
Adjust weights based on prediction errors.
Complete Network
Expand to 4-5 nodes for enhanced learning.

# Running the Code
Execute the script to observe data processing and outputs. Verify the network's ability to learn addition.

### Task 5 

# Neural Style Transfer using TensorFlow

This project implements a **Neural Style Transfer (NST)** model using a pretrained **VGG19** convolutional neural network in TensorFlow.
The model generates a new image by combining the **content** of one image with the **style** of another.

Colab link - https://colab.research.google.com/drive/1Y1zUsU3NV_QXrJ2APzM24mEesbnjtzku?usp=sharing 

## Overview

Neural Style Transfer leverages the feature extraction capabilities of deep convolutional neural networks to separate and recombine image content and style.
The algorithm optimizes an input image to minimize the difference between:


## Features

* Implementation using TensorFlow and VGG19
* Supports custom content and style images
* Adjustable style and content loss weights
* Produces Nice-quality stylized images

## Usage

1. Open the notebook in Google Colab.
2. Set the runtime to use **GPU**:
   *Runtime → Change runtime type → Hardware accelerator → GPU*
3. Upload your **content** and **style** images.
4. Adjust the following parameters if needed:
5. Run all cells to generate the stylized image.

## Output
* A **stylized image** combining the content and style of the input images.
* The result is displayed using Matplotlib at the end of execution.



