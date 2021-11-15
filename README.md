# Sentiment-Analysis
Final Project for Neural Network and Implementation.

This repository contains code for creating word vectors and performing sentiment analysis on hate speech for Bengali and Hindi with machine learning models using PyTorch.

NNTI_final_project_task_1.ipynb = Original word embeddings using skipgram.

SGNS.ipynb = Word embeddings from Skipgram with Negative Sampling (task 3)

LSTM.py = original network for Task 2 for sentiment analysis.

DataPreprocess.py = Data cleaning and converting pandas data frame to torch text data frames.

TrainEvalLoops.py = functions to train and evaluate CNN model

model.py = CNN network for Challenge task 3 for sentiment analysis

main.py = reading file, calling network, outputting results

Data Folder = contains original csv and tsv files for Hindi and Bengali. Model weights [word embeddings] for both Hindi and Bengali, from basic skiagram and for negative sampling. Contains print statements from running our main.py various ways. Stopwords for both Hindi and Bengali.
