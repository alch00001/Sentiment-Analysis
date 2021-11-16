# NNTI Final Project (Sentiment Analysis & Transfer Learning)
NNTI (WS-2021), Saarland University, Germany

This repository contains code for creating word vectors and performing sentiment analysis on hate speech for Bengali and Hindi with machine learning models using PyTorch.

NNTI_final_project_task_1.ipynb = Original word embeddings using skipgram.

SGNS.ipynb = Word embeddings from Skipgram with Negative Sampling (task 3)

LSTM.py = original network for Task 2 for sentiment analysis.

DataPreprocess.py = Data cleaning and converting pandas data frame to torch text data frames.

TrainEvalLoops.py = functions to train and evaluate CNN model

model.py = CNN network for Challenge task 3 for sentiment analysis

main.py = reading file, calling network, outputting results

Data Folder = contains original csv and tsv files for Hindi and Bengali. Model weights [word embeddings] for both Hindi and Bengali, from basic skiagram and for negative sampling. Contains print statements from running our main.py various ways. Stopwords for both Hindi and Bengali.

## Introduction
This is a final project for the course **Neural Networks: Theory and Implementation (NNTI)**. This project will introduce you to Sentiment Classification and Analysis. *Sentiment analysis* (also known as *opinion mining* or *emotion AI*) refers to the use of natural language processing, text analysis, computational linguistics, and/or biometrics to systematically identify, extract, quantify, and study affective states and subjective information. *Transfer learning* is a machine learning research problem focusing on storing knowledge gained while solving one problem and applying it to a different but related problem. In this project, we want you to create a neural sentiment classifier completely from scratch. You  first train it  on  one  type  of  dataset  and  then apply it  to  another  related  but  different dataset.  You are expected to make use of concepts that you have learnt in the lecture.  The project is divided into three tasks, the details of which you can find below.

## Repository
We have created this Github repository for this project.  You will need to -
* fork the repository into your own Github account.
* update your forked repository with your code and solutions.
* submit the report and the link to your public repository for the available code.

## Distribution of Points
The points in this project are equally distributed among the three tasks.  You will able to scorea maximum of 10 points per task, resulting to a total of 30 points in the entire project.  How the 10 points are allotted for each task, is mentioned in the guidelines.

## Task 1: Word Embeddings
Neural networks operate on numerical data and not on string or characters.  In order to train a neural network or even any machine learning model on text data, we first need to convert the text data in some form of numerical representation before feeding it to the model.  There are obviously multiple ways to do this, some of which you have come across during the course of this lecture, like the one-hot encoding method.  However, traditional methods like one-hot encoding were eventually replaced by neural Word Embeddings like Word2Vec [[1, 2](#references)] and GloVe [[3](#references)].  A *word embedding* or *word vector* is a vector representation of an input word that captures the meaning of that word in a semantic vector space.  You can find a video lecture from Stanford about Word2Vec here for better understanding. For this task, you are expected to create your own word embeddings from  scratch. You are supposed to use the HASOC Hindi [[4](#references)] sentiment dataset and train a neural network to extract word embeddings for the data. The unfinished code for this task is already in place inthe corresponding Jupyter notebook which you can find in the repository.

* Follow the instructions in the notebook, complete the code, and run it
* Save the trained word embeddings
* Update your repository with the completed notebook

## Task 2: Sentiment Classifier & Transfer Learning
In this task you are expected to reproduce ***Subtask A*** from the HASOC paper [[4](#references)] using the Hindi word embeddings from Task 1.  Then, you will apply your knowledge of transfer learning by using  your  model  from Task 1 to train Bengali word embeddings and then use the trained classifier to predict hate speech on this Bengali data set.  The data is already included in the repository.

You are expected to read some related research work (for example, encoder-decoder architecture, attention mechanism, etc.)  in neural sentiment analysis and then create an end-to-end neural network architecture for the task. After training, you should report the accuracy score of the model on test data. Follow the steps below:

* **Binary neural sentiment classifier:**  Implement a binary neural sentiment classifier for  the  Hindi  section  of  the  corpus. Use  your  word  embeddings  from  Task  1  for  that. Report the accuracy score.
* **Preprocess the Bengali data:** Split off a part of the Bengali corpus such that it roughly equals the Hindi corpus in size and distribution of classes (hatespeech/non-hatespeech). Then, apply the preprocessing pipeline from Task 1 to the new data. You can deviate from the pipeline, but should justify your decision.
* **Bengali  word  embeddings:**  Use  the  model  you  created  in  Task  1  to  create  Bengali word embeddings.
* **Apply** classifier  to  Bengali  data,  and  report  accuracy.   Retrain  your  model  with  the Bengali data.  Report the new accuracy and justify your findings. 

## Task 3: Challenge Task
In this third and final task of this project, you are expected to -

* Read multiple resources about the state-of-the-art work related to sentiment classification and analysis
* Try to come up with methodologies that would possibly improve your existing results
* Improve on the 3 accuracy scores from Task 2

Note: The task here should be a change in the model architecture, data representation, different approach, or some other similar considerable change in your process pipeline.  Please note that although you should consider fine-tuning the model hyperparameters manually, just doing that does not count as a change here.
