# Submission for Kaggle Titanic Challenge
## Prediction of Survival of Titanic Passengers via Conventional Feature Engineering
### Morgan Masters, Bryce Struttman

Here, Bryce and I developed an algorithm over the course of 2 days which achieved 83% accuracy in predicting the survival of passengers in the Titanic disaster. Bryce developed the pipeline from raw data to curated Pandas dataframes. My role was to engineer the features used to train our models, as well as to select appropriate models and validation strategies.

We decided to take a non-neural approach to the solution of this problem to counteract the weakness of neural networks in data-poor situations. 

In this project:

- [x] Convert non-numeric data to categorical (one-hot encoded) variables.

- [x] Scale numeric values to zero mean, unit variance.

- [x] Compare performances of dominant algorithms, such as support-vector machine classification and random forest classifiers.

- [x] Use bootstrap aggregation to refine results through averaging across the performance of many trained classifiers.

- [x] Cross-validate to gain perspective on model performance before running the model on the test set.
