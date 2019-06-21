# ann_churn_modelling
Churn Modelling using Logistic Regression v/s Deep Learning

Using churn modelling data for bank - https://www.kaggle.com/aakash50897/churn-modellingcsv

The idea was to check model comparison, for a binominal variable of churning, using basic logistic regression (code is commented) 
and using Deep Learning using keras and tensor flow. The results were interesting. 
Basic Logit - accuracy around 80%, using DL - using two hidden layers - accuracy increased to 83% and again increased hidden layer 
by one more, accuracy increased to 80%

Second, tried Evaluating the artifical neural network using KerasClassifier and cross validation score of sickit learn. The idea was to evaluate if the variance of the model accuracy was high or not and if its changing based on datset and what is the average accuracy across 10 folds of training data and what is the variance. That would essentially mean, n epochs would for every fold.

Then also tried to have Dropout Regularization to reduce overfitting if needed. (used in the modelcomparison code)
