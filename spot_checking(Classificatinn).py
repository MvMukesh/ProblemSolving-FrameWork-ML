''' 
# Spot-Check is Pick out random samples for examination in order to ensure high quality algo(Classification)

Spot-Checking  on   Classification

Spot-checking algorithms is about getting a quick assessment of a bunch of different algorithms on machine learning problem to know what algorithms to focus on and what to discard.

*********

Benefits of spot-checking algorithms on machine learning problems:
    - Speed
    - Objective
    - Results
    
*********    
    
Dataset Used : Pima Indians onset of Diabetes
Test harness : 10-fold cross validation [To demonstrate how to spot-check ML algorithm]
Performance Evaluation[algo.]: mean accuracy error(MAE)
'''

''' # Algorithm Overview
## Start with 2 linear ML algorithms:
   - LogistricRegression()
   - LinearDiscriminantAnalysis()

# Then look at 4 non-linear machine learning  algorithms:
   - kNN  :    KNeighborsClassifier()
   - Naive Bayes  :   GaussianNB()
   - Classification and Regression Trees   :   DecisionTreeClassifier()
   - SVM   :   SVC()
'''

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
kfold = KFold(n_splits=10, random_state=7)




''' 1. Logistic Regression '''
# 
# LR ==> assumes a Gaussian distribution for numeric input variables and can model binary classification problems
# 
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv = kfold)

print(results.mean())
#   0.76951469583



''' 2. Linear Discriminant Analysis '''
# LDA ==> a statistical technique for binary and multiclass classification
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv= kfold)

print(results.mean())
#   0.773462064252



''' 3. kNN '''
# KNN ==> uses a distance metric to find k most similar instances in training data for a new instance and takes mean outcome of neighbors as prediction
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv = kfold)

print(results.mean)
# 0.726555023923


''' 4. Naive Bayes '''
#
# NB ==> calculates probability of each class and conditional probability of each class given each input value
#   These probabilities are estimated for new data and multiplied together, assuming that they are all independent (Naive assumption)
#
model = GaussianNB()

results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())
#  0.75517771702



''' 5. CART/decision tree '''
#
# DT ==> construct a binary tree from training data.
#
model = DecisionTreeClassifier()

results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())
# 0.686056049214



''' 6. Support Vector Machine'''
#
# SVM ==> seek a line that best separates two classes. 
# Use of differnet kernel functions via kernel parameter is important
# By default, Radial Basis Function is used
#
model = SVC()

results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())
#   0.651025290499
