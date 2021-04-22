# Assignment 1: Classification

## Deadlines: 
- Submission: **11:59pm, 23 April, 2021** (Friday week 7, Sydney time)
- PASTA: https://comp5318.it.usyd.edu.au/PASTA/

## Programming Language

1. Python version 3.7.0
2. Numpy: 1.18.2
3. Pandas: 1.0.3
4. Scikit-learn: 0.22.2.post1 – identical contents to 0.22.2. See: https://scikitlearn.org/stable/whats_new/v0.22.html

## Steps
### Step 1: Data Preparation
#### The data: Beast Cancer Wisconsin

- File: **breast-cancer-wisconsin.csv**. This file includes the attribute (feature) headings and each row corresponds to one individual. Missing attributes in the dataset are recorded with a ‘?’.
- *699* examples: *9* numeric attributes, *2* classes:
    + Benign breast cancer tumours: **class1**
    + Malignant breast cancer tumours: **class2**

#### Data Pre-Processing:
- Pre-process the dataset, before applying the classfication algorithms.
- *3* three types of pre-processing:
1. The missing attribute values should be replaces with the mean value of the column using **sklearn.impute.SimpleImputer**.
2. Normalisation of each attribute should be performed using a min-max scaler to normalise the values between [0,1] with **sklearn.preprocessing.MinMaxScaler**.
3. The classes **class1** and **class2** should be changed to 0 and 1 respectively.
4. The value of each attribute should be formatted to 4 decimal places using .4f.


### Step 2: Classification alogrithms with 10-fold cross-validation

- Classifiers should use the sklearn modules from the tutorials
1. Nearest Neighbor,
2. Logistic Regression,
3. Naïve Bayes,
4. Decision Tree,
5. Bagging,
6. Ada Boost
7. Gradient Boostin

- All random states in the classifiers should be set to **random_state=0**.
- Evaluate the performance of these classifiers using **10-fold cross validation** from **sklearn.model_selection.StratifiedKFold** with these options:
``` 
    cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
```

- For each classifier, write a function that accepts the required input and that outputs the average crossvalidation score:
```
    def exampleClassifier(X, y, [options]):
    … 
    return scores, scores.mean()
```
where X contains the attribute values and y contains the class (as in the tutorial exercises)

#### K-Nearest Neighbour
```
    def kNNClassifier(X, y, K)
```
It should use the KNeighboursClassifier from sklearn.neighbours.

#### Logistic Regression
```
    def logregClassifier(X, y)
```
It should use LogisticRegression from sklearn.linear_model.

#### Naïve Bayes
```
    def nbClassifier(X, y)
```
It should use GaussianNB from sklearn.naive_bayes

#### Decision Tree
```
    def dtClassifier(X, y)
```
It should use DecisionTreeClassifier from sklearn.tree, with information gain (the entropy criterion)

#### Ensembles
```
    def bagDTClassifier(X, y, n_estimators, max_samples, max_depth)
    def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth)
    def gbClassifier(X, y, n_estimators, learning_rate)
```
These functions should implement **Bagging, Ada Boost and Gradient Boosting** using *BaggingClassifier, AdaBoostClassifier and GradientBoostingClassifier* from **sklearn.ensemble**. All should combine Decision Trees with information gain.

### Step 3: Parameter Tuning

- For two other classifiers, **Linear SVM** and **Random Forest**, we would like to find the best parameters using grid search with 10-fold stratified cross validation (GridSearchCV in sklearn). 
- The split into training and test subsets should be done using train_test_split from sklearn.model_selection with stratification and random_state=0 (as in the tutorials but with random_state=0).
- *Hint*: You need to pass StratifiedKFold as an argument to GridSearchCV, not cv=10 as in the tutorials. This ensures that random_state=0.
- Write the following functions:
**Linear SVM**
```
    def bestLinClassifier(X,y)
```
It should use SVC from **sklearn.svm**.
The grid search should consider the following values for the parameters C and gamma:
```
    C = {0.001, 0.01, 0.1, 1, 10, 100}
    gamma = {0.001, 0.01, 0.1, 1, 10, 100}
```
The function should print the best parameters found, the best-cross validation accuracy score and the best test set accuracy score, see Section 4.

**Random Forest**
```
    def bestRFClassifier(X,y)
```
It should use RandomForestClassifier from sklearn.ensemble with information gain and max_features set to ‘sqrt’.
The grid search should consider the following values for the parameters n_estimators and max_leaf_nodes:
```
    n_estimators = {10, 30}
    max_leaf_nodes = {4, 16}
```
The function should print the best parameters found, best-cross validation accuracy score and best test set accuracy score, see Section 4.


### Step 4: Submission details - PASTA

#### How to submit

Your program will need to be named **MyClassifier**. If you use Jupyter notebook to write your code, please do the following:
- Ensure that you **export your Jupyter notebook as Python code** and **only submit the Python code to PASTA**
- Ensure that you have a main method that is able to **parse 3 command line arguments**. PASTA will run your code with 3 different arguments. Details about the 3 arguments are given below.
1. The *first argument* is **the path to the data file**.
2. The *second argument* is **the name of the algorithm** to be executed or the option for print the preprocessed dataset:
    a. **NN** for *Nearest Neighbour*.
    b. **LR** for *Logistic Regression*.
    c. **NB** for *Naïve Bayes*.
    d. **DT** for *Decision Tree*.
    e. **BAG** for *Ensemble Bagging DT*.
    f. **ADA** for *Ensemble ADA boosting DT*.
    g. **GB** for *Ensemble Gradient Boosting*.
    h. **RF** for *Random Forest*.
    i. **SVM** for *Linear SVM*.
    j. **P** for *printing the pre-processed dataset*.

3. The *third argument* is optional, and should only be supplied to algorithms which require parameters, namely NN, BAG, ADA and GB. It is the path to the file containing the parametervalues for the algorithm. 

The file should be formatted as a csv file like in the following examples:
a. NN (note the capital K); an example for 5-Nearest Neighbour:
```
    K 5
```
b. BAG:
```
    n_estimators,max_samples,max_depth
    100,100,2
```
c. ADA:
```
    n_estimators,learning_rate,max_depth
    100,0.2,3
```
d. GB:
```
    n_estimators,learning_rate
    100,0.2
```
For algorithms which do not require any parameters (LR, NB, DT, RF, SVM, P), the third argument should not be supplied.

The file paths (the first and third arguments) represent files that will be supplied to your program for reading. You can test your submission using any files you like, but PASTA will provide your submission with its own files for testing, so do not assume any specific filenames.

Your program must be able to correctly infer X and y from the file. Please do not hard-code the number of features and examples. For the last few tests we will use a dataset with different number of features and examples than the given breast cancer dataset. The new dataset will contain a header line and the last column will correspond to the class, as the given breast cancer dataset.

The following examples show how your program would be run:
1. We want to run the k-Nearest Neighbour classifier, the data is in a file called breast-cancerwisconsin-normalised.csv, and the parameters are stored in a file called param.csv:

```
    python MyClassifier.py breast-cancer-wisconsin-normalised.csv NN param.csv
```

2. We want to run Naïve Bayes and the data is in a file called breast-cancer-wisconsinnormalised.csv:

```
    python MyClassifier.py breast-cancer-wisconsin-normalised.csv NB
```

3. We want to run the data pre-processing task and the data is in a file called breast-cancerwisconsin.csv:

```
    python MyClassifier.py breast-cancer-wisconsin.csv P
```

Only option P assumes non-pre-processed input data (and it needs to pre-process and print this data). All other options (NN, LR, etc) assume already pre-processed data (i.e. normalized, scaled, missing values
replaced, etc) and don’t need to do any data pre-processing.

